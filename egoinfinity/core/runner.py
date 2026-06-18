"""PipelineRunner — topo-sort + resume + cascade + --force.

Reads a ``PipelineConfig``, instantiates the configured Stage classes
via the registry, and runs them in declared order. Skips any Stage that
``is_done()`` per the auto resume policy.

CLI ``--force <stage>`` and ``--cascade`` are honored via the
``ResumeConfig.force_stages`` + ``cascade_force`` fields (set by
``cli/process.py`` before calling ``run()``).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .config import PipelineConfig, StageConfig
from .registry import _import_stages_module, get_stage
from .stage import Stage, StageContext
from .state import ClipState


log = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        _import_stages_module()    # populate the registry

    def run(
        self,
        clip_id: str,
        *,
        manifest: dict | None = None,
        artifacts_root: Path | None = None,
    ) -> int:
        """Back-compat single-clip alias for :meth:`run_clip`."""
        return self.run_clip(
            clip_id, manifest=manifest, artifacts_root=artifacts_root)

    def run_many(
        self,
        specs,
        *,
        keep_going: bool = True,
        stages: list[str] | None = None,
        with_deps: bool = False,
    ) -> int:
        """Run a batch of clips. Per-clip failures are isolated; returns 0
        iff every clip succeeded. With ``keep_going=False`` it stops at the
        first failing clip.

        ``stages`` (if given) runs only that stage subset per clip via
        :meth:`run_stages` (force-run + prereq check) instead of the full
        resume pipeline."""
        n = len(specs)
        if n == 0:
            log.warning("no clips to process")
            return 0
        results: list[tuple[str, int]] = []
        for i, spec in enumerate(specs, 1):
            log.info("==== clip %d/%d: %s ====", i, n, spec.clip_id)
            try:
                if stages is not None:
                    rc = self.run_stages(
                        spec.clip_id,
                        stages,
                        artifacts_root=spec.artifacts_root,
                        with_deps=with_deps,
                    )
                else:
                    rc = self.run_clip(
                        spec.clip_id,
                        manifest=spec.manifest,
                        artifacts_root=spec.artifacts_root,
                    )
            except Exception:
                log.exception("clip %s crashed before/around the stage loop", spec.clip_id)
                rc = 1
            results.append((spec.clip_id, rc))
            if rc != 0 and not keep_going:
                log.error("clip %s failed; stopping (--fail-fast)", spec.clip_id)
                break
        n_ok = sum(1 for _, rc in results if rc == 0)
        n_fail = len(results) - n_ok
        log.info("batch done: %d/%d clips ok, %d failed", n_ok, n, n_fail)
        if n_fail:
            log.info("failed clips: %s", [c for c, rc in results if rc != 0])
        return 0 if n_fail == 0 else 1

    def run_clip(
        self,
        clip_id: str,
        *,
        manifest: dict | None = None,
        artifacts_root: Path | None = None,
    ) -> int:
        root = Path(artifacts_root) if artifacts_root else self.config.artifacts_dir
        store = ArtifactStore(root, clip_id)
        if manifest is not None:
            store.manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True))
        state = ClipState(store)

        # Instantiate enabled stages
        stage_instances: list[tuple[Stage, StageConfig]] = []
        for s_cfg in self.config.pipeline:
            if not s_cfg.enabled:
                log.info("[%s] disabled by config — skip", s_cfg.name)
                continue
            try:
                cls = get_stage(s_cfg.name, s_cfg.backend)
            except KeyError as e:
                # Stages registered in later phases — silent skip at info level.
                log.debug("[%s] no registered impl (%s) — skip", s_cfg.name, e)
                continue
            stage_instances.append((cls(), s_cfg))

        # Apply force_stages
        if self.config.resume.force_stages:
            downstream = self._build_downstream_map(stage_instances)
            for forced in self.config.resume.force_stages:
                if self.config.resume.cascade_force:
                    log.info("[force-cascade] forgetting %s + downstream", forced)
                    state.cascade_forget(forced, downstream)
                else:
                    log.info("[force] forgetting %s", forced)
                    state.forget(forced)

        # Run in declared order
        n_total = len(stage_instances)
        n_skipped = n_run = n_failed = 0
        t_pipeline_start = time.time()
        for i, (stage, s_cfg) in enumerate(stage_instances, 1):
            ctx = StageContext(
                clip_id=clip_id,
                artifacts=store,
                state=state,
                config=self.config,
                stage_config=s_cfg,
                log=logging.getLogger(f"egoinfinity.{stage.name}"),
            )
            prefix = f"[{i}/{n_total}] {stage.name}"
            if self.config.resume.policy == "auto" and stage.is_done(ctx):
                log.info("%s SKIP (done)", prefix)
                n_skipped += 1
                continue
            log.info("%s RUN (backend=%s)", prefix, s_cfg.backend)
            state.mark_running(stage.name, backend=s_cfg.backend, args=s_cfg.args)
            t0 = time.time()
            try:
                stage.run(ctx)
            except Exception:
                log.exception("%s FAILED", prefix)
                state.forget(stage.name)
                n_failed += 1
                return 1
            outputs_hash = {o: store.hash(stage.name, o) for o in stage.outputs}
            duration = time.time() - t0
            state.mark_done(
                stage.name,
                backend=s_cfg.backend,
                args=s_cfg.args,
                duration_s=duration,
                outputs=outputs_hash,
            )
            log.info("%s done (%.1fs)", prefix, duration)
            n_run += 1

        total = time.time() - t_pipeline_start
        log.info("pipeline done in %.1fs — ran=%d skipped=%d failed=%d",
                 total, n_run, n_skipped, n_failed)
        return 0

    # ── Explicit stage-subset run (the `egoinfinity run` path) ───────────────

    def _ctx(self, store, state, s_cfg) -> StageContext:
        return StageContext(
            clip_id=store.clip_id,
            artifacts=store,
            state=state,
            config=self.config,
            stage_config=s_cfg,
            log=logging.getLogger(f"egoinfinity.{s_cfg.name}"),
        )

    def _execute_one(self, stage: Stage, s_cfg: StageConfig, store, state) -> int:
        """Run one stage (mark running/done, hash outputs). Returns 0/1."""
        ctx = self._ctx(store, state, s_cfg)
        state.mark_running(stage.name, backend=s_cfg.backend, args=s_cfg.args)
        t0 = time.time()
        try:
            stage.run(ctx)
        except Exception:
            log.exception("[%s] FAILED", stage.name)
            state.forget(stage.name)
            return 1
        outputs_hash = {o: store.hash(stage.name, o) for o in stage.outputs}
        state.mark_done(
            stage.name,
            backend=s_cfg.backend,
            args=s_cfg.args,
            duration_s=time.time() - t0,
            outputs=outputs_hash,
        )
        log.info("[%s] done (%.1fs)", stage.name, time.time() - t0)
        return 0

    def run_stages(
        self,
        clip_id: str,
        requested: list[str],
        *,
        artifacts_root: Path | None = None,
        with_deps: bool = False,
    ) -> int:
        """Force-run an explicit subset of stages on ONE existing clip, in
        declared (DAG) order.

        Unlike :meth:`run_clip`, a named stage runs regardless of its
        ``enabled`` flag (you asked for it explicitly). Before running, every
        selected stage's ``upstream`` must already be done OR also selected;
        otherwise this errors (return 2) unless ``with_deps=True``, which
        auto-runs the missing upstream first.
        """
        root = Path(artifacts_root) if artifacts_root else self.config.artifacts_dir
        store = ArtifactStore(root, clip_id)
        state = ClipState(store)

        # All configured stages (regardless of enabled — explicit run), in order.
        cfg_by_name: dict[str, StageConfig] = {c.name: c for c in self.config.pipeline}
        declared_order = [c.name for c in self.config.pipeline]

        unknown = [n for n in requested if n not in cfg_by_name]
        if unknown:
            log.error("unknown stage(s): %s (not in %s)", unknown, "configs/defaults.yaml")
            return 2

        _inst: dict[str, tuple[Stage, StageConfig]] = {}

        def inst(name: str) -> tuple[Stage, StageConfig]:
            if name not in _inst:
                c = cfg_by_name[name]
                _inst[name] = (get_stage(c.name, c.backend)(), c)
            return _inst[name]

        def is_up_done(name: str) -> bool:
            if name in cfg_by_name:
                st, c = inst(name)
                return st.is_done(self._ctx(store, state, c))
            return state.is_done(name)   # upstream not configured: trust state.json

        requested_set = set(requested)
        run_set = list(requested)

        # with_deps: pull in transitive upstream that isn't done yet.
        if with_deps:
            stack = list(requested)
            in_set = set(run_set)
            while stack:
                st, _ = inst(stack.pop())
                for up in st.upstream:
                    if up not in cfg_by_name or up in in_set:
                        continue
                    if not is_up_done(up):
                        run_set.append(up)
                        in_set.add(up)
                        stack.append(up)

        run_set_set = set(run_set)

        # Prereq check: every to-run stage's upstream must be in run_set or done.
        missing: list[tuple[str, str]] = []
        for name in run_set:
            st, _ = inst(name)
            for up in st.upstream:
                if up in run_set_set or is_up_done(up):
                    continue
                missing.append((name, up))
        if missing:
            for child, up in missing:
                log.error("%s needs upstream %s (not done)", child, up)
            log.error("run the missing upstream first, or pass --with-deps to auto-run it.")
            return 2

        # Run in declared order. Explicitly-requested stages are force-run
        # (forget first); with_deps-added upstream runs only if not already done.
        n_ran = 0
        for name in declared_order:
            if name not in run_set_set:
                continue
            stage, s_cfg = inst(name)
            if name in requested_set:
                state.forget(name)                       # explicit -> always re-run
            elif stage.is_done(self._ctx(store, state, s_cfg)):
                log.info("[%s] SKIP (done, dep already satisfied)", name)
                continue
            log.info("[%s] RUN (backend=%s)", name, s_cfg.backend)
            rc = self._execute_one(stage, s_cfg, store, state)
            if rc != 0:
                return 1
            n_ran += 1
        log.info("ran %d stage(s) on %s", n_ran, clip_id)
        return 0

    @staticmethod
    def _build_downstream_map(
        stage_instances: list[tuple[Stage, Any]],
    ) -> dict[str, list[str]]:
        """{stage_name: [stages_that_list_it_as_upstream]}."""
        out: dict[str, list[str]] = {}
        names = [s.name for s, _ in stage_instances]
        instances_by_name = {s.name: s for s, _ in stage_instances}
        for n in names:
            out[n] = []
        for child in stage_instances:
            stage, _ = child
            for up in stage.upstream:
                if up in out:
                    out[up].append(stage.name)
        return out
