"""Pipeline + per-stage config dataclasses with YAML loading.

Choice B1: Python dataclass + light YAML loader (no Hydra). Type-safe,
IDE-friendly, programmatic composition.

Usage::

    cfg = PipelineConfig.from_yaml("configs/defaults.yaml")
    cfg = cfg.with_overrides(["depth.backend=depthanything", "depth.args.tta=true"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    """One entry in the pipeline list."""
    name: str
    backend: str = "default"
    enabled: bool = True            # False = skip unconditionally; True = run per resume policy
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "StageConfig":
        # Accept enabled: "optional" as a documentation alias for True (kept enabled,
        # but stage is treated as "may be skipped if its inputs absent"). The
        # actual skipping logic lives in the runner; the config just records intent.
        enabled = d.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.lower() not in ("false", "no", "off", "disabled")
        return cls(
            name=d["name"],
            backend=d.get("backend", "default"),
            enabled=bool(enabled),
            args=dict(d.get("args", {})),
        )


@dataclass
class ResumeConfig:
    """How the runner decides which stages to skip."""
    policy: str = "auto"                              # "auto" = skip stages whose outputs exist; "never" = full re-run
    force_stages: list[str] = field(default_factory=list)
    cascade_force: bool = False                        # if a forced stage's downstream should also re-run

    @classmethod
    def from_dict(cls, d: dict) -> "ResumeConfig":
        return cls(
            policy=d.get("policy", "auto"),
            force_stages=list(d.get("force_stages", [])),
            cascade_force=bool(d.get("cascade_force", False)),
        )


@dataclass
class PipelineConfig:
    """Top-level pipeline config."""
    artifacts_dir: Path
    pipeline: list[StageConfig]
    resume: ResumeConfig = field(default_factory=ResumeConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            artifacts_dir=Path(d.get("artifacts_dir", "./artifacts")),
            pipeline=[StageConfig.from_dict(s) for s in d.get("pipeline", [])],
            resume=ResumeConfig.from_dict(d.get("resume", {})),
        )

    def stage(self, name: str) -> StageConfig | None:
        for s in self.pipeline:
            if s.name == name:
                return s
        return None

    def with_overrides(self, overrides: list[str]) -> "PipelineConfig":
        """Apply CLI ``--set 'stage.field=value'`` overrides; returns a new config.

        Supported paths:
            <stage>.backend=<value>
            <stage>.enabled=<bool>
            <stage>.args.<key>=<value>
            artifacts_dir=<path>
            resume.policy=<value>
            resume.cascade_force=<bool>
        """
        # Deep-copy via dict round-trip
        new = PipelineConfig.from_dict(_to_dict(self))
        for o in overrides:
            if "=" not in o:
                raise ValueError(f"override must be 'path=value', got: {o!r}")
            path, value = o.split("=", 1)
            _apply_override(new, path.strip(), value.strip())
        return new


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_dict(cfg: PipelineConfig) -> dict:
    return {
        "artifacts_dir": str(cfg.artifacts_dir),
        "pipeline": [
            {"name": s.name, "backend": s.backend, "enabled": s.enabled, "args": dict(s.args)}
            for s in cfg.pipeline
        ],
        "resume": {
            "policy": cfg.resume.policy,
            "force_stages": list(cfg.resume.force_stages),
            "cascade_force": cfg.resume.cascade_force,
        },
    }


def _parse_value(v: str) -> Any:
    """Best-effort YAML-style scalar parse for CLI override values."""
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    if v.lower() in ("null", "none", "~"):
        return None
    # Try int / float
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v   # string


def _apply_override(cfg: PipelineConfig, path: str, value: str) -> None:
    parts = path.split(".")
    pv = _parse_value(value)

    if path == "artifacts_dir":
        cfg.artifacts_dir = Path(value)
        return

    if parts[0] == "resume":
        if len(parts) != 2:
            raise ValueError(f"resume override needs 1 sub-key: {path}")
        setattr(cfg.resume, parts[1], pv)
        return

    # stage-level: <stage>.<field>[.<sub>]=value
    stage = cfg.stage(parts[0])
    if stage is None:
        raise ValueError(f"unknown stage in override: {parts[0]!r}")
    if len(parts) == 2:
        if parts[1] == "args":
            raise ValueError(f"use 'args.<key>=value' to set a single arg, not 'args=...'")
        setattr(stage, parts[1], pv)
        return
    if parts[1] == "args" and len(parts) >= 3:
        # args.<key>[.<sub>...]
        key_path = parts[2:]
        target = stage.args
        for k in key_path[:-1]:
            target = target.setdefault(k, {})
        target[key_path[-1]] = pv
        return
    raise ValueError(f"unsupported override path: {path}")
