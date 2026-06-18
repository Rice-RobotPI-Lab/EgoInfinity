"""RetargetStage — bridge to retarget/scripts/test.py (charlierkj's module).

The retarget code lives under ``retarget/`` and is maintained
independently (separate conda env, separate ckpts, separate
dependencies — MuJoCo 3.6, MJX, pytorch-kinematics, etc.). This stage
does NOT call retarget functions directly; instead it subprocess-invokes
``retarget/scripts/test.py`` after the pkl has been adapted into
SamplesSequence directory format by the upstream
``export_retarget_samples`` stage.

Inputs (per clip):

    <fav_dir>/retarget_samples/
    ├── hand_joints.bin       (T, max_h, 21, 3) float32
    ├── hand_meta.json
    ├── scene.json
    └── depth.mp4

Outputs (per clip, per robot):

    <fav_dir>/retarget/<robot>/
    ├── trajectory.npz        Final joint trajectories
    ├── input_viz.mp4         Hand-keypoint overlay on depth.mp4
    ├── robot_sim.mp4         MuJoCo offscreen render of retargeted robot
    └── metrics.npz           Per-frame IK convergence + per-window stats

Backends == robots (each ckpt lives at ``retarget/ckpts/<robot>.pt``):

    g1          — Unitree G1
    franka      — Franka FR3 (bimanual)
    robonaut2   — NASA Robonaut2
    xlerobot    — Xlerobot bimanual

Environment requirements
------------------------

Retarget inference needs mujoco 3.6 + pytorch-kinematics + roma + torch.
These coexist with the main pipeline deps in the single unified
``egoinfinity`` conda env (validated under torch 2.6 + PyOpenGL 3.1.10;
note PyOpenGL must be >= 3.1.10 for mujoco's EGL renderer). So by default
this stage runs in the CURRENT interpreter (``_resolve_retarget_python``
returns ``sys.executable`` when mujoco/torch/pytorch_kinematics are
importable). Overrides:

    RETARGET_PYTHON   force a specific interpreter (e.g. a legacy separate env)
    RETARGET_REPO     point to a non-sibling retarget checkout

(Training — scripts/train.py — additionally needs jax[cuda12] + mujoco-mjx;
those are NOT required for inference and are not part of the unified env.)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from ..core.registry import register_stage
from ..core.stage import Stage, StageContext


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

SUPPORTED_ROBOTS = ("g1", "franka", "robonaut2", "xlerobot")


def _current_interp_has_retarget_deps() -> bool:
    """True if the CURRENT interpreter can run retarget (mujoco + torch +
    pytorch_kinematics importable). In the unified `egoinfinity` env the
    pipeline and retarget share one env, so we just run in-process-env."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("mujoco", "torch", "pytorch_kinematics"))


def _resolve_retarget_python() -> str:
    """Find the Python interpreter for retarget.

    Priority:
      1. RETARGET_PYTHON env var (explicit override)
      2. the CURRENT interpreter, if it already has the retarget deps
         (the unified `egoinfinity` env case — pipeline + retarget in one env)
      3. ~/miniconda3/envs/egoinfinity/bin/python (legacy separate env)
      4. fall back to current sys.executable
    """
    p = os.environ.get("RETARGET_PYTHON", "").strip()
    if p:
        return p
    if _current_interp_has_retarget_deps():
        return sys.executable
    cands = [
        Path.home() / "miniconda3" / "envs" / "egoinfinity" / "bin" / "python",
        Path("/opt/conda/envs/egoinfinity/bin/python"),
    ]
    for c in cands:
        if c.is_file():
            return str(c)
    return sys.executable


def _resolve_retarget_repo() -> Path:
    p = os.environ.get("RETARGET_REPO", "").strip()
    if p:
        return Path(p)
    return REPO_ROOT / "retarget"


class _RetargetStage(Stage):
    """Generic retarget stage. The robot name comes from the backend slot."""

    name: ClassVar[str] = "retarget"
    upstream: ClassVar[tuple[str, ...]] = ("export_retarget_samples",)
    outputs: ClassVar[tuple[str, ...]] = ("trajectory.npz",)
    robot: ClassVar[str] = ""

    def run(self, ctx: StageContext) -> None:
        clip_root = ctx.artifacts.root
        samples_dir = clip_root / "retarget_samples"
        if not samples_dir.is_dir():
            raise RuntimeError(
                f"retarget_samples/ not found at {samples_dir}. "
                f"Run the upstream export_retarget_samples stage first.")

        robot = self.robot or ctx.stage_config.backend or "g1"
        if robot not in SUPPORTED_ROBOTS:
            raise RuntimeError(
                f"retarget: unsupported robot {robot!r}. "
                f"Supported: {SUPPORTED_ROBOTS}")

        retarget_repo = _resolve_retarget_repo()
        if not retarget_repo.is_dir():
            raise RuntimeError(
                f"retarget repo not found at {retarget_repo}. "
                f"Set RETARGET_REPO env var or clone retarget under "
                f"{REPO_ROOT}/retarget.")

        ckpt = retarget_repo / "ckpts" / f"{robot}.pt"
        if not ckpt.is_file():
            raise RuntimeError(
                f"retarget ckpt not found at {ckpt}. "
                f"See retarget/README.md for download / training instructions.")

        out_dir = clip_root / "retarget" / robot
        out_dir.mkdir(parents=True, exist_ok=True)

        retarget_py = _resolve_retarget_python()
        test_script = retarget_repo / "scripts" / "test.py"

        # Pre-check: retarget env must have mujoco importable. Fail with a
        # clear message instead of a cryptic ModuleNotFoundError mid-run.
        precheck = subprocess.run(
            [retarget_py, "-c", "import mujoco, torch"],
            capture_output=True, text=True,
        )
        if precheck.returncode != 0:
            raise RuntimeError(
                f"interpreter {retarget_py} is missing retarget deps "
                f"(mujoco / torch). The unified egoinfinity env needs:\n"
                f"    pip install mujoco==3.6.0 pytorch-kinematics==0.9.1 roma trimesh\n"
                f"    pip install 'PyOpenGL>=3.1.10'   # mujoco EGL renderer\n"
                f"or set RETARGET_PYTHON to a fully-configured interpreter. "
                f"See README.md (Retarget) + {retarget_repo}/README.md.\n"
                f"Underlying error:\n  {precheck.stderr.strip()}")

        cmd = [
            retarget_py, str(test_script),
            str(samples_dir),
            "--robot", robot,
            "--ckpt", str(ckpt),
            "--out", str(out_dir),
            "--no-preview",   # pipeline is headless; the interactive MuJoCo
                              # viewer (launch_passive) needs an X display and
                              # would otherwise crash AFTER artifacts are saved.
        ]
        # Per-stage args from config can add e.g. --torso_alpha / --smooth_sigma
        for k, v in ctx.stage_config.args.items():
            flag = f"--{k.replace('_', '-')}"
            if v is True:
                cmd.append(flag)
            elif v is False:
                continue
            else:
                cmd.extend([flag, str(v)])

        # retarget/scripts/test.py expects cwd = retarget_repo (sys.path tricks).
        # Force headless GL for the offscreen robot_sim.mp4 render (EGL, same
        # backend WiLoR's pyrender uses) so it works without an X display.
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        ctx.log.info("running: %s", " ".join(cmd))
        ctx.log.info("cwd: %s", str(retarget_repo))
        ret = subprocess.run(cmd, env=env, cwd=str(retarget_repo))
        if ret.returncode != 0:
            raise RuntimeError(
                f"retarget/scripts/test.py failed (exit {ret.returncode}) "
                f"for robot={robot}")


# Register one Stage class per robot (each with backend = robot name).
for _robot in SUPPORTED_ROBOTS:
    _cls = type(
        f"Retarget{_robot.capitalize()}Stage",
        (_RetargetStage,),
        {"robot": _robot},
    )
    register_stage("retarget", backend=_robot)(_cls)
    globals()[_cls.__name__] = _cls

del _robot, _cls
