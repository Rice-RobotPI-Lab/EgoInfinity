# Robot model assets — third-party attribution

The MJCF / URDF / mesh assets under this directory are **derived from
third-party robot descriptions**, adapted (scaled, re-pathed, split into
visualization / MJX variants) for EgoInfinity retargeting. They are NOT
original to this project. Each upstream carries its own license and
attribution requirements that apply independently of the repo's MIT license.

| Dir | Robot | Upstream source | License (file in dir) |
|---|---|---|---|
| `franka_fr3/` | Franka FR3 (bimanual) | [MuJoCo Menagerie — `franka_fr3`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3) | Apache-2.0 + Willow Garage BSD-3 — [`franka_fr3/LICENSE`](franka_fr3/LICENSE) |
| `unitree_g1/` | Unitree G1 | [MuJoCo Menagerie — `unitree_g1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) | BSD-3 (Unitree Robotics) — [`unitree_g1/LICENSE`](unitree_g1/LICENSE) |
| `xlerobot/` | XLeRobot | [ManiSkill — `assets/xlerobot`](https://github.com/haosulab/ManiSkill) (converted from its URDF) | Apache-2.0 — [`xlerobot/LICENSE`](xlerobot/LICENSE) + [`NOTICE`](xlerobot/NOTICE) |
| `robonaut2/` | NASA Robonaut 2 | [NASA JSC `r2_description`](https://gitlab.com/nasa-jsc-robotics/r2_description) | [NOSA v1.3](https://opensource.org/licenses/NASA-1.3) — see [`robonaut2/NOTICE`](robonaut2/NOTICE) |

Franka/G1/XLeRobot reproduce their upstream `LICENSE` verbatim in-dir
(permissive Apache-2.0 / BSD-3). Robonaut 2 is NOSA v1.3: its `NOTICE`
attributes NASA + links the upstream repo and the canonical license text
(matching the long-standing gkjohnson/nasa-urdf-robots convention).

> **Optional:** confirm the Menagerie/ManiSkill pins match the committed
> assets' versions. To avoid NOSA's recipient-agreement / modification-notice
> obligations entirely, robonaut2 can be dropped (the other three robots are
> permissively licensed).

The trained retargeting policies (`retarget/ckpts/*.pt`) are EgoInfinity's own
(see `retarget/README.md`); their distributability is downstream of the robot
models credited above.
