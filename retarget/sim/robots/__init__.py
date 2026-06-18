"""
Robot registry.

ROBOT_CONFIGS  — per-robot runtime configuration (env class, paths, joint configs).
SAMPLE_CONFIGS — per-robot trajectory sampling parameters for training.
ENV_CONFIGS    — per-robot JaxVecEnv configuration (RobotConfig dataclass).

To add a new robot:
  1. Create sim/robots/<name>/env.py          — subclass BaseEnv; define _MJCF_MJX_PATH,
                                                _ARM_JOINTS, _EE_BODY
  2. Create sim/robots/<name>/config.py       — define CONFIG dict and ENV_CONFIG
  3. Create sim/robots/<name>/sample_config.py — define SAMPLE_CONFIG dict
  4. Register all three below with a unique key.
"""

from sim.robots.g1.config        import CONFIG as _G1,        ENV_CONFIG as _G1_ENV
from sim.robots.franka.config    import CONFIG as _FRANKA,    ENV_CONFIG as _FRANKA_ENV
from sim.robots.robonaut2.config import CONFIG as _ROBONAUT2, ENV_CONFIG as _ROBONAUT2_ENV
from sim.robots.xlerobot.config  import CONFIG as _XLEROBOT,  ENV_CONFIG as _XLEROBOT_ENV

from sim.robots.g1.sample_config        import SAMPLE_CONFIG as _G1_SC
from sim.robots.franka.sample_config    import SAMPLE_CONFIG as _FRANKA_SC
from sim.robots.robonaut2.sample_config import SAMPLE_CONFIG as _ROBONAUT2_SC
from sim.robots.xlerobot.sample_config  import SAMPLE_CONFIG as _XLEROBOT_SC

ROBOT_CONFIGS = {
    "g1":        _G1,
    "franka":    _FRANKA,
    "robonaut2": _ROBONAUT2,
    "xlerobot":  _XLEROBOT,
}

SAMPLE_CONFIGS = {
    "g1":        _G1_SC,
    "franka":    _FRANKA_SC,
    "robonaut2": _ROBONAUT2_SC,
    "xlerobot":  _XLEROBOT_SC,
}

ENV_CONFIGS = {
    "g1":        _G1_ENV,
    "franka":    _FRANKA_ENV,
    "robonaut2": _ROBONAUT2_ENV,
    "xlerobot":  _XLEROBOT_ENV,
}
