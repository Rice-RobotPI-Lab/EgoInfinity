"""Stage implementations.

Import each module here to trigger ``@register_stage`` decorators at
package-import time. Add new stages by importing them below.
"""
from . import extract_frames   # noqa: F401
from . import filter as _filter   # noqa: F401 (`filter` is a builtin; rebind)
from . import phase1   # noqa: F401
from . import post_track   # noqa: F401
from . import refresh   # noqa: F401  (phase1-internal re-run tools as stages)
from . import retarget   # noqa: F401  (stub; maintained separately by retarget team)
