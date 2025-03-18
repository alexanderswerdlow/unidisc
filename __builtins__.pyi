from ipdb import set_trace as st
from decoupled_utils import start_timing as start_timing
from decoupled_utils import end_timing as end_timing
ENABLE_TIMING: bool
ENABLE_TIMING_SYNC: bool
DEVICE_BACKEND_TYPE: str
exists = lambda v: v is not None