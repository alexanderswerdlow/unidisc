"""
A collection of assorted utilities for deep learning with no required dependencies aside from PyTorch, NumPy and the Python stdlib.

TODO: Validate required Python version.
"""
import contextlib
import builtins
import functools
import glob
import hashlib
import io
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from functools import lru_cache, partial, wraps
from importlib import import_module
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
import inspect
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

# TPUs:
# wait_device_ops
# https://github.com/pytorch/xla/blob/2766f9f756e8c8150ea3b7df98762c2f82d66e39/examples/debug/train_resnet_benchmark.py#L18

# From diffusers
import importlib
import importlib.metadata

# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False):
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py#L281
@lru_cache
def is_torch_cuda_available():
    return torch.cuda.is_available()

@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."
    _torch_xla_available, _torch_xla_version = _is_package_available("torch_xla", return_version=True)
    if not _torch_xla_available:
        return False

    import torch_xla

    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True

def get_available_backend():
    if is_torch_cuda_available():
        backend = "cuda"
    elif is_torch_xla_available():
        backend = "xla"
    else:
        backend = "cpu"
    return backend

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    builtins.HAS_XLA_SPAWNED = False

def use_dist():
    return dist.is_available() and dist.is_initialized()

def get_device():
    return torch.device(f"cuda:{get_rank()}")

def get_tpu_devices():
    import torch_xla.core.xla_model as xm
    return xm.get_xla_supported_devices()

def get_num_nodes():
    if is_torch_cuda_available():
        rprint(f"Warning: get_num_nodes() is not supported for CUDA. Returning world size // num devices.")
        return get_world_size() // get_num_devices()
    if is_torch_xla_available():
        from torch_xla._internal import tpu
        return tpu.num_tpu_workers()
    
def get_num_devices():
    """
    Number of physical "devices" on a single node. In CUDA, this is the number of GPUs. In XLA, this is the number of TPUs *chips* so, e.g., a v4 slice will always return 4 (even if part of a larger slice).
    """
    if is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 1

def get_world_size():
    if is_torch_xla_available():
        import torch_xla.runtime as xr
        return xr.world_size()
    elif use_dist():
        return dist.get_world_size()
    elif 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 1

@lru_cache
def get_xla_rank():
    # When using spmd, these return 0 regardless of node [e.g., essentially return the local rank]
    # import torch_xla.core.xla_model as xm
    # return xm.get_ordinal()
    # from accelerate import PartialState
    # return PartialState().local_process_index
    from torch_xla._internal import tpu
    task_id = tpu.task_id() # Num chips
    worker_id = tpu.worker_id() # Num workers [e.g., which node]
    return task_id if task_id is not None else worker_id

def get_rank(check_for_group: bool = False):
    """ Helper function that returns torch.distributed.get_rank() if DDP has been initialized otherwise it returns 0.
    """
    if os.environ.get("FORCE_MAIN_RANK", "0") == "1":
        return 0
    elif is_torch_xla_available():
        if builtins.HAS_XLA_SPAWNED:
            return get_xla_rank()
        else:
            return 0
    elif use_dist():
        return dist.get_rank()
    elif (rank := os.environ.get("RANK", None)) is not None:
        return int(rank) # RANK is set by torch.distributed.launch
    elif (rank := os.environ.get("SLURM_PROCID", None)) is not None:
        return int(rank) # SLURM_PROCID is set by SLURM
    elif check_for_group:
        # if neither pytorch, SLURM  env vars are set
        # check NODE_RANK/GROUP_RANK and LOCAL_RANK env vars
        # assume global_rank is zero if undefined
        node_rank = os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", 0))
        local_rank = os.environ.get("LOCAL_RANK", 0)
        return 0 if (int(node_rank) == 0 and int(local_rank) == 0) else 1
    else:
        return 0
    
def is_main_process():
    return get_rank() == 0

def get_local_rank():
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm
        return xm.get_local_ordinal()
    else:
        return int(os.environ.get("LOCAL_RANK", 0))

def is_local_main_process():
    return get_local_rank() == 0

def get_num_gpus() -> int:
    return get_world_size()

def rank_zero_fn(fn):
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any):
        if is_main_process():
            return fn(*args, **kwargs)
        return None

    return wrapped_fn

def barrier():
    if use_dist() or getattr(builtins, "HAS_XLA_SPAWNED", False):
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        code_context = inspect.getframeinfo(frame).code_context[0].strip()
        debug_log_func(f"Before barrier - {filename}:{lineno} - {code_context}")
        if is_torch_xla_available() and getattr(builtins, "HAS_XLA_SPAWNED", False):
            import torch_xla.core.xla_model as xm
            xm.rendezvous('barrier')
        else:
            torch.distributed.barrier()
        debug_log_func(f"After barrier - {filename}:{lineno} - {code_context}")
        
def get_hostname():
    return __import__('socket').gethostname().removesuffix('.eth')

def get_slurm_job_id():
    if os.environ.get("SLURM_ARRAY_JOB_ID", None) is not None and os.environ.get("SLURM_ARRAY_TASK_ID", None) is not None:
        job_str = f"{os.environ.get('SLURM_ARRAY_JOB_ID')}_{os.environ.get('SLURM_ARRAY_TASK_ID')}"
    elif os.environ.get("SLURM_JOB_ID", None) is not None:
        job_str = os.environ.get("SLURM_JOB_ID")
    else:
        job_str = None
    return job_str

def get_restart_str():
    restart_str = f"r{os.environ.get('SLURM_RESTART_COUNT', '0')}_" if os.environ.get('SLURM_RESTART_COUNT', None) is not None else ""
    if "TORCHELASTIC_RESTART_COUNT" in os.environ and os.environ.get("TORCHELASTIC_RESTART_COUNT", '0') != '0':
        restart_str = f"r{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}_"
    return restart_str

def get_slurm_filename_info():
    job_str = f"{get_slurm_job_id()}__" if get_slurm_job_id() is not None else ""
    restart_str = get_restart_str()
    if "SLURM_NODEID" in os.environ:
        node_str = f"{os.environ.get('SLURM_NODEID', '')}_"
    elif "SLURM_PROCID" in os.environ:
        node_str = f"{os.environ.get('SLURM_PROCID', '')}_"
    else:
        node_str = ""

    return f"{job_str}{restart_str}{node_str}{get_rank()}"

def get_slurm_log_prefix():
    if "SLURM_NODEID" in os.environ:
        slurm_nodestr = f", Node:{os.environ.get('SLURM_NODEID', 'N/A')}"
    elif "SLURM_PROCID" in os.environ:
        slurm_nodestr = f", Node:{os.environ.get('SLURM_PROCID', 'N/A')}"
    else:
        slurm_nodestr = ""

    jobid = get_slurm_job_id()
    jobid_str = f", JID:{jobid}" if jobid is not None else ""
    restart_str = f", {get_restart_str()}" if get_restart_str() != "" else ""
    return f"Rank:{get_rank()}{slurm_nodestr}{jobid_str}{restart_str}"

def slurm_prefix_func():
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    return f"{timestamp} [{get_slurm_log_prefix()}]"

try:
    from unidisc.utils.logging_utils import log_info, log_debug, set_logger, log_memory
    set_logger(__name__)
    info_log_func = log_info
    debug_log_func = log_debug
    prefix_func = None
except Exception as e:
    print(e)
    def simple_print_func(*args, **kwargs):
        print(*args)
    info_log_func = simple_print_func
    debug_log_func = simple_print_func
    prefix_func = slurm_prefix_func

def gprint(*args, main_process_only=False, **kwargs):
    """
    Prints to console + log as INFO on regardless of rank.
    """
    if prefix_func is not None:
        args = (prefix_func(), *args)
    info_log_func(*args, main_process_only=main_process_only, **kwargs)

def dprint(*args, **kwargs):
    """
    Prints to console + log as DEBUG on regardless of rank.
    """
    if prefix_func is not None:
        args = (prefix_func(), *args)
    debug_log_func(*args, **kwargs)

def rprint(*args, **kwargs):
    """
    Prints to console + log as INFO on main process only. All ranks also print to log file [as DEBUG] if called, but this is not required [e.g., no barrier used]
    """
    gprint(*args, main_process_only=True, **kwargs)

def mprint(*args, **kwargs):
    log_memory(*args, **kwargs)
    # dprint(*args, **kwargs)

log_func = rprint

def process_file_prefix():
    datetime_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")[:-5]
    return f"{get_slurm_filename_info()}_{get_hostname()}_{datetime_str}"

def get_info():
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log_func(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_params(model):
    log_func(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_func(f"Unfrozen Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log_func(f"Frozen Parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")


def calculate_storage_size(obj, storage_view_sizes, count_views=False):
    if isinstance(obj, torch.Tensor):
        storage = obj.storage()
        storage_id = id(storage)
        element_size = storage.element_size()
        storage_size = storage.size() * element_size
        view_size = obj.numel() * element_size

        # We count storage size only for the first time we encounter the storage
        if storage_id not in storage_view_sizes:
            storage_view_sizes[storage_id] = storage_size
            print_size = storage_size
        else:
            print_size = 0 if not count_views or not obj._is_view() else view_size

        if count_views or not obj._is_view():
            log_func(f"{'View' if obj._is_view() else 'Storage'} Tensor: " f"shape {obj.size()}, size {print_size / (1024 ** 2):.2f} MB")

        return print_size if count_views or not obj._is_view() else 0  # Count views only if requested
    elif isinstance(obj, dict):
        # Recurse for dictionaries
        return sum(calculate_storage_size(v, storage_view_sizes, count_views) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        # Recurse for lists or tuples
        return sum(calculate_storage_size(item, storage_view_sizes, count_views) for item in obj)
    elif hasattr(obj, "__dataclass_fields__"):
        # Recurse for dataclasses based on their fields
        fields = getattr(obj, "__dataclass_fields__")
        return sum(calculate_storage_size(getattr(obj, f), storage_view_sizes, count_views) for f in fields)
    else:
        # Non-Tensor, non-dict, non-list objects are not measured
        return 0


def calculate_total_size(obj, count_views=False):
    storage_view_sizes = defaultdict(int)
    total_size = calculate_storage_size(obj, storage_view_sizes, count_views)
    total_unique_storage_size = sum(storage_view_sizes.values())
    log_func(f"Total unique storage size: {total_unique_storage_size / (1024 ** 2):.2f} MB")
    if count_views:  # Only add view sizes to total if requested
        total_view_size = total_size - total_unique_storage_size
        log_func(f"Total view size (if counted): {total_view_size / (1024 ** 2):.2f} MB")
    else:
        log_func(f"Total size (without counting views): {total_size / (1024 ** 2):.2f} MB")

    return total_size


def save_tensor_dict(tensor_dict: dict, path):
    output_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, Tensor):
            if v.dtype == torch.float16 or v.dtype == torch.bfloat16:
                output_dict[k] = v.to(dtype=torch.float32).detach().cpu().numpy()
            else:
                output_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            output_dict[f"np_{k}"] = v
        else:
            output_dict[k] = v
    np.savez_compressed(path, **output_dict)


def load_tensor_dict(path: Path, object_keys=[]):
    from jaxtyping import BFloat16 # TODO: Remove dependency
    tensor_dict = {}
    np_dict = np.load(path, allow_pickle=True)
    for k, v in np_dict.items():
        if k in object_keys:
            tensor_dict[k] = v
        elif v.dtype == BFloat16:
            tensor_dict[k] = torch.from_numpy(v.astype(np.float32)).to(dtype=torch.bfloat16)
        elif k.startswith("np_"):
            tensor_dict[k.removeprefix("np_")] = v
        else:
            tensor_dict[k] = torch.from_numpy(v)
    return tensor_dict


def tensor_hash(tensor):
    """Computes a SHA256 hash of a tensor. Useful for debugging to check equality in different places."""
    tensor_bytes = tensor.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def module_hash(module: Optional[dict] = None, state_dict: Optional[dict] = None):
    assert module is not None or state_dict is not None
    state_dict = module.state_dict() if module is not None else state_dict
    sorted_state_dict = {k: state_dict[k] for k in sorted(state_dict)}
    params_cat = torch.cat([v.flatten() for _, v in sorted_state_dict.items()])
    return tensor_hash(params_cat)

def parameter_hash(params: list[torch.Tensor]):
    return tensor_hash(torch.cat([p.cpu().flatten() for p in params]))

def find_diff_params(state_dict_1, state_dict_2):
    diff_keys = set(state_dict_1.keys()) ^ set(state_dict_2.keys())  # Symmetric difference to find keys not in both
    matched_keys = set(state_dict_1.keys()) & set(state_dict_2.keys())  # Intersection to find keys in both

    # Check for differences in matched keys
    for key in matched_keys:
        if not torch.equal(state_dict_1[key], state_dict_2[key]):
            diff_keys.add(key)

    return diff_keys


def init_from_ckpt(module, path, ignore_keys=None, unfrozen_keys=None, strict=False, truncate=None, only_incl=None, verbose=True):
    log_func(f"Loading {module.__class__.__name__} from checkpoint: {path}")
    log_func(f"Strict Load: {strict}, Ignoring: {ignore_keys}, Unfreezing: {unfrozen_keys}, Truncating: {truncate}")

    if ignore_keys is None:
        ignore_keys = ()

    if unfrozen_keys is None:
        unfrozen_keys = ()

    sd = torch.load(path, map_location="cpu")

    if "state_dict" in sd.keys():
        sd = sd["state_dict"]
    elif "weight" in sd.keys():
        sd = sd["weight"]

    num_deleted = defaultdict(int)
    for k in list(sd):
        for ik in ignore_keys:
            if k.startswith(ik):
                num_deleted[ik] += 1
                del sd[k]

    for k, v in num_deleted.items():
        log_func(f"Deleted {v} keys due to ignore_key: {k}")

    if truncate is not None:
        for k in list(sd):
            if k.startswith(truncate):
                sd[k.replace(truncate, "")] = sd[k]
            del sd[k]

    num_ignored = defaultdict(int)
    for n in module.state_dict().keys():
        if n not in sd.keys():
            for ik in ignore_keys:
                if ik in n:
                    num_ignored[ik] += 1
                else:
                    log_func(f"Missing {n}")

    if only_incl is not None:
        for k in list(sd):
            keep = False
            for ik in only_incl:
                if ik in k:
                    keep = True
            if not keep:
                del sd[k]

    for k, v in num_ignored.items():
        log_func(f"Missing {v} keys due to ignore_key: {k}")

    for n in sd.keys():
        if n not in module.state_dict().keys():
            log_func(f"Unexpected {n}")

    checkpoint_keys = set(sd.keys())
    current_keys = set(module.state_dict().keys())

    if verbose:
        log_func(f"Loading: {checkpoint_keys.intersection(current_keys)}")
    else:
        log_func(f"Loading {len(checkpoint_keys.intersection(current_keys))} keys into the model: {str(module.__class__)}")

    module.load_state_dict(sd, strict=strict)

    if len(unfrozen_keys) > 0:
        for n, p in module.named_parameters():
            p.requires_grad_ = False
            for unfrozen_name in unfrozen_keys:
                if unfrozen_name in n:
                    p.requires_grad_ = True
                    log_func(f"Unfreezing: {n}")

    log_func(f"Restored from {path}")


def check_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total_memory = torch.cuda.get_device_properties(int(get_local_rank())).total_memory

    allocated_percent = (allocated / total_memory) * 100
    reserved_percent = (reserved / total_memory) * 100

    log_func(f"Allocated memory: {allocated_percent:.2f}%")
    log_func(f"Reserved memory: {reserved_percent:.2f}%")
    log_func(f'Available devices (CUDA_VISIBLE_DEVICES): {os.environ.get("CUDA_VISIBLE_DEVICES")}')

    assert allocated_percent <= 5
    assert reserved_percent <= 5


def load_checkpoint_from_url(url: str, file_path: Optional[str] = None) -> Path:
    if file_path is None:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if file_path is not None:
            filename = file_path

        file_path = Path.home() / ".cache" / "pretrained_weights" / filename

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_path):
        log_func(f'Downloading: "{url}" to {file_path}\n')
        torch.hub.download_url_to_file(url, file_path, progress=True)

    return file_path


# Copied from torch.profiler.profiler
def tensorboard_trace_handler(dir_name: str, record_memory: bool = False, worker_name: Optional[str] = None, use_gzip: bool = True):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof: torch.profiler.profile) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"

        chrome_trace_path = os.path.join(dir_name, file_name)
        memory_trace_path = os.path.join(dir_name, "memory_timeline.html")
        if is_main_process() and record_memory:
            try:
                log_func(f"Exporting memory timeline: {memory_trace_path}")
                prof.export_memory_timeline(memory_trace_path)
            except Exception as e:
                log_func(f"Failed to export memory timeline: {e}")
        
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)

        log_func(f"Exporting chrome trace to {chrome_trace_path}")
        prof.export_chrome_trace(chrome_trace_path)

    return handler_fn

def get_date_time_str():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")[:-3]

def save_memory_profile(profile_dir):
    import wandb
    rank_postfix = f"_rank_{get_rank()}" if use_dist() else ""
    log_func(f"Saving memory profile to {profile_dir}")
    os.makedirs(profile_dir, exist_ok=True)
    torch.cuda.memory._dump_snapshot(f"{profile_dir}/memory_snapshot{rank_postfix}.pickle")
    os.system(
        f"python -m torch.cuda._memory_viz trace_plot {profile_dir}/memory_snapshot{rank_postfix}.pickle -o {profile_dir}/memory_snapshot{rank_postfix}.html"
    )
    torch.cuda.memory._save_segment_usage(f"{profile_dir}/segment{rank_postfix}.svg")
    torch.cuda.memory._save_memory_usage(f"{profile_dir}/memory{rank_postfix}.svg") 
    torch.cuda.memory._record_memory_history(enabled=None)

    log_func(f"Saved memory snapshot at: {profile_dir}/memory_snapshot{rank_postfix}.pickle")
    log_func(f"Run the following to view the snapshot:\npython -m http.server --directory {profile_dir.resolve()} 6008")

    if is_main_process() and wandb.run is not None:
        wandb.log({'profile': wandb.Html(f"{profile_dir}/memory_snapshot{rank_postfix}.html")})
        wandb.log({'profile': wandb.Html(f"{profile_dir}/memory_timeline{rank_postfix}.html")})

def print_memory(verbose: bool = False, print_output: bool = True):
    max_cur_reserved, max_peak_reserved, max_peak_allocated, max_cur_allocated = -1, -1, -1, -1
    max_cur_reserved_device, max_peak_reserved_device, max_peak_allocated_device, max_cur_allocated_device = -1, -1, -1, -1
    for device in range(torch.cuda.device_count()):
        current_reserved_memory_MB = torch.cuda.memory_reserved(device=torch.device(f'cuda:{device}')) / (1024**2)
        peak_reserved_memory_MB = torch.cuda.max_memory_reserved(device=torch.device(f'cuda:{device}')) / (1024**2)
        peak_allocated_memory_MB = torch.cuda.max_memory_allocated(device=torch.device(f'cuda:{device}')) / (1024**2)
        current_allocated_memory_MB = torch.cuda.memory_allocated(device=torch.device(f'cuda:{device}')) / (1024**2)

        if current_reserved_memory_MB > max_cur_reserved:
            max_cur_reserved = current_reserved_memory_MB
            max_cur_reserved_device = device
        
        if peak_reserved_memory_MB > max_peak_reserved:
            max_peak_reserved = peak_reserved_memory_MB
            max_peak_reserved_device = device

        if peak_allocated_memory_MB > max_peak_allocated:
            max_peak_allocated = peak_allocated_memory_MB
            max_peak_allocated_device = device

        if current_allocated_memory_MB > max_cur_allocated:
            max_cur_allocated = current_allocated_memory_MB
            max_cur_allocated_device = device

    if verbose:
        log_func(torch.cuda.memory_summary(abbreviated=False))

    if print_output:
        log_func(f"GPU Cur Reserved: {max_cur_reserved:.2f}MB on rank {max_cur_reserved_device}, Cur Allocated: {max_cur_allocated:.2f}MB on rank {max_cur_allocated_device}, Peak Reserved: {max_peak_reserved:.2f}MB on rank {max_peak_reserved_device}, Peak Allocated: {max_peak_allocated:.2f}MB on rank {max_peak_allocated_device}")

    return max_cur_reserved

def print_memory_summary():
    val = print_memory(verbose=False, print_output=False)
    log_func(f"GPU Cur Reserved: {val:.2f}MB, {val / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100:.2f}%")

@contextlib.contextmanager
def show_memory_usage(empty_cache: bool = True, verbose: bool = False, print_output=False, show_caller: bool = True):
    synchronize_device()
    if empty_cache: clear_cache()
    
    if show_caller:
        callers = [
            f"{frame.function} (...{frame.filename[-10:]}:{frame.lineno})"
            for frame in inspect.stack()
            if "contextlib" not in frame.filename
        ][1:5]
    
        decorated_func_name = ", ".join([inspect.currentframe().f_back.f_code.co_name, inspect.currentframe().f_back.f_back.f_code.co_name, inspect.currentframe().f_back.f_back.f_back.f_code.co_name])
        caller_info = decorated_func_name + " " + " -> ".join(reversed(callers))
        log_func(f"Before context (called by {caller_info}): {print_memory(verbose, print_output=False)}MB cur reserved")
        caller_str = f", called by {caller_info} "
    else:
        caller_str = ""

    yield

    synchronize_device()
    if empty_cache: clear_cache()
    log_func(f"After context{caller_str}: {print_memory(verbose, print_output=print_output)}MB cur reserved")

@contextlib.contextmanager
def profile_memory(enable: bool = True, empty_cache: bool = True):
    with contextlib.ExitStack() as stack:
        stack.enter_context(show_memory_usage(empty_cache=empty_cache))
        if enable and is_main_process(): torch.cuda.memory._record_memory_history()
        yield
        if enable: save_memory_profile(Path(f"output/profile/{get_date_time_str()}"))

def profile_memory_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with profile_memory():
            return func(*args, **kwargs)
    return wrapper

class Profiler:
    def __init__(self, output_dir, warmup_steps: int = 5, active_steps: int = 3, record_memory: bool = False, record_memory_only: bool = False):
        self.record_memory = record_memory
        self.profile_dir = Path(output_dir) / "profile"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        wait, warmup, active, repeat = 0, warmup_steps, active_steps, 0
        self.total_steps = (wait + warmup + active) * (1 + repeat)
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler_kwargs = dict(record_shapes=True, with_stack=True)
        if record_memory_only:
            pass
        else:
            profiler_kwargs.update(
                dict(
                    with_modules=True,
                    with_flops=True,
                ),
            )

        self.profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=tensorboard_trace_handler(self.profile_dir, record_memory=record_memory),
            profile_memory=record_memory,
            **profiler_kwargs
        )
        self.profiler.start()

    def step(self, global_step: int):
        self.profiler.step()
        return global_step >= (self.total_steps - 1)

    def finish(self):
        self.profiler.stop()
        if use_dist():
            torch.distributed.barrier()
        if is_main_process():
            import wandb
            traces = glob.glob(f"{self.profile_dir}/*.pt.trace.json*")
            for trace in traces:
                log_func(f"Adding {trace}")
                wandb.save(trace, base_path=self.profile_dir, policy="now")                

        if use_dist():
            torch.distributed.barrier()


def get_pdb():
    return import_module("pdb") if (any(["_pdbpp_path_hack" in str(p) for p in sys.path]) or find_spec("ipdb") is None) else import_module("ipdb")

def _breakpoint(rank: Optional[int] = None, traceback: Optional[Any] = None):
    if get_num_gpus() > 1:
        if (is_main_process() if rank is None else get_rank() == rank):
            if is_torch_xla_available():
                from fairseq import pdb
                pdb.set_trace()
            else:
                old_stdin = None
                if isinstance(sys.stdin, io.TextIOWrapper):
                    old_stdin = sys.stdin
                    sys.stdin = open(0)
                try:
                    log_func('Breakpoint triggered. You may need to type "up" to get to the correct frame')
                    log_func(f'Traceback: {traceback}')
                    if traceback is not None:
                        get_pdb().post_mortem(traceback)
                    else:
                        get_pdb().set_trace()
                finally:
                    if old_stdin is not None:
                        sys.stdin.close()
                        sys.stdin = old_stdin
        barrier()
    else:
        if traceback is not None:
            get_pdb().post_mortem(traceback)
        else:
            log_func("Breakpoint triggered. You may need to type \"up\" to get to the correct frame")
            get_pdb().set_trace(sys._getframe(1))

def set_global_exists():
    builtins.exists = lambda v: v is not None

def set_global_breakpoint():
    import ipdb

    builtins.breakpoint = _breakpoint
    builtins.st = ipdb.set_trace  # We import st everywhere
    builtins.ug = lambda: globals().update(locals())

def set_timing_builtins(enable: bool = False, sync: bool = False):
    builtins.start_timing = partial(start_timing, builtin=True, enable=False, sync=False)
    builtins.end_timing = partial(end_timing, builtin=True, enable=False, sync=False)
    builtins.ENABLE_TIMING = enable
    builtins.ENABLE_TIMING_SYNC = sync

def synchronize_device():
    if is_torch_cuda_available():
        torch.cuda.synchronize()
    elif is_torch_xla_available():
        import torch_xla
        torch_xla.sync()
        xm.wait_device_ops()

def clear_cache():
    if is_torch_cuda_available():
        # This caused untold grief. Without calling `_cuda_clearCublasWorkspaces`,
        # some model configurations would eventually cause a CUDA OOM during inference. See:
        # https://github.com/pytorch/pytorch/issues/99835, https://github.com/pytorch/pytorch/issues/105181
        torch._C._cuda_clearCublasWorkspaces()
        torch._dynamo.reset()
        import gc; gc.collect()
        torch.cuda.empty_cache()
    elif is_torch_xla_available():
        rprint("Clearing cache not supported for XLA")

def start_timing(message: str, enable: bool = False, sync: bool = False, builtin: bool = False):
    if (builtin and ENABLE_TIMING) or enable:
        if (builtin and ENABLE_TIMING_SYNC) or sync:
            synchronize_device()
        torch.cuda.nvtx.range_push(f"[SYNC] {message}" if ((builtin and ENABLE_TIMING_SYNC) or sync) else message)

    return time.time()

def end_timing(start_time: Optional[float] = None, enable: bool = False, sync: bool = False, builtin: bool = False):
    if (builtin and ENABLE_TIMING) or enable:
        if (builtin and ENABLE_TIMING_SYNC) or sync:
            synchronize_device()
        torch.cuda.nvtx.range_pop()

    if start_time is not None:
        return time.time() - start_time
    
@contextlib.contextmanager
def breakpoint_on_error():
    set_global_breakpoint()
    try:
        yield
    except Exception as e:
        print("Exception...", e)
        traceback.print_exc()
        breakpoint(traceback=e.__traceback__)
        raise e

@contextlib.contextmanager
def get_time_sync(enable: bool = True):
    if enable and is_main_process():
        synchronize_device()
        start_time = time.time()
    yield
    if enable and is_main_process():
        synchronize_device()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

def write_to_file(path: Path, text: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as file:
            file.write(text + "\n")
    except:
        log_func(f"Could not write to {path}")


def to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  if isinstance(obj, dict):
    return {k : to(v, device) for k, v in obj.items()}
  if isinstance(obj, tuple):
    return tuple(to(v, device) for v in obj)
  if isinstance(obj, list):
    return [to(v, device) for v in obj]
  return obj

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_num_gpus()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(to(pickle.loads(buffer), torch.device('cpu')))

    return data_list

def get_modules(model: torch.nn.Module, cls: Any):
    children = list(model.children())
    if isinstance(model, cls):
        return [model]
    elif len(children) == 0:
        return []
    else:
        return [ci for c in children for ci in get_modules(model=c, cls=cls)]

map_chars = {
    "/" : "__",
    " " : "_",
}

def sanitize_filename(filename: str) -> str:
    return "".join(map_chars.get(c, c) for c in filename if c.isalnum() or map_chars.get(c, c) in (" ", ".", "_", "-", "__"))


def hash_str_as_int(s: str):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8


def torch_to_numpy(arr: Tensor):
    if arr.dtype == torch.bfloat16:
        return arr.float().cpu().detach().numpy()
    else:
        return arr.cpu().detach().numpy()
    
def to_numpy(arr):
    if isinstance(arr, Tensor):
        return torch_to_numpy(arr)
    else:
        return arr

class try_except:
    def __init__(self, raise_error: bool = False, write_error_to_file: bool = False, clear_cuda_cache: bool = False):
        self.raise_error = raise_error
        self.write_error_to_file = write_error_to_file
        self.clear_cuda_cache = clear_cuda_cache

    def __call__(self, f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return inner

    def __enter__(self):
        if self.clear_cuda_cache:
            clear_cache()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            try:
                log_func(f"Exception caught: {exc_type}")
                log_func(traceback.format_exc())
                if self.write_error_to_file:
                    timestamp = int(time.time_ns())
                    with open(f"exception_{timestamp}_{process_file_prefix()}.out", "w") as file:
                        file.write(traceback.format_exc())
            except Exception as e:
                print(f"Error writing to file: {e}")

            if self.clear_cuda_cache:
                clear_cache()

            if self.raise_error:
                raise exc_value
        return True  # Suppress the exception if raise_error is False

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device=device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError("Invalid type for move_to")



import types
def run_with_named_function(name, func, *args, **kwargs):
    """
    Runs the given function inside a dynamically created function with the specified name.
    This causes the specified name to appear in the stack trace.

    E.g., x = run_with_named_function(f"iter_{i}", self._ddpm_update, x, t, dt, x0=x0, x0_unmask=x0_unmask, **kwargs)
    
    Parameters:
    - name (str): The desired name to appear in the stack trace.
    - func (callable): The function to execute.
    - *args: Positional arguments to pass to func.
    - **kwargs: Keyword arguments to pass to func.
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    code = wrapper.__code__
    new_code = types.CodeType(
        code.co_argcount,
        code.co_posonlyargcount if hasattr(code, "co_posonlyargcount") else 0,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_filename,
        name,  # Set the function name in the code object
        code.co_firstlineno,
        code.co_lnotab,
        code.co_freevars,
        code.co_cellvars
    )
    new_func = types.FunctionType(
        new_code,
        wrapper.__globals__,
        name,
        wrapper.__defaults__,
        wrapper.__closure__,
    )
    return new_func(*args, **kwargs)