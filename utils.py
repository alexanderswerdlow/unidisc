"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import fsspec
import torch
from timm.scheduler import CosineLRScheduler
import omegaconf
import rich
import rich.syntax
import rich.tree

from decoupled_utils import rank_zero_fn, rprint
from decoupled_utils import (get_hostname, get_num_devices, get_tpu_devices, gprint,
                             is_torch_cuda_available, is_torch_xla_available, rprint)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        gprint(f"{name} has nans: {tensor}")


class CosineDecayWarmupLRScheduler(CosineLRScheduler, torch.optim.lr_scheduler._LRScheduler):
    """Wrap timm.scheduler.CosineLRScheduler
    Enables calling scheduler.step() without passing in epoch.
    Supports resuming as well.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on
        # whether we're using the scheduler every epoch or every
        # step.
        # Otherwise, lightning will always call step (i.e.,
        # meant for each epoch), and if we set scheduler
        # interval to "step", then the learning rate update will
        # be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


class Sampler:
    def __init__(self, shape):
        self.shape = shape

    def _sampling_noise(self):
        pass

    def _hard_sample(self, logits):
        pass

    def _soft_sample(self, logits):
        return 0

    def sample(self, logits):
        noise = self._sampling_noise()
        noise = noise[: logits.shape[0], :]
        logits = logits + noise.to(dtype=logits.dtype, device=logits.device)
        hard_sample = self._hard_sample(logits)
        soft_sample = self._soft_sample(logits)
        return soft_sample + (hard_sample - soft_sample).detach()


class TopKSampler(Sampler):
    def __init__(self, k, shape, gamma_tau=1.0):
        super().__init__(shape)
        self.k = k
        self.gamma_tau = gamma_tau
        self.num_betas = 10
        self.sampler = torch.distributions.gamma.Gamma(1 / k * torch.ones(self.num_betas, *self.shape), 1.0)

    def _sampling_noise(self):
        noise = self.sampler.sample()
        beta = self.k / torch.arange(1, self.num_betas + 1, 1, dtype=torch.float32)
        beta = beta[:, None, None]
        assert beta.ndim == noise.ndim
        s = noise / beta
        s = torch.sum(s, axis=0)
        s = s - math.log(10.0)
        s = self.gamma_tau * (s / self.k)
        return s

    def _hard_sample(self, logits):
        assert logits.ndim == 2
        thresholds, _ = torch.sort(logits, dim=-1)
        thresholds = thresholds[:, -self.k][:, None]
        return (logits >= thresholds).type(logits.dtype)

    def _soft_sample(self, logits):
        soft_top_k = logits - torch.mean(logits, dim=-1, keepdim=True)
        return soft_top_k / torch.norm(soft_top_k, dim=-1, keepdim=True)


class DeterministicTopK(TopKSampler):
    def __init__(self, k):
        super().__init__(k, shape=(1, 1))

    def _sampling_noise(self):
        return 0

    def discreize(self, x):
        hard_sample = self._hard_sample(x)
        soft_sample = self._soft_sample(x)
        return soft_sample + (hard_sample - soft_sample).detach()


class GumbelSampler(Sampler):

    def __init__(self, shape, temperature=1.0):
        super().__init__(shape)
        self.temperature = temperature

    def _sampling_noise(self):
        return -(1e-10 - (torch.rand(*self.shape) + 1e-10).log()).log()

    def _hard_sample(self, logits):
        assert logits.ndim == 2
        indices = torch.argmax(logits, dim=-1)
        zeros = logits * 0
        ones = torch.ones_like(logits[:, :, :1])
        return torch.scatter(zeros, -1, indices[:, :, None], ones)

    def _soft_sample(self, logits):
        return torch.nn.functional.softmax(logits / self.temperature, dim=-1)


class BinarySampler(GumbelSampler):

    def sample(self, probs):
        # TODO(subhamsahoo): use the temperature parameter.
        pos_noise = self._sampling_noise().to(dtype=probs.dtype, device=probs.device)
        neg_noise = self._sampling_noise().to(dtype=probs.dtype, device=probs.device)
        del_noise_exp = (neg_noise - pos_noise).exp()
        hard_sample = (probs * (1 + del_noise_exp) > 1).to(probs.dtype)
        soft_sample = probs / (probs + (1 - probs) * del_noise_exp)
        return soft_sample + (hard_sample - soft_sample).detach()


class GaussianSampler:
    def __init__(self):
        self.softplus = torch.nn.Softplus()

    def sample(self, x):
        assert x.ndim == 2
        n = x.shape[-1] // 2
        mu = x[:, :n]
        sigma = self.softplus(x[:, n:]).sqrt()
        return mu + sigma * torch.randn_like(mu)


def is_global_rank_zero():
    """Helper function to determine if the current process is global_rank 0 (the main process)"""
    # Try to get the pytorch RANK env var
    # RANK is set by torch.distributed.launch
    rank = os.environ.get("RANK", None)
    if rank is not None:
        return rank == 0

    # Try to get the SLURM global rank env var
    # SLURM_PROCID is set by SLURM
    slurm_rank = os.environ.get("SLURM_PROCID", None)
    if slurm_rank is not None:
        return slurm_rank == 0

    # Try to get the MPI global rank env var
    mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
    if mpi_rank is not None:
        return mpi_rank == 0

    # if neither pytorch, SLURM nor MPI env vars are set
    # check NODE_RANK/GROUP_RANK and LOCAL_RANK env vars
    # assume global_rank is zero if undefined
    node_rank = os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", 0))
    local_rank = os.environ.get("LOCAL_RANK", 0)
    return node_rank == 0 and local_rank == 0


def get_rank():
    """Helper function that returns torch.distributed.get_rank() if DDP has been initialized otherwise it returns 0."""

    if is_global_rank_zero():
        return 0
    else:
        return torch.distributed.get_rank()


def set_numa_affinity(gpu_index, verbose=False):
    import pynvml as nvml

    nvml.nvmlInit()
    """This util will assign to the current process the cpu cores set that resides on the same NUMA
    node as the GPU. Typically if you have 8 GPUs, then the first 4 are on the first NUMA node and
    the remaining 4 are on the second.

    `gpu_index` is typically the same as `LOCAL_RANK` in the distributed training, but beware that
    `CUDA_VISIBLE_DEVICES` could impact that. e.g. `CUDA_VISIBLE_DEVICES=0,7` won't do the right
    thing - then you will probably want to remap the ids with something like:

    ```
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        gpu_index = ids[gpu_index] # remap
    ```

    """

    num_elements = math.ceil(os.cpu_count() / 64)
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    affinity_string = ""
    for j in nvml.nvmlDeviceGetCpuAffinity(handle, num_elements):
        # assume nvml returns list of 64 bit ints
        affinity_string = f"{j:064b}{affinity_string}"
    affinity_list = [int(x) for x in affinity_string]
    affinity_list.reverse()  # so core 0 is the 0th element
    affinity_to_set = [i for i, e in enumerate(affinity_list) if e != 0]

    if verbose:
        cores = os.sched_getaffinity(0)
        gprint(f"before: {len(cores)} visible cpu cores: {cores}")

    try:
        os.sched_setaffinity(0, affinity_to_set)
    except Exception as e:
        gprint(f"Failed to set affinity: {e}")

    if verbose:
        cores = os.sched_getaffinity(0)
        gprint(f"after: {len(cores)} visible cpu cores: {cores}")


from typing import Dict, Union

import torch
from torch.nn import Module


def grad_norm(module: Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.

    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {f"{group_separator}{name}": p.grad.data.norm(norm_type) for name, p in module.named_parameters() if p.grad is not None}
    total_norm = torch.tensor(list(norms.values())).norm(norm_type)
    return norms, total_norm

has_set_omega_conf_resolvers = False

def set_omega_conf_resolvers():
    global has_set_omega_conf_resolvers
    if has_set_omega_conf_resolvers:
        return
    has_set_omega_conf_resolvers = True
    import omegaconf
    from omegaconf import OmegaConf

    def get_dir_name(_root_):
        if str(_root_.mode) == "eval":
            return "eval"
        elif _root_.debug:
            return "debug"
        else:
            return _root_.data.train

    def getpythoncmd(_root_):
        return _root_.python_orig + "--multi_gpu \\\n" if (_root_.trainer.devices * _root_.trainer.num_nodes > 1) else _root_.python_orig

    def custom_batch_size():
        if is_torch_cuda_available() and torch.cuda.get_device_properties(0).total_memory >= 23 * 1024 * 1024 * 1024:
            return 64
        elif is_torch_cuda_available() and torch.cuda.get_device_properties(0).total_memory >= 10 * 1024 * 1024 * 1024:
            return 32
        else:
            return 28

    def get_slurm_name(_root_):
        return _root_.slurm_name if hasattr(_root_, "slurm_name") and _root_.slurm_name is not None else _root_.wandb.project
    
    partition_time_limit_min = {
        "partition_name": 60 * 6,
    }
    
    gpu_constraints = {
        "cluster_name": "gpu_constraints", # e.g. "A5000|A6000"
    }

    partitions = {
        "cluster_name": "partition_name",
    }


    babel_exclude_nodes = set()
    if os.environ.get("BAD_NODES", None) is not None:
        babel_exclude_nodes.update(os.environ.get("BAD_NODES").split(","))

    exclude_nodes = {
        "cluster_name": "nodes_to_exclude",
    }

    def get_hostname_split():
        return get_hostname().split("-")[0].split(".")[0]

    omegaconf.OmegaConf.register_new_resolver("getpythoncmd", getpythoncmd)
    omegaconf.OmegaConf.register_new_resolver("get_dir_name", get_dir_name)
    omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
    omegaconf.OmegaConf.register_new_resolver("device_count", get_num_devices)
    omegaconf.OmegaConf.register_new_resolver("eval", eval)
    omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
    omegaconf.OmegaConf.register_new_resolver("find_grad_accum", lambda x, y: round(x / y))
    omegaconf.OmegaConf.register_new_resolver("find_partition", lambda: partitions[get_hostname_split()] if get_hostname_split() in partitions else "all")
    omegaconf.OmegaConf.register_new_resolver("find_constraint", lambda: gpu_constraints[get_hostname_split()] if get_hostname_split() in gpu_constraints else "")
    omegaconf.OmegaConf.register_new_resolver("is_ar", lambda parameterization: parameterization == "ar")
    omegaconf.OmegaConf.register_new_resolver("kv_cache_batch_size", lambda eval_batch_size, cfg: eval_batch_size * 2 if cfg is not None else eval_batch_size)
    omegaconf.OmegaConf.register_new_resolver("exclude_nodes", lambda: exclude_nodes[get_hostname_split()] if get_hostname_split() in exclude_nodes else "")
    omegaconf.OmegaConf.register_new_resolver("get_slurm_name", get_slurm_name)
    

    def adjust_n_blocks(_root_):
        return (
            (_root_.model.base_n_blocks - 1 if _root_.model.base_n_blocks < 24 else _root_.model.base_n_blocks - 2)
            if str(_root_.backbone) == "maskdit"
            else _root_.model.base_n_blocks
        )

    omegaconf.OmegaConf.register_new_resolver("adjust_n_blocks", adjust_n_blocks)
    omegaconf.OmegaConf.register_new_resolver("partition_limit", lambda x: partition_time_limit_min[x] if x in partition_time_limit_min else 60 * 6)
    omegaconf.OmegaConf.register_new_resolver("custom_batch_size", custom_batch_size)
    omegaconf.OmegaConf.register_new_resolver("get_repo_dir", lambda: os.getenv("UNIDISC_DIR", str(Path(__file__).parent)))


@rank_zero_fn
def _print_config(config, resolve: bool = True, save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if save_cfg:
        with fsspec.open("config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)

def set_torch_defaults(benchmark=True):
    torch.set_float32_matmul_precision("medium")
    if is_torch_cuda_available():
        rprint(f"Setting torch defaults")
        exec("import torch.backends.cuda as cuda")
        exec("import torch.backends.cudnn as cudnn")
        exec("cudnn.enabled = True")
        if benchmark:
            exec("cudnn.benchmark = True")
        else:
            rprint(f"Warning: Not benchmarking")
        exec("cudnn.allow_tf32 = True")
        exec("cuda.matmul.allow_tf32 = True")
        exec("cudnn.deterministic = False")
    else:
        rprint(f"Warning: CUDA not available. Not setting defaults.")

from torch.distributed.elastic.multiprocessing.errors import (ChildFailedError,
                                                              record)
from torch.distributed.elastic.multiprocessing.errors.handlers import \
    get_error_handler

_NOT_AVAILABLE = "<N/A>"
class ErrorHandler:
    def __init__(self, error_handler=None):
        self.error_handler = error_handler or get_error_handler()

    def __enter__(self):
        assert self.error_handler is not None
        self.error_handler.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if issubclass(exc_type, SystemExit) and exc_value.code == 0:
                return True  # Prevents SystemExit with code 0 from stopping the program
            elif issubclass(exc_type, ChildFailedError):
                rank, failure = exc_value.get_first_failure()
                if failure.error_file != _NOT_AVAILABLE:
                    self.error_handler.dump_error_file(failure.error_file, failure.exitcode)
                else:
                    rprint(
                        "local_rank %s FAILED with no error file. "
                        "Decorate your entrypoint fn with @record for traceback info. "
                        "See: https://pytorch.org/docs/stable/elastic/errors.html",
                        rank
                    )
                return False  # Re-raises the exception
            self.error_handler.record_exception(exc_value)
        return False  # Any other exceptions will be re-raised


def convert_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "attn_out" in k:
            new_key = k.replace("attn_out", "attention.attn_out")
        elif "attn_qkv" in k:
            new_key = k.replace("attn_qkv", "attention.attn_qkv")
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

from accelerate.utils import extract_model_from_parallel
def apply_compile(model, **compile_kwargs):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in extract_model_from_parallel(model).blocks.named_children():
        transformer_block = torch.compile(transformer_block, **compile_kwargs)
        extract_model_from_parallel(model).blocks.register_module(layer_id, transformer_block)

    output_layer = torch.compile(extract_model_from_parallel(model).output_layer, **compile_kwargs)
    extract_model_from_parallel(model).register_module("output_layer", output_layer)

def compile_model(config, model):
    compile_kwargs = dict()

    if config.backbone == "maskdit":
        compile_kwargs["dynamic"] = True

    compile_kwargs["mode"] = config.trainer.compile_mode
    rprint(f"Using compile mode: {config.trainer.compile_mode}")

    if getattr(config.trainer, "sd3_compile_config", True):
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        rprint(f"Using SD3 compile config")

    if config.trainer.compile_fullgraph:
        compile_kwargs["fullgraph"] = True
        rprint(f"Using fullgraph compile")

    if getattr(config.trainer, "compile_per_layer", False):
        apply_compile(model, **compile_kwargs)
    else:
        model = torch.compile(model, **compile_kwargs)

    return model
