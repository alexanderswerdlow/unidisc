import os
import sys
from contextlib import ExitStack
from pathlib import Path

from constants import CONFIG_PATH, LIB_DIR
sys.path.append(str(LIB_DIR / "hydra_submitit_launcher"))

import builtins
import random
import re
import signal
import traceback
from copy import deepcopy
from datetime import datetime

import hydra
import numpy as np
import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from safetensors.torch import load_file, save_file

import dataloader
from model import Diffusion
import utils
import wandb
from decoupled_utils import (check_gpu_memory_usage, get_hostname,
                             get_local_rank, get_rank, get_slurm_filename_info,
                             get_slurm_log_prefix, get_tpu_devices,
                             get_world_size, gprint, is_local_main_process,
                             is_main_process, is_torch_cuda_available,
                             is_torch_xla_available, print_params,
                             process_file_prefix, profile_memory, rank_zero_fn,
                             rprint, set_global_breakpoint, set_global_exists,
                             set_timing_builtins, try_except)
from utils import (ErrorHandler, _print_config, convert_state_dict_keys, set_omega_conf_resolvers, set_torch_defaults)

# Only needed when debugging hydra
# os.environ["HYDRA_FULL_ERROR"] = "1"

set_global_breakpoint()  # Overrides breakpoint() to use ipdb.set_trace() instead and handle distributed training
set_global_exists()
set_omega_conf_resolvers()

if is_torch_xla_available():
    from jax_smi import initialise_tracking

def _load_from_checkpoint(config, tokenizer):
    OmegaConf.resolve(config)
    if "hf" in config.backbone:
        return Diffusion(config=config, tokenizer=tokenizer).to("cuda")

    return Diffusion.load_from_checkpoint(config.eval.checkpoint_path, tokenizer=tokenizer, config=config)

@rank_zero_fn
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        rprint(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        rprint("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        rprint(f"First {k} tokens:", tokenizer.decode(first))
        rprint("ids:", first)
        rprint(f"Last {k} tokens:", tokenizer.decode(last))
        rprint("ids:", last)


def generate_samples(config, tokenizer):
    rprint("Generating samples.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        rprint("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length, num_strides=num_strides, dt=1 / config.sampling.steps
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
    
    rprint("Text samples:", text_samples)
    if not config.sampling.semi_ar:
        rprint("Generative perplexity:", model.gen_ppl_metric.compute())
    return text_samples


def instantiate_wandb(config, accelerator):
    if is_torch_xla_available():
        gprint("Initializing wandb for XLA")
    if config.mode == 'eval':
        config.wandb.project = f"{config.wandb.project}-eval"
    elif config.mode == 'zero-shot-eval':
        config.wandb.project = f"{config.wandb.project}-zero-shot-eval"

    if config.wandb.group is not None:
        config.wandb.group = str(config.wandb.group)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    wandb_kwargs = dict(config.wandb)

    if getattr(config, "sweep_id", None) is not None:
        rprint(f"Setting Wandb group to {config.sweep_id}")
        wandb_kwargs["group"] = config.sweep_id
    del wandb_kwargs["project"]
    accelerator.init_trackers(
        config.wandb.project, config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True), init_kwargs=dict(wandb=wandb_kwargs)
    )

    if getattr(config.trainer, "log_code", True) and is_main_process():
        if "matrix" in get_hostname():
            rprint(f"Not logging code to wandb on {get_hostname()}")
        else:
            rprint(f"Logging code to wandb from {Path(__file__).parent}")
            try:
                wandb.run.log_code(
                    root=str(Path(__file__).parent),
                    include_fn=lambda path: any(path.endswith(f) for f in (".py", ".yaml", ".yml", ".txt", ".md")),
                    exclude_fn=lambda path, root: any(x in os.path.relpath(path, root) for x in ("output", "multirun", "logs", "wandb")),
                )
            except Exception as e:
                rprint(f"Failed to log code to wandb: {e}")

    with open_dict(config):
        try:
            config.wandb_url = wandb.run.get_url()
            wandb.define_metric("global_samples")
            wandb.define_metric("effective_global_tokens")
            wandb.define_metric("effective_global_step")
            wandb.define_metric("train_metrics/samples")
            wandb.define_metric("trainer/loss", step_metric="global_samples")
        except Exception as e:
            rprint(f"Failed to get wandb url: {e}")

def instantiate_model(config, tokenizer):
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        rprint("Disabling EMA.")
        model.ema = None

    return model

def gconf(config, attr):
    return getattr(config, attr, None)


def has_ckpt(config, attr):
    return gconf(config, attr) is not None and utils.fsspec_exists(gconf(config, attr))


def set_env_vars(config):
    import torch
    hostname = __import__("socket").gethostname()
    rprint(f"Starting Training on {hostname}")
    import torch
    # os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((Path.home() / ".cache" / "torchinductor").resolve())

    if not is_torch_xla_available():
        try:
            # Applies the equivalent of ulimit -l unlimited to this process [and children]
            # This caused a significant amount of pain to figure out
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
            if is_local_main_process():
                gprint(f"Successfully set RLIMIT_MEMLOCK to {hard}")
        except ValueError as e:
            rprint(f"Failed to set RLIMIT_MEMLOCK: {e}")
        except resource.error as e:
            rprint(f"Error setting RLIMIT_MEMLOCK: {e}")
    else:
        rprint(f"Not setting RLIMIT_MEMLOCK on XLA")

    if "matrix-3-28" in hostname or "matrix-3-26" in hostname:
        rprint(f"Disabling NCCL P2P")
        os.environ["NCCL_P2P_DISABLE"] = "1"

    if os.environ.get("TORCH_DISTRIBUTED_DEBUG", "") != "":
        assert False, f"TORCH_DISTRIBUTED_DEBUG is set to: {os.environ.get('TORCH_DISTRIBUTED_DEBUG')}. Please unset it as it starts a gloo backend."

    if config.model.use_spda_attn:
        os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
        os.environ["TORCH_CUDNN_MHA_ENABLED"] = "1"
        rprint("Setting SPDA Flags")

    if config.trainer.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

def update_config_before_resolution(config):
    import torch
    if hasattr(config, "training"):
        rprint(f"'training' has been refactored to 'trainer'. Please update the config.")
        
    with open_dict(config):
        config.output_dir = os.getcwd()
        config.logging_dir = os.getcwd()
        if config.model.use_kv_cache is False and config.mode == "eval" and config.loader.eval_batch_size > 1:
            config.loader.eval_batch_size = max(config.loader.eval_batch_size, 16)
        
        # todo revert?
        if getattr(config.eval, 'txt_img_ratio', None) is not None:
            # 2,1,0.5,0.25
            tot = config.model.length
            # if its 2:1, then distribute the tokens as 2/3, 1/3
            # if its 1:1, then distribute the tokens as 1/2, 1/2
            # if its 0.5:1, then distribute the tokens as 2/3, 1/3
            # if its 0.25:1, then distribute the tokens as 1/4, 3/4
            if config.eval.txt_img_ratio == 2:
                # do first 2/3 tokens as text, last 1/3 as image
                config.model.txt_length = int(tot * 2/3)
            elif config.eval.txt_img_ratio == 1:
                config.model.txt_length = int(tot / 2)
            elif config.eval.txt_img_ratio == 0.5:
                config.model.txt_length = int(tot * 2/3)
            elif config.eval.txt_img_ratio == 0.25:
                config.model.txt_length = int(tot / 4)
            config.model.img_length = tot - config.model.txt_length
            config.model.length = config.model.txt_length + config.model.img_length
            # config.eval.attention_caching_txt_to_img_ratio = config.model.txt_length // 20
            
        if getattr(config.eval, "varying_seq_len_ratio", False):
            assert getattr(config.eval, "sampling_step_ratio", None) is not None, "Must set both varying_seq_len_ratio and sampling_step_ratio"
            config.sampling.steps = int(config.model.length * config.eval.sampling_step_ratio)

        if getattr(config.eval, "ablation_config", False):
            if config.parameterization == "ar":
                rprint(f"WARNING!!!!! FORCING AR PARAMS")
                config.trainer.ar_shift = True
                config.model.full_attention = False

            config.data.keep_tensordict_on_disk = True
            if is_torch_cuda_available():
                if any(x.lower() in torch.cuda.get_device_name().lower() for x in ["v100", "1080", "2080", "quadro", "titan"]) or torch.cuda.get_device_capability()[0] <= 7:
                    rprint(f"Using 2080Ti/V100, setting precision to fp32")
                    config.trainer.precision = "no"
                    config.model.force_optimized_native_attn = False
                    config.trainer.compile = False
                    if any(x.lower() in torch.cuda.get_device_name().lower() for x in ["2080", "quadro"]):
                        config.loader.eval_batch_size = config.loader.eval_batch_size // 7
                        config.loader.batch_size = config.loader.batch_size // 7
                    elif any(x.lower() in torch.cuda.get_device_name().lower() for x in ["1080", "titan"]):
                        config.loader.eval_batch_size = config.loader.eval_batch_size // 6
                        config.loader.batch_size = config.loader.batch_size // 6
                    else:
                        config.loader.eval_batch_size = config.loader.eval_batch_size // 2
                        config.loader.batch_size = config.loader.batch_size // 2
                elif "a5000" in torch.cuda.get_device_name().lower() or "a4500" in torch.cuda.get_device_name().lower():
                    config.loader.eval_batch_size = config.loader.eval_batch_size // 2
                    config.loader.batch_size = config.loader.batch_size // 2
                else:
                    rprint(f"Found {torch.cuda.get_device_name()}")
                    config.loader.eval_batch_size = config.loader.eval_batch_size // 2
                    config.loader.batch_size = config.loader.batch_size // 2
            
            if getattr(config, "parametierzation", None) == "ar" and config.eval.cfg is not None:
                config.loader.eval_batch_size = config.loader.eval_batch_size // 2
                config.loader.batch_size = config.loader.batch_size // 2

            config.loader.eval_batch_size = max(config.loader.eval_batch_size, 1)
            config.loader.batch_size = max(config.loader.batch_size, 1)

            if getattr(config, "parametierzation", None) == "ar":
                config.trainer.compile = False
            
        if getattr(config.sampling, "sampling_step_frac", None) is not None:
            config.sampling.steps = int(config.model.length * config.sampling.sampling_step_frac)
            rprint(f"Setting sampling steps to {config.sampling.steps}")
        
        if os.environ.get("SUBMITIT_FOLDER") is not None or os.environ.get("CUSTOM_SBATCH_LAUNCHER", "0") == "1":
            rprint(f'Using submitit folder: {os.environ.get("SUBMITIT_FOLDER", "")}, setting slurm=True')
            config.slurm = True

        if (config.debug is False or os.environ.get("HYDRA_RUN_DIR_NAME", None) is not None) and torch.distributed.is_torchelastic_launched():
            config.trainer.restart_on_failure = True
            rprint(f"Setting restart_on_failure to True")

        if config.trainer.restart_on_failure and config.mode == 'train':
            if os.environ.get("HYDRA_RUN_DIR", None) is None and os.environ.get("HYDRA_RUN_DIR_NAME", None) is None:
                os.environ["HYDRA_RUN_DIR"] = config.output_dir
                rprint(f"Setting HYDRA_RUN_DIR to {os.environ['HYDRA_RUN_DIR']}")
            else:
                rprint(f"Not setting HYDRA_RUN_DIR, already set to {os.environ.get('HYDRA_RUN_DIR', 'N/A')}, and HYDRA_RUN_DIR_NAME is set to {os.environ.get('HYDRA_RUN_DIR_NAME', 'N/A')}")

            os.environ["RESTART_FAULT_TOLERANT"] = "1"
            rprint(f"Setting RESTART_FAULT_TOLERANT to 1")
        elif config.trainer.restart_on_failure:
            rprint(f"Restart_on_failure is True, but mode is not 'train', so not setting restart fault tolerant")

        relevant_vars = {}
        for key, value in os.environ.items():
            if "SLURM" in key or "NCCL" in key or "TORCH" in key:
                relevant_vars[key] = value

        config.env_vars = relevant_vars

        if config.trainer.profile_memory:
            config.trainer.max_steps = 2

        if config.debug and config.trainer.force_enable_checkpointing is False and (config.trainer.ckpt_steps is None or config.trainer.ckpt_steps > 0):
            config.trainer.ckpt_steps = 10000
            rprint(f"Only checkpointing every {config.trainer.ckpt_steps} steps in debug mode")

        if config.loader.global_batch_size is None:
            config.loader.global_batch_size = config.loader.batch_size * config.trainer.accumulate_grad_batches * (1 if is_torch_xla_available() else get_world_size())
            config.loader.eval_global_batch_size = config.loader.global_batch_size
            if config.trainer.scale_lr_by_batch_size:
                config.optim.lr = config.optim.lr * (config.loader.global_batch_size / 512)
            rprint(f"Setting global batch size to {config.loader.global_batch_size}, lr to {config.optim.lr}")

        if config.mode != 'train':
            config.checkpointing.resume_wandb = False
            config.wandb.resume = None

        if config.trainer.use_spmd_distributed_checkpointing is None:
            config.trainer.use_spmd_distributed_checkpointing = is_torch_xla_available() and config.trainer.xla_spmd

        if config.trainer.disable_all_eval_generation:
            config.eval.num_masking_viz_batches=0
            config.eval.num_uncond_sample_batches=0
            config.eval.num_sample_batches=0
            config.eval.num_random_masking=0
            config.eval.generate_samples=False
            config.trainer.log_flops=False
            config.eval.log_every_n_evals=-1
            config.eval.log_every_n_fid = -1
            config.model.image_model_fid_eval = False
            rprint("Disabling all eval generation!!!")

        if os.environ.get("XLA_IR_DEBUG", "0") == "1":
            config.trainer.tpu_profile = True

        if config.checkpointing_root_dir is not None:
            assert "checkpoints" in config.checkpointing.save_dir
            relative_path = Path(*Path(config.checkpointing.save_dir).relative_to(config.root_output_dir).parts[1:])
            full_checkpointing_dir = Path(config.checkpointing_root_dir) / relative_path
            if config.checkpointing_root_dir is not None:
                old_save_dir = Path(config.output_dir) / "checkpoints"
                full_checkpointing_dir.mkdir(parents=True, exist_ok=True)
                try:
                    if old_save_dir.exists():
                        rprint(f"WARNING: Cannot create symlink from {old_save_dir} to {full_checkpointing_dir} because {old_save_dir} exists.")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        old_save_dir = Path(*old_save_dir.parts[:-1]) / f"checkpoints_{timestamp}"
                    
                    old_save_dir.symlink_to(full_checkpointing_dir, target_is_directory=True)
                    rprint(f"Created softlink from {old_save_dir} to {full_checkpointing_dir}")

                    # Create a symlink from the parent of full_checkpointing_dir named "original" back to config.output_dir
                    original_link = full_checkpointing_dir.parent / "original_output_dir"
                    if not original_link.exists():
                        original_link.symlink_to(Path(config.output_dir).resolve(), target_is_directory=True)
                        rprint(f"Created softlink from {original_link} to {config.output_dir}")
                    else:
                        rprint(f"WARNING: Symlink {original_link} already exists. Skipping creation.")

                except OSError as e:
                    rprint(f"Error creating softlinks: {e}")

        assert getattr(config.data, "allow_label", False) == getattr(config.trainer, "add_label", False) == (getattr(config.model, "add_labels", None) is not None) == getattr(config.eval, "class_conditional_fid", False), f"Mismatching values: data.allow_label={config.data.allow_label}, trainer.add_label={config.trainer.add_label}, model.add_labels={config.model.add_labels}, eval.class_conditional_fid={config.eval.class_conditional_fid}"

        if getattr(config.loader, "num_eval_workers", None) is not None and config.loader.num_workers == 0:
            rprint(f"Setting num_eval_workers to 0 because num_workers is 0")
            config.loader.num_eval_workers = 0

    if config.trainer.disable_all_checkpointing:
        gprint("-"*50)
        gprint(f"WARNING: DISABLING ALL CHECKPOINTING!!!!")
        gprint("-"*50)
        gprint(f"WARNING: DISABLING ALL CHECKPOINTING!!!!")
        gprint("-"*50)
        config.trainer.ckpt_steps = 100000000

    if config.sampling.steps != config.sampling.max_sampling_steps:
        rprint(f"WARNING!!!! steps {config.sampling.steps} != max_sampling_steps {config.sampling.max_sampling_steps}")
        config.sampling.max_sampling_steps = config.sampling.steps

def get_latest_ckpt(config, input_dir):
    if input_dir is None or not Path(input_dir).exists():
        rprint(f"Project dir {input_dir} does not exist")
        return None
    
    if config.trainer.xla_spmd and is_torch_xla_available():
        rprint(f"XLA SPMD detected, using XLA checkpointing")
        if any(Path(input_dir).iterdir()):
            rprint(f"Found existing files/folders in {input_dir}")
            return input_dir
        else:
            rprint(f"No folders found in {input_dir}")
            return None

    folders = [str(folder) for folder in Path(input_dir).iterdir() if folder.is_dir() and ((folder / "model.safetensors").exists() or (folder / "config.yaml").exists())]

    if len(folders) == 0:
        rprint(f"No folders found in {input_dir}")
        return None

    def _inner(folder):
        return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

    folders.sort(key=_inner)
    rprint(f"Found folders: {folders}")
    input_dir = folders[-1]
    return input_dir

def is_sweep():
    try:
        subdir = HydraConfig.get().sweep.subdir
        rprint(f"Found sweep subdir: {subdir}")
        return True
    except omegaconf.errors.InterpolationToMissingValueError:
        return False
    
def get_sweep_run_name(config):
    try:
        subdir = HydraConfig.get().sweep.subdir
        sweep_str = f"{subdir}_"
        is_sweep = True
    except omegaconf.errors.InterpolationToMissingValueError:
        is_sweep = False
        sweep_str = f"{os.environ.get('SLURM_JOB_ID', '')}_"

    if getattr(config, "training", None) is not None and getattr(getattr(config, "training", None), "force_keys", None) is not None:
        rprint("Using legacy keys")
        forced_keys = set(config.training.force_keys)
    else:
        forced_keys = set(getattr(config.trainer, "forced_keys", []))

    if is_sweep:
        print(
            f"Getting sweep keys: {HydraConfig.get().job.sweep_keys}, Tasks: {HydraConfig.get().overrides.task}, {getattr(config.trainer, 'forced_keys', [])}"
        )
        valid_keys = set(HydraConfig.get().job.sweep_keys)
        for task in HydraConfig.get().overrides.task:
            if task.removeprefix("+").split("=")[0] in valid_keys or task.removeprefix("+").split("=")[0] in forced_keys:
                sweep_str += f"{task.removeprefix('+').split('=')[0].split('.')[-1]}={task.removeprefix('+').split('=')[1]}__"
                if task.removeprefix("+").split("=")[0] in forced_keys:
                    forced_keys.remove(task.removeprefix("+").split("=")[0])
                    print(f"Forced key: {task.removeprefix('+').split('=')[0]}={task.removeprefix('+').split('=')[1]}")

    for key in sorted(list(forced_keys)):
        sweep_str += f"{key.split('.')[-1]}={OmegaConf.select(config, key)}__"

    rprint(f"Sweep: {is_sweep=}, {sweep_str=}")
    return "" if sweep_str == "" else sweep_str[:-2]

def save_config_to_ckpt(config, output_dir, model):
    with try_except(write_error_to_file=True, clear_cuda_cache=True):
        with read_write(config):
            with open_dict(config):
                config.state.ckpt_step = model.global_step
                config.state.num_evals = model.num_evals

        OmegaConf.save(config=config, f=Path(output_dir) / "config.yaml")
        rprint(f"Saved global step {model.global_step}")

def determine_ckpt(config):
    has_recent_ckpt = False
    rprint(f"Looking at checkpoint path: {getattr(config.checkpointing, 'resume_ckpt_path', None)}")
    if (
        config.checkpointing.resume_from_ckpt
        and (latest_ckpt := get_latest_ckpt(config, getattr(config.checkpointing, "resume_ckpt_path", None))) is not None
        and (Path(latest_ckpt) / "config.yaml").exists()
    ):
        ckpt_path = latest_ckpt
        has_recent_ckpt = True
        if config.slurm:
            config.wandb.resume = "must"
        rprint(f"Resuming from checkpoint {ckpt_path}")
    elif config.checkpointing.resume_from_ckpt and getattr(config.checkpointing, "initial_resume_ckpt_path", None) is not None:
        ckpt_path = config.checkpointing.initial_resume_ckpt_path
        rprint(f"Resuming from initial checkpoint {ckpt_path}")
    else:
        ckpt_path = None

    if ckpt_path is not None and (config.checkpointing.resume_wandb or has_recent_ckpt):
        loaded = OmegaConf.load(Path(ckpt_path) / "config.yaml")
        if loaded.wandb.id is not None:
            config.wandb.id = str(loaded.wandb.id)
            rprint(f"Found wandb id: {config.wandb.id}")
        else:
            rprint(f"No wandb id found in checkpoint {ckpt_path}")

    if config.checkpointing.resume_wandb and config.wandb.id is not None:
        config.wandb.resume = "must"
        rprint(f"Resuming wandb, setting must, run id: {config.wandb.id}")
    elif config.slurm and config.wandb.id is None:
        if os.environ.get("SLURM_ARRAY_TASK_COUNT", "") != "" and int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "")) > 1:
            config.wandb.id = str(os.environ.get("SLURM_ARRAY_JOB_ID")) + f"_{os.environ.get('SLURM_ARRAY_TASK_ID')}"
        else:
            config.wandb.id = str(os.environ.get("SLURM_JOB_ID"))
        rprint(f"Setting wandb id to {config.wandb.id}")

    if config.checkpointing.initial_resume_ckpt_path is not None and config.checkpointing.resume_wandb:
        assert config.wandb.id is not None

    if config.ckpt is not None:
        ckpt_path = config.ckpt
        rprint(f"Running eval with checkpoint {ckpt_path}")

    if config.wandb.id is not None:
        config.wandb.id = str(config.wandb.id)

    if config.wandb.id is None or getattr(config.trainer, "force_new_wandb_id", False):
        config.wandb.id = wandb.util.generate_id()
        config.wandb.resume = "allow"
        rprint(f"Set wandb id: {config.wandb.id}")

    rprint(f"Using wandb id: {config.wandb.id}")
    subdir = get_sweep_run_name(config)
    rprint(f"Wandb name: {config.wandb.name}, Wandb subdir: {subdir}")

    if config.wandb.name == 'default':
        config.wandb.name = None
    else:
        config.wandb.name = (
            (f"{config.wandb.name}_" if config.wandb.name else "")
            + (f"{subdir}_" if (subdir is not None and subdir != "") else "")
            + f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )

    if getattr(config.wandb, "group", None) is None and subdir is not None and config.debug and os.environ.get("SLURM_ARRAY_JOB_ID", "") != "":
        config.wandb.group = os.environ.get("SLURM_ARRAY_JOB_ID")
        rprint(f"Wandb group: {config.wandb.group}")

    return ckpt_path

def run(config, tokenizer):
    import torch
    from accelerate import (Accelerator, DataLoaderConfiguration,
                            DDPCommunicationHookType,
                            DistributedDataParallelKwargs,
                            FullyShardedDataParallelPlugin)
    from accelerate.state import AcceleratorState
    from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration

    set_torch_defaults(config.trainer.benchmark)

    set_env_vars(config)
    update_config_before_resolution(config)
    ckpt_path = determine_ckpt(config)
    OmegaConf.resolve(config)
    if is_torch_cuda_available():
        check_gpu_memory_usage()

    if is_torch_cuda_available():
        rprint(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        rprint(f"GPU={torch.cuda.get_device_name()}, device compute capabilities={torch.cuda.get_device_capability()}, pytorch compute capabilities={torch.cuda.get_arch_list()}")
    elif is_torch_xla_available():
        rprint(f"XLA Devices={get_tpu_devices()}")

    rprint(
        f"Initial GROUP_RANK: {os.environ.get('GROUP_RANK', 'N/A')}, RANK: {os.environ.get('RANK', 'N/A')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}, MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'N/A')}, MASTER_PORT: {os.environ.get('MASTER_PORT', 'N/A')}, TORCHELASTIC_RUN_ID: {os.environ.get('TORCHELASTIC_RUN_ID', 'N/A')}, TORCHELASTIC_RESTART_COUNT: {os.environ.get('TORCHELASTIC_RESTART_COUNT', 'N/A')}, TORCHELASTIC_MAX_RESTARTS: {os.environ.get('TORCHELASTIC_MAX_RESTARTS', 'N/A')}, LOCAL_WORLD_SIZE: {os.environ.get('LOCAL_WORLD_SIZE', 'N/A')}, Elastic: {torch.distributed.is_torchelastic_launched()}"
    )
    rprint(f"Computed Rank: {get_rank()}, Local Rank: {get_local_rank()}, World Size: {get_world_size()}")

    # This lets us have start_timing and end_timing functions and a global enable/disable
    # We always use torch.cuda.synchronize before/after as otherwise the timing is not very meaningful
    sync_timing = (config.trainer.nvtx_profile and getattr(config.trainer, "sync_nvtx_timing", True)) or getattr(config.trainer, "sync_timing", False)
    set_timing_builtins(enable=config.trainer.nvtx_profile, sync=sync_timing)

    num_nodes = config.trainer.num_nodes
    with open_dict(config):
        config.trainer = OmegaConf.merge(config.trainer, dict(mixed_precision=config.trainer.precision, log_with="wandb", log_gradients=None))
    if getattr(config.trainer, "process_dataloader_only", False):
        gprint("Processing dataloader only")
        train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer, device="cpu", skip_train=(config.mode == 'eval' and not config.eval.val_with_train_data))
        gprint(f"Exiting after processing dataloader")
        return

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=config.logging_dir,
        automatic_checkpoint_naming=config.checkpointing.use_automatic_naming,
        save_on_each_node=False,
    )

    accelerate_kwargs = dict()
    gradient_kwargs = dict()
    if config.trainer.fsdp and not (config.trainer.xla_spmd and is_torch_xla_available()):
        rprint("Using FSDP...")
        if config.backbone == "llama" or config.backbone == "gemma":
            os.environ["ACCELERATE_USE_FSDP"] = "true"
            os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
            os.environ["FSDP_BACKWARD_PREFETCH"] = "NO_PREFETCH" # Saved memory
            os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
            os.environ["FSDP_FORWARD_PREFETCH"] = "false"
            os.environ["FSDP_OFFLOAD_PARAMS"] = "false"
            os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"
            os.environ["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"
            os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
            os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
            fsdp_plugin = FullyShardedDataParallelPlugin()
        else:
            os.environ["ACCELERATE_USE_FSDP"] = "true"
            os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"  # or "SIZE_BASED_WRAP"
            if config.backbone == "elm":
                os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "OpenELMDecoderLayer"
                os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE" 
                os.environ["FSDP_SHARDING_STRATEGY"] = "HYBRID_SHARD_ZERO2"
            else:
                # Fastest but requires more memory: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch
                os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE" 
                # See: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy
                os.environ["FSDP_SHARDING_STRATEGY"] = "HYBRID_SHARD_ZERO2"
                os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "DDiTBlock" 

            # SHARDED_STATE_DICT is a bit faster, but more complicated as later on we need to merge the shards.
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullOptimStateDictConfig, FullStateDictConfig)
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True), # SHARDED_STATE_DICT
            )

        if config.trainer.compile or config.trainer.use_orig_params is True:
            # https://github.com/huggingface/transformers/pull/24591/files
            fsdp_plugin.use_orig_params = True
            rprint("Using orig params for FSDP. This is required for torch.compile to work.")

        accelerate_kwargs["fsdp_plugin"] = fsdp_plugin
        gradient_kwargs["sync_each_batch"] = False

        if getattr(config.trainer, "fsdp_sync_each_batch", False): # Reduce memory usage: https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization#nosync-requires-additional-gpu-memory-when-using-fsdp
            rprint("Using sync each batch for Chameleon")
            gradient_kwargs["sync_each_batch"] = True

    elif config.trainer.xla_spmd is False: # For XLA FSDP, we init where we normally prepare()
        rprint("Using DDP...")
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=config.trainer.find_unused_parameters,
            comm_hook=DDPCommunicationHookType.BF16,
            static_graph=config.trainer.accumulate_grad_batches == 1,
            gradient_as_bucket_view=True,
        )
        # bucket_cap_mb=32,

        # Not needed right now
        from datetime import timedelta

        from accelerate.utils import InitProcessGroupKwargs
        init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        accelerate_kwargs["kwargs_handlers"] = [ddp_kwargs, init_process_group_kwargs]
    else:
        rprint(f"Did not choose DDP or FSDP.")

    if config.trainer.accumulate_grad_batches <= 0:
        gprint("WARNING!!!!!! Accumulate grad batches is <= 0, setting to 1")
        config.trainer.accumulate_grad_batches = 1

    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=config.trainer.accumulate_grad_batches,
        adjust_scheduler=False, # We manually adjust our LR for accumulate_grad_batches
        sync_with_dataloader=False,
        **gradient_kwargs
    )

    if config.trainer.mixed_precision == "bf16" and (is_torch_cuda_available() and not torch.cuda.is_bf16_supported()):
        rprint(f"No BF16 GPU found, falling back to FP16")
        config.trainer.mixed_precision = "fp16"

    if config.trainer.mixed_precision == "fp32":
        config.trainer.mixed_precision = "no"
    else:
        if is_torch_xla_available():
            os.environ["ACCELERATE_DOWNCAST_BF16"] = "true"

    rprint(f"Mixed precision: {config.trainer.mixed_precision}")

    if config.seed is None or getattr(config.eval, 'set_random_gen_seed', False):
         # do not ask why, has to do something with seeds being reset by val_epoch_end so if you don't execute this code, your generations in val_epoch_end will be same across gpus
        accelerate_kwargs["rng_types"] = []
        rprint("No seed provided, disabling accelerate RNG synchronization")

    accelerator = Accelerator(
        mixed_precision=config.trainer.mixed_precision,
        log_with=config.trainer.log_with,
        project_config=accelerator_project_config,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False, non_blocking=False),
        **accelerate_kwargs,
    )

    gprint(f"Distributed Type: {accelerator.distributed_type}, Accelerator state: {AcceleratorState()}")
    num_processes = AcceleratorState().num_processes
    if getattr(config.trainer, "global_num_warmup_steps", None) is not None:
        rprint(f"Global num_warmup_steps was: {config.lr_scheduler.num_warmup_steps}. Applying to num_warmup_steps")
        config.lr_scheduler.num_warmup_steps = config.trainer.global_num_warmup_steps

    if getattr(config.trainer, "global_num_training_steps", None) is not None:
        rprint(f"Global num_training_steps was: {config.lr_scheduler.num_training_steps}. Applying to num_training_steps")
        config.lr_scheduler.num_training_steps = config.trainer.global_num_training_steps

    if not config.trainer.disable_adjust_num_warmup_steps:
        rprint(f"Original num_warmup_steps was: {config.lr_scheduler.num_warmup_steps}")
        config.lr_scheduler.num_warmup_steps = config.lr_scheduler.num_warmup_steps * num_processes
        rprint(f"Setting num_warmup_steps to: {config.lr_scheduler.num_warmup_steps}")

        if hasattr(config.lr_scheduler, "num_training_steps"):
            rprint(f"Original num_training_steps was: {config.lr_scheduler.num_training_steps}")
            config.lr_scheduler.num_training_steps = config.lr_scheduler.num_training_steps * num_processes
            rprint(f"Setting num_training_steps to: {config.lr_scheduler.num_training_steps}")

    assert config.trainer.allow_dynamic_nodes or (os.environ.get("XLA_USE_SPMD", "0") == "1") or accelerator.num_processes == (
        config.trainer.devices * num_nodes
    ), f"Expected {config.trainer.devices * num_nodes} GPUs but got {accelerator.num_processes} processes."

    compute_dtyle = torch.float32
    if accelerator.mixed_precision == "fp16":
        compute_dtyle = torch.float16
    elif accelerator.mixed_precision == "bf16":
        compute_dtyle = torch.bfloat16

    if compute_dtyle != torch.bfloat16:
        rprint(f"WARNING!!!! Compute dtype is: {compute_dtyle}")
    else:
        rprint(f"Compute dtype is: {compute_dtyle}")

    if is_main_process():
        instantiate_wandb(config, accelerator)

    run_cmd = get_run_cmd(config)
    with open_dict(config):
        config.trainer.devices = accelerator.num_processes
        config.trainer.dtype = str(compute_dtyle)
        if hasattr(config, "state"):
            config.state.cmd = run_cmd
        else:
            config.state = OmegaConf.create(dict(cmd=run_cmd))

    OmegaConf.set_readonly(config, True)

    if getattr(config.trainer, "attach_oom_observer", False):
        from torchtnt.utils.oom import attach_oom_observer
        attach_oom_observer(output_dir=str(os.getcwd()), trace_max_entries=500000)
        rprint(f"Attached OOM observer to {os.getcwd()}")
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer, device=accelerator.device, skip_train=(config.mode == 'eval' and not config.eval.val_with_train_data))
    model = Diffusion(config=config, tokenizer=valid_ds.tokenizer, device=accelerator.device)

    if is_main_process():
        print_params(model.backbone)

    try:
        if getattr(config.model, "image_model", False) is False:
            _print_batch(train_ds, valid_ds, tokenizer)
    except:
        pass

    get_ema_path = lambda x: Path(x) / "ema.ckpt"
    SAMPLER_NAME = "weighted_dataset_sampler"

    def save_model_hook(models, weights, output_dir):
        nonlocal model, accelerator, train_ds

        if is_main_process():
            with try_except(write_error_to_file=True):
                if getattr(model, "ema", None) is not None:
                    torch.save(accelerator.unwrap_model(model).ema.state_dict(), get_ema_path(output_dir))
                    rprint(f"Saved EMA to {get_ema_path(output_dir)}")

            save_config_to_ckpt(config, output_dir, model)

            with try_except(write_error_to_file=True):
                if config.data.use_weighted_tensordict_sampler:
                    from accelerate.utils import save
                    output_sampler_file = output_dir.joinpath(f"{SAMPLER_NAME}_train.bin")
                    save(train_ds.sampler.state_dict(), output_sampler_file, save_on_each_node=False, safe_serialization=False)
                    rprint(f"Sampler state for dataloader saved in {output_sampler_file}")

    initial_global_step = None
    def load_model_hook(models, input_dir):
        nonlocal initial_global_step, model, train_ds
        config_path = os.path.join(input_dir, "config.yaml")
        ckpt_config = OmegaConf.load(config_path)
        initial_global_step = ckpt_config.state.ckpt_step
        model.global_step = initial_global_step
        try:
            if hasattr(config.state, "num_evals"):
                model.num_evals = config.state.num_evals
        except Exception as e:
            rprint(f"Error loading model: {e}")
        rprint(f"Loaded global step {initial_global_step}")

        state_dict = None
        if getattr(config.checkpointing, "load_from_old_attention_format", False):
            state_dict = load_file(os.path.join(input_dir, "model.safetensors"))
            state_dict = convert_state_dict_keys(state_dict)

        if getattr(model, "ema", None) is not None:
            if get_ema_path(input_dir).exists():
                rprint(f"Loading EMA from {get_ema_path(input_dir)}")
                model.ema.load_state_dict(torch.load(get_ema_path(input_dir), map_location='cpu'))
            else:
                rprint(f"No EMA found, initializing EMA with state_dict")
                if state_dict is None:
                    state_dict = load_file(os.path.join(input_dir, "model.safetensors"))

                # We likely don't need the unwrap, but just to be safe
                accelerator.unwrap_model(models[0]).load_state_dict(state_dict)
                from models.ema import EMAModel
                model.ema = EMAModel(accelerator.unwrap_model(models[0]).parameters(), decay=config.trainer.ema)

        if config.data.use_weighted_tensordict_sampler and not is_torch_xla_available(): # and not config.eval.test_eval_speed:
            input_sampler_file = Path(input_dir).joinpath(f"{SAMPLER_NAME}_train.bin")
            if train_ds is not None and input_sampler_file.exists():
                train_ds.sampler.load_state_dict(torch.load(input_sampler_file))
            rprint("All dataloader sampler states loaded successfully")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    model.init_dataloader(train_ds, valid_ds)
    model.set_accelerator(accelerator, ckpt_path)
    model.set_callbacks()

    if getattr(config.checkpointing, "load_from_text_model", None) is not None:
        rprint(f"Loading from text model")
        model.custom_load_checkpoint()

    if getattr(config.checkpointing, "load_from_lightning_ckpt", None) is not None:
        ckpt = torch.load(config.checkpointing.load_from_lightning_ckpt)
        initial_global_step = ckpt["global_step"]
        state_dict_ = {k.removeprefix("backbone."): v for k, v in ckpt["state_dict"].items() if "backbone" in k}
        state_dict_ = {k.replace(".attn_", ".attention.attn_"): v for k, v in state_dict_.items()}
        accelerator.unwrap_model(model.backbone).load_state_dict(state_dict_)

        if config.trainer.ema > 0:
            model.ema.load_state_dict(ckpt["ema"])

        rprint(f"Loaded lightning ckpt: {config.checkpointing.load_from_lightning_ckpt}")

    if initial_global_step is not None:
        # The load_hooks are before accelerate does it's loading and it overwrites model.global_step if we set it there
        model.global_step = initial_global_step
        rprint(f"Set global step to {initial_global_step}")

    contexts = []
    if config.trainer.nvtx_profile:
        contexts.append(torch.autograd.profiler.emit_nvtx(record_shapes=True))

    if config.trainer.profile_memory:
        contexts.append(profile_memory())

    using_torch_elastic = torch.distributed.is_torchelastic_launched()
    if using_torch_elastic:
        rprint(f"Torchelastic launched: {torch.distributed.is_torchelastic_launched()}")
        contexts.append(ErrorHandler())

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)

        rprint(f"output_dir: {config.output_dir}")
        model.to(accelerator.device)
        if config.mode == 'train':
            model.train()
        elif config.mode == 'eval':
            if config.eval.standalone_fid:
                model.validate(None)
            else:
                model.validate(None)
        elif config.mode == 'zero-shot-eval':
            model.zero_shot_eval()
        else:
            raise ValueError(f"Invalid mode: {config.mode}")

    accelerator.end_training()


def get_run_cmd(config):
    orig_argv = deepcopy(sys.argv)

    prepend_argv = []
    if "HYDRA_RUN_DIR" in os.environ:
        prepend_argv.append(f"HYDRA_RUN_DIR='{os.environ['HYDRA_RUN_DIR']}'")
    else:
        prepend_argv.append(f"HYDRA_RUN_DIR='{str(Path(config.output_dir).resolve())}'")

    if orig_argv[1].startswith("experiments=["):
        orig_argv[1] = orig_argv[1].removeprefix("experiments=[").removesuffix("]")
        orig_argv[1] = f"experiments=\'[{orig_argv[1]}]\'"

    if os.environ.get("CUSTOM_SBATCH_LAUNCHER", "0") == "1":
        sbatch_script_path = 'scripts/slurm.sh'
        orig_argv.pop(0)
        orig_argv = ["sbatch", f"--nodes={os.environ.get('SLURM_NNODES', '1')}", f"--gpus-per-node={os.environ.get('SLURM_GPUS_PER_NODE', '1')}", f"--partition={os.environ.get('SLURM_JOB_PARTITION', 'all')}", sbatch_script_path] + orig_argv
    else:
        prepend_argv.append("accelerate launch")

    full_run_cmd = " ".join(prepend_argv + orig_argv)
    rprint(f"Full run cmd: {full_run_cmd}")
    return full_run_cmd

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
@try_except()
def main(config):
    if is_sweep():
        print(f"Checking if we need to requeue for job id {os.environ['SLURM_JOB_ID']}")
        from unidisc.utils.slurm_requeue import check_requeue
        check_requeue()
        print(f"Done checking if we need to requeue for job id {os.environ['SLURM_JOB_ID']}.")

    """Main entry point for trainer."""
    import torch  # Causes issue pickling if imported by default.
    if is_torch_xla_available():
        builtins.HAS_XLA_SPAWNED = True
        os.environ['PJRT_DEVICE'] = 'TPU'

        if config.trainer.precision == "bf16":
            os.environ['XLA_USE_BF16'] = '1'

        if config.devices == 1 and config.trainer.xla_spmd is False and config.trainer.fsdp is False:
            os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
            os.environ['TPU_VISIBLE_CHIPS'] = '0'
            gprint(f"Setting TPU_PROCESS_BOUNDS: {os.environ['TPU_PROCESS_BOUNDS']}")
            gprint(f"Setting TPU_VISIBLE_CHIPS: {os.environ['TPU_VISIBLE_CHIPS']}")

        if config.trainer.tpu_eager:
            os.environ['XLA_USE_EAGER_DEBUG_MODE'] = '1'

        if config.trainer.tpu_compile_debug:
            os.environ['PT_XLA_DEBUG'] = '1'
            os.environ['PT_XLA_DEBUG_LEVEL'] = '2'
            os.environ['XLA_IR_DEBUG'] = '1'
            os.environ['XLA_HLO_DEBUG'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            os.environ['TF_CPP_VMODULE'] = 'xla_graph_executor=5,pjrt_computation_client=3'

        # We intentionally set these after to avoid import side effects
        spmd_mesh, axis_names, num_nodes = None, None, None
        if config.trainer.xla_spmd:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.spmd as xs
            import torch_xla.runtime as xr
            from accelerate import PartialState
            from torch_xla._internal import tpu
            auto_spmd = getattr(config.trainer, "auto_spmd", False)

            xr.use_spmd(auto=auto_spmd) # Auto causes a crash
            force_global_devices = getattr(config.trainer, "force_global_devices", None)
            force_local_devices = getattr(config.trainer, "force_local_devices", None)
            assert (force_global_devices is None) == (force_local_devices is None), "Must set both or neither"

            if force_global_devices is not None:
                num_global_devices = force_global_devices
                num_local_devices = force_local_devices
                gprint(f"Using force global devices: num_global_devices={num_global_devices}, num_local_devices={num_local_devices}")
            else:
                num_global_devices = xr.global_runtime_device_count()
                num_local_devices = tpu.num_available_devices()
                assert num_global_devices == tpu.num_expected_global_devices()
                assert tpu.num_available_devices() == tpu.num_available_chips() == tpu.num_local_processes()

            num_nodes = num_global_devices // num_local_devices
            spmd_mesh_shape = getattr(config.trainer, "spmd_mesh", None)
            if spmd_mesh_shape is None:
                spmd_mesh_shape = (num_nodes, num_local_devices, 1)

            if getattr(config.trainer, "force_disable_replicas", False):
                spmd_mesh_shape = (1, num_global_devices, 1)
                rprint(f"Forcing disable replicas: {spmd_mesh_shape}")

            if auto_spmd is False:
                if getattr(config.trainer, "spmd_multislice", None) is not None:
                    from torch_xla.distributed.spmd import HybridMesh
                    ici_mesh_shape = spmd_mesh_shape
                    dcn_mesh_shape = (config.trainer.spmd_multislice, 1, 1)
                    spmd_mesh = HybridMesh(ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape, axis_names=('data','fsdp','tensor'))
                    rprint(f"Using multislice: {config.trainer.spmd_multislice}: {ici_mesh_shape} {dcn_mesh_shape}, {spmd_mesh.shape()}")
                else:
                    spmd_mesh = xs.Mesh(np.array(range(num_global_devices)), spmd_mesh_shape, ('dcn', 'fsdp', 'model'))
                xs.set_global_mesh(spmd_mesh)

            config.devices = 1
            config.nodes = 1

            with read_write(config):
                with open_dict(config):
                    config.state = OmegaConf.create(dict(spmd_mesh=spmd_mesh_shape))
                    config.state.axis_names = axis_names
                    config.state.num_nodes = num_nodes
                    config.state.num_global_devices = num_global_devices
                    config.state.num_local_devices = num_local_devices
                    config.state.worker_ips = tpu.get_worker_ips()
                    if os.environ.get("TPU_NAME") is not None:
                        config.state.tpu_name = os.environ.get("TPU_NAME")

        if config.trainer.tpu_eager:
            import torch_xla
            torch_xla.experimental.eager_mode(True)

        if config.trainer.tpu_profile:
            if config.trainer.tpu_profile_markers:
                os.environ['XLA_IR_DEBUG'] = '1'
                os.environ['XLA_HLO_DEBUG'] = '1'
            import torch_xla.debug.profiler as xp
            server = xp.start_server(9012)

        if config.trainer.tpu_cache:
            import torch_xla.runtime as xr
            readonly = not is_main_process()
            rprint(f"Initializing TPU cache with readonly={readonly}")
            xr.initialize_cache(str((Path.home() / '.cache' / 'unidisc' / f"tpu_{get_rank()}_{get_hostname().replace('-', '_')}").resolve()), readonly=readonly)

        if config.trainer.enable_jax_smi:
            initialise_tracking()
            rprint("Initializing jax-smi")

    from unidisc.utils.logging_utils import set_logger
    set_logger(f"{__name__} {get_slurm_log_prefix()}", Path(f"{get_slurm_filename_info()}_{get_hostname().replace('-', '_')}.out"))

    if is_torch_xla_available():
        import torch_xla.runtime as xr
        gprint(
                f"Computed Rank: {get_rank()}, "
                f"Is Main Process: {is_main_process()}, "
                f"Is Local Main Process: {is_local_main_process()}, "
                f"XLA world size: {xr.world_size()}, "
                f"XLA Model Ordinal: {xm.get_ordinal()}, "
                f"XLA Global Ordinal: {xr.global_ordinal()}, "
                f"XLA Supported Devices: {xm.get_xla_supported_devices()}, "
                f"Accelerate Local Process Index: {PartialState().local_process_index}, "
                f"Task ID: {tpu.task_id()}, "
                f"Worker ID: {tpu.worker_id()} "
                f"global device count: {xr.global_runtime_device_count()}, "
                f"local process count: {xr.local_process_count()}, "
                f"local device count: {xr.local_device_count()}, "
                f"addressable device count: {xr.addressable_device_count()}, "
                f"num_expected_global_devices: {tpu.num_expected_global_devices()}, "
                f"num_available_devices: {tpu.num_available_devices()}, "
                f"num_available_chips: {tpu.num_available_chips()}, "
                f"num_local_processes: {tpu.num_local_processes()}, "
                f"process_bounds_size: {tpu.process_bounds_size()}, "
                f"get_worker_ips: {tpu.get_worker_ips()}, "
                f"Computed Num Nodes: {num_nodes}, "
                f"Specified Mesh: {spmd_mesh_shape}, "
                f"Specified Mesh Axes: {axis_names}"
            )

        gprint(f"LIBTPU_INIT_ARGS: {os.environ.get('LIBTPU_INIT_ARGS', 'None')}")
        gprint(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'None')}")
    
    if getattr(config.trainer, "disable_ddp_optimizer", False):
        torch._dynamo.config.optimize_ddp = False

    if config.seed is not None:
        if config.mode == 'eval':
            config.seed = config.seed + 1000 * int(get_rank())
        else:
            config.seed = config.seed + int(get_rank())
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if is_torch_cuda_available():
            # TODO: Is seed all desired? Does it set the same one on all GPUs even in multi-process?
            torch.cuda.manual_seed_all(config.seed)

        if is_torch_xla_available():
            import torch_xla.core.xla_model as xm
            xm.set_rng_state(config.seed)
        gprint(f"Set seed: {config.seed}")
    else:
        rprint("No seed provided")

    _print_config(config, resolve=True, save_cfg=True)

    with open(f"env_vars_{get_slurm_filename_info()}_{get_hostname().replace('-', '_')}.txt", "w") as f:
        for key, value in os.environ.items():
            f.write(f"{key}={value}\n")

    tokenizer = dataloader.get_tokenizer(config)

    if "tokens" in config.data.train and (config.loader.num_workers > 0 or getattr(config.data, "force_mp_spawn", False)):
        from torch import multiprocessing as mp
        try:
            rprint(f"Start already method set to: {mp.get_start_method()}")
        except:
            mp.set_start_method("spawn")
            rprint(f"Start method set to: {mp.get_start_method()}")

    rprint(f"Mode: {config.mode}")
    if config.mode == "sample_eval":
        generate_samples(config, tokenizer)
    else:
        try:
            run(config, tokenizer)
        except Exception as e:
            rprint(f"Traceback: {traceback.format_exc()}")
            rprint(f"Exception: {e}")
    
            timestamp = int(__import__("time").time_ns())
            error_filepath = f"exception_{timestamp}_{process_file_prefix()}.out"
            with open(error_filepath, "w") as file:
                file.write(traceback.format_exc())
            rprint(f"See error file {Path(error_filepath).resolve()} for traceback")
            
            if is_torch_xla_available():
                exit(1)

            if ("SLURM_JOB_ID" not in os.environ) and ("RESTART_FAULT_TOLERANT" not in os.environ) and not is_torch_xla_available():
                gprint(f"Entering debugger")
                breakpoint(traceback=e.__traceback__)
            else:
                rprint(f"Not breaking, SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}, RESTART_FAULT_TOLERANT: {os.environ.get('RESTART_FAULT_TOLERANT')}")

            if "RESTART_FAULT_TOLERANT" in os.environ:
                sigterm_handler = signal.getsignal(signal.SIGTERM)
                if callable(sigterm_handler):
                    rprint(f"Calling SIGTERM handler")
                    sigterm_handler(signal.SIGTERM, None)

                try:
                    if config.trainer.num_nodes > 1 and config.debug is False and is_main_process():
                        wandb.alert(title="Exception!", text=f"{e}, {traceback.format_exc()}")
                except:
                    pass
            raise e
        finally:
            pass

if __name__ == "__main__":
    main()
