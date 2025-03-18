from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import extract_model_from_parallel
from image_utils import Im

from decoupled_utils import is_main_process, gprint, rprint

log_info = gprint
log_error = gprint


def load_from_ckpt(cfg, accelerator: Optional[Accelerator], model: nn.Module, load_model: bool, load_accelerator_state: bool = False) -> int:
    """
    Loads the model [or just returns the checkpoint global step]
    """
    if cfg.trainer.ckpt == "latest":
        # Get the most recent checkpoint
        dirs = os.listdir(cfg.checkpoint_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
        path = dirs[-1] if len(dirs) > 0 else None
    else:
        path = Path(cfg.trainer.ckpt)

    if path.is_dir() and any(child.is_dir() and child.name == "state" for child in path.iterdir()):
        path = path / "state"

    if path is None:
        log_error(f"Checkpoint '{cfg.trainer.ckpt}' does not exist. Exiting.")
        raise FileNotFoundError
    else:
        log_info(f"Resuming from checkpoint {path}")

        # TODO: @Tsung-Wei Ke tested this and found that it doesn't work, at least in some of the cases we used.
        # We should see if we can still load the optimizer states.

        # from accelerate.utils.modeling import load_checkpoint_in_model
        # if path.is_file() or cfg.trainer.load_weights_only_no_state:
        #     load_checkpoint_in_model(model, str(path))
        # else:
        
        if load_model:
            if accelerator is not None and path.parent.stem == "state" and load_accelerator_state:
                log_info("Loading accelerator state!")
                accelerator.load_state(path.parent)
        
            state_dict = torch.load(path, map_location='cpu')
            if cfg.trainer.ignore_clip_weights:
                state_dict = {k:v for k,v in state_dict.items() if 'clip' not in k and 'mapper.position_embedding' not in k and 'up_proj' not in k}
            if cfg.trainer.ignore_pos_emb_weights:
                state_dict = {k:v for k,v in state_dict.items() if 'cross_attn_pos_emb' not in k}
            model.load_state_dict(state_dict, strict=cfg.trainer.strict_load)
        try:
            if path.is_file():
                global_step = int(path.parent.parent.name.split("_")[-1])
            else:
                global_step = int(path.name.split("_")[-1] if "_" in path.name else path.parent.name.split("_")[-1])
        except:
            log_error(f"Could not parse global step from checkpoint path {path}. Setting to 0.")
            global_step = 0

        # first_epoch = global_step // num_update_steps_per_epoch
        first_epoch = 0
        log_info(f"Continuing from epoch {first_epoch} and global step {global_step}")
        return global_step


def handle_checkpointing_dirs(cfg, prefix: str):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if cfg.checkpointing.checkpoints_total_limit is not None:
        if not os.path.exists(cfg.checkpointing.save_dir):
            os.makedirs(cfg.checkpointing.save_dir, exist_ok=True)
            
        checkpoints = os.listdir(cfg.checkpointing.save_dir)
        checkpoints = [
            d for d in checkpoints 
            if d.startswith(f"{prefix}_") 
            and len(os.listdir(os.path.join(cfg.checkpointing.save_dir, d))) >= 1
            and sum(f.stat().st_size for f in Path(os.path.join(cfg.checkpointing.save_dir, d)).rglob('*')) > 10 * 1024 * 1024
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= cfg.checkpointing.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - cfg.checkpointing.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            log_info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            log_info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(cfg.checkpointing.save_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)


@dataclass
class TrainingState:
    epoch_step: int  # Step in the current epoch. Resets every epoch.
    num_epoch_steps: int  # Total number of steps in the current epoch. [E.g., dataloader size on a single GPU]
    global_step: int  # Current number of steps which does not reset.
    true_step: int
    epoch: int
    split: Optional[Any] = None
    current_run_global_step: Optional[int] = None
    batch: Optional[dict] = None


class Trainable(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict, state: TrainingState) -> dict:
        ...

    @abstractmethod
    def set_training_mode(self):
        ...

    @abstractmethod
    def set_inference_mode(self):
        ...

    @abstractmethod
    def checkpoint(self, accelerator: Accelerator, state: TrainingState, path: Path):
        ...

    def run_inference(self, batch: dict, state: TrainingState, accelerator: Optional[Accelerator] = None) -> dict[str, Im]:
        ...

    def on_sync_gradients(self):
        pass

    def get_param_groups(self) -> Optional[dict[str, Any]]:
        return None
    
    def process_input(self, batch: dict) -> Any:
        return batch


def check_every_n_steps(
    state: TrainingState,
    n: Optional[int],
    run_first: bool = False,
    all_processes: bool = False,
    decay_steps: bool = False,
    max_eval_interval: Optional[int] = None,
    decrease_n_runs: Optional[int] = None,
):
    if n is None or n <= 0: return False
    if decay_steps:
        max_eval_interval = max_eval_interval or n * 2
        decrease_n_runs = decrease_n_runs or 5
        n = min(n * ((state.global_step // (decrease_n_runs * n)) + 1), max_eval_interval)
    
    return ((state.global_step % n == 0 or (state.current_run_global_step is not None and state.current_run_global_step == 0)) and (run_first or (state.global_step > 0 and state.current_run_global_step > 0))) and (is_main_process() or all_processes)


def check_every_n_epochs(state: TrainingState, n: Optional[int], run_first: bool = False, all_processes: bool = False):
    # Check if the current step is the last one in the epoch. We always want to run on the last step of the epoch. If we have n=5, then we run at the end of epochs 0 [if except_first == False], 5, 10, 15, etc.
    return (
        n is not None
        and (state.epoch_step == state.num_epoch_steps - 1)
        and ((state.epoch + 1) % n == 0 or (state.epoch == 0 and run_first))
        and (is_main_process() or all_processes)
    )


def every_n_steps(func, *wrapper_args, **wrapper_kwargs):
    @wraps(func)
    def wrapper(state: TrainingState, *args, **kwargs):
        if check_every_n_steps(state, *wrapper_args, **wrapper_kwargs):
            return func(*args, **kwargs)

    return wrapper


def every_n_epochs(func, *wrapper_args, **wrapper_kwargs):
    @wraps(func)
    def wrapper(state: TrainingState, *args, **kwargs):
        if check_every_n_epochs(state, *wrapper_args, **wrapper_kwargs):
            return func(*args, **kwargs)

    return wrapper


def unwrap(model):
    """
    In DDP/torch.compile and some other situations, our nn.Module is wrapped so to access class attributes we often need to unwrap it.
    """
    # equiv to. unwrap
    if PartialState._shared_state == {}:
        # Accelerate is initialized
        return extract_model_from_parallel(model)
    else:
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            return model.module
        else:
            return model

def linear_warmup(current_step: int, warmup_steps: int, final_value: float, initial_value: float = 0.0, start_step: int = 0):
    current_step = max(0, current_step - start_step)
    if current_step < warmup_steps:
        return initial_value + (final_value - initial_value) * (current_step / max(1, warmup_steps))
    else:
        return final_value

def get_parameters(module):
    params = []
    
    def recursive_collect(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding) or child.__class__.__name__ == "EmbeddingLayer":
                continue
            else:
                params.extend(list(child.parameters(recurse=False)))
                recursive_collect(child)
    
    recursive_collect(module)
    return params

def count_parameters(module):
    return sum(p.numel() for p in get_parameters(module) if p.requires_grad)

def incremental_dict_update(data, new_data):
    data.update(new_data)
    return data
    for key, value in new_data.items():
        if key in data and isinstance(value, torch.Tensor):
            if value.numel() == 1:
                data[key] = (data[key] + value) / 2
            else:
                data[key] = torch.cat([data[key], value])
        else:
            data[key] = value
    return data

if __name__ == "__main__":
    for i in range(50000):
        if check_every_n_steps(TrainingState(epoch_step=i, num_epoch_steps=10, global_step=i, epoch=0, true_step=i), 500, decay_steps=True):
            print(i)