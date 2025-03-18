import functools
import itertools
import os
import signal
import subprocess
import sys
import time
import typing
from functools import partial
from pathlib import Path
from types import FrameType
from contextlib import nullcontext

import transformers
from constants import HF_TOKEN, HF_CACHE_DIR
import hydra
import hydra.utils
import torch
import torch.utils.checkpoint
from accelerate.utils import gather, gather_object
from omegaconf import open_dict, read_write
from safetensors.torch import load_file

import models.noise_schedule as noise_schedule
import utils
import wandb
from decoupled_utils import (barrier, dprint, get_slurm_job_id, get_world_size, gprint,
                             is_local_main_process, is_main_process,
                             is_torch_cuda_available, is_torch_xla_available,
                             module_hash, parameter_hash, print_memory,
                             rank_zero_fn, rprint, save_memory_profile,
                             show_memory_usage, try_except, use_dist)
from unidisc.tokenizers.image_tokenizers import get_vae as tokenizer_get_vae
from unidisc.utils.xla_utils import (tpu_spmd_dataloader, wrap_xla_fsdp)
from model_utils import BPD, NLL, Perplexity, empty_device_cache, log, CIDErScore, Accuracy
from unidisc.utils.trainer_utils import (TrainingState, check_every_n_epochs,
                           check_every_n_steps, handle_checkpointing_dirs, count_parameters)
from utils import compile_model, grad_norm

is_xla_available = is_torch_xla_available()
if is_xla_available:
    rprint("Using standalone torchmetrics on XLA")
    from unidisc.utils.standalone_metrics import MetricCollection
else:
    from torchmetrics import MetricCollection

def init(self, config, tokenizer, device):
    import models
    import models.elm_custom as elm_custom

    self.global_step = 0
    self.current_run_global_step = 0
    self.current_run_fwd_bwd_pass = 0
    self.num_evals = 0

    self.config = config
    self.device = device
    self.image_model = False
    self.unified_model = False

    self.dtype = (
        torch.float32
        if ("fp32" in self.config.trainer.precision or "no" in self.config.trainer.precision)
        else (torch.bfloat16 if "bf16" in self.config.trainer.precision else torch.float16)
    )
    rprint(f"Set compute dtype in model: {self.dtype}")

    if getattr(self.config.model, "image_model", False):
        self.image_model = True
        if "tokens" not in self.config.data.train:
            self.vae = self.get_vae()
            if self.config.data.cond_resolution is not None:
                self.cond_vae = self.get_cond_vae()
        else:
            self.vae = None
            self.cond_vae = None

    if getattr(self.config.model, "unified_model", False):
        self.unified_model = True

    self.tokenizer = tokenizer
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.trainer.antithetic_sampling
    self.importance_sampling = self.config.trainer.importance_sampling
    self.change_of_variables = self.config.trainer.change_of_variables
    if getattr(self.config.trainer, "add_label", False):
        assert self.image_model and self.unified_model
        
    if self.image_model is False or self.unified_model:
        self.vocab_size = len(self.tokenizer)
        if getattr(self.config.model, "force_text_vocab_size", None) is not None:
            self.vocab_size = self.config.model.force_text_vocab_size
        if not hasattr(self.tokenizer, "mask_token") or self.tokenizer.mask_token is None:
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id
    if self.image_model:
        if self.unified_model:
            self.text_vocab_size = self.vocab_size
            self.vocab_size += self.config.model.image_vocab_size
            self.image_vocab_size = self.config.model.image_vocab_size
            if getattr(self.config.model, "add_labels", None) is not None:
                rprint(f"Adding labels: {self.config.model.add_labels}")
                self.vocab_size += self.config.model.add_labels
            rprint(f"Text vocab size: {self.text_vocab_size}, Image vocab size: {self.image_vocab_size}")
        else:
            self.vocab_size = self.config.model.image_vocab_size + 1
            self.mask_index = self.vocab_size - 1
            self.text_vocab_size = 0
    else:
        self.text_vocab_size = self.vocab_size

    rprint(f"Vocab size: {self.vocab_size}, Mask index: {self.mask_index}")
    rprint(f"Image Model: {self.image_model}, Unified Model: {self.unified_model}")
    self.parameterization = self.config.parameterization

    tf_kwargs = dict(device_map=self.device, use_auth_token=HF_TOKEN, torch_dtype=self.dtype if (self.config.model.use_lora or self.config.trainer.low_precision_params) else torch.float32, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    tf_kwargs['attn_implementation'] = 'sdpa' if is_xla_available else 'flash_attention_2'
    force_sdpa_attention = os.environ.get("UNIDISC_FORCE_CHAMELEON_SDPA_ATTENTION", "0") == "1"
    force_eager_attention = os.environ.get("UNIDISC_FORCE_EAGER_ATTENTION", "0") == "1"
    if force_sdpa_attention:
        tf_kwargs['attn_implementation'] = 'sdpa'
        rprint("WARNING!!!! Forcing SDPA Attention")
    if force_eager_attention:
        tf_kwargs['attn_implementation'] = 'eager'
        rprint("WARNING!!!! Forcing Eager Attention")

    if is_xla_available:
        del tf_kwargs['cache_dir']
        rprint(f"Using cache dir: {HF_CACHE_DIR}")
    
    if self.config.backbone == "dit":
        dit_kwargs = dict(mask_index=self.mask_index)
        if getattr(self.config.trainer, "use_orig_unidisc_dit", False):
            from accelerate.utils import set_seed; set_seed(42)
            if self.config.model.full_attention:
                import models.dit_orig
                _backbone_cls = models.dit_orig.DIT
                rprint("WARNING!!!! Using original DIT")
                dit_kwargs.pop('mask_index')
            else:
                import models.autoregressive_orig
                _backbone_cls = models.autoregressive_orig.AR
                dit_kwargs['causal'] = not self.config.model.full_attention
                rprint(f"WARNING!!!! Using original AR DIT, {dit_kwargs}")
        else:
            import models.dit
            _backbone_cls = models.dit.DIT
            dit_kwargs['text_vocab_size'] = self.text_vocab_size
            dit_kwargs['autocast_dtype'] = self.dtype
            dit_kwargs['device'] = self.device
            dit_kwargs['static_img_sl'] = self.static_img_sl
            dit_kwargs['static_txt_sl'] = self.static_txt_sl
            
        self.backbone = _backbone_cls(
            config=self.config,
            vocab_size=self.vocab_size,
            **dit_kwargs
        )
        utils.print_trainable_parameters(self.backbone)
        if self.config.model.mup:
            self.get_base_shapes_for_mup(self.backbone)
    elif self.config.backbone == "elm":
        del tf_kwargs['attn_implementation']
        config = transformers.AutoConfig.from_pretrained(self.config.model.model_id, **tf_kwargs)
        config.extra_tokens = self.vocab_size - config.vocab_size
        config.full_attention = self.config.model.full_attention
        config.is_compiled = self.is_compiled
        _cls = elm_custom.OpenELMForCausalLM if self.config.trainer.scratch else partial(elm_custom.OpenELMForCausalLM.from_pretrained, pretrained_model_name_or_path=self.config.model.model_id)
        self.backbone = _cls(
            config=config,
        )
        if self.config.model.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["qkv_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )            
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.model.transformer.token_embeddings_extra.requires_grad_(True)
            if hasattr(self.backbone.model, "lm_extra"):
                self.backbone.model.lm_extra.requires_grad_(True)
        else:
            self.backbone.requires_grad_(True)
            self.backbone.train()
            rprint("Using Full ELM")

        if getattr(self.config.trainer, "scratch", False):
            rprint("Training from scratch")
            self.backbone.apply(self.backbone._init_weights)
        
        if getattr(self.config.trainer, "use_gradient_checkpointing", False):
            self.backbone.gradient_checkpointing_enable()
        utils.print_trainable_parameters(self.backbone)
    elif self.config.backbone == "ar":
        self.backbone = models.autoregressive.AR(self.config, vocab_size=self.vocab_size, mask_index=self.mask_index)
    else:
        raise ValueError(f"Unknown backbone: {self.config.backbone}")

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking
    self.softplus = torch.nn.Softplus()
    if getattr(self.config.trainer, "disable_torchmetrics", False) is False:
        # metrics are automatically reset at end of epoch
        metrics = MetricCollection(
            {
                "nll": NLL(sync_on_compute=False),
                "bpd": BPD(sync_on_compute=False),
                "ppl": Perplexity(sync_on_compute=False),
            },
            compute_groups=(not is_torch_xla_available() and not getattr(self.config.trainer, "disable_distributed_torchmetrics", False))
        )
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    if getattr(self.config.trainer, "log_seperate_modal_losses", False):
        self.txt_metrics = metrics.clone(prefix="train/")
        self.img_metrics = metrics.clone(prefix="train/")
    
    if getattr(self.config.eval, "compute_chameleon_perplexity", False) or getattr(self.config.eval, "wino_chameleon", False):
        rprint("[INFO] Loading Big Chameleon Model")
        # pip install 'git+ssh://git@github.com/alexanderswerdlow/image_utils.git@wip_v1' --force-reinstall
        from image_utils import Im
        from transformers import (ChameleonForConditionalGeneration, ChameleonProcessor)
        self.chameleon_model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", torch_dtype=torch.bfloat16).to("cuda")
        self.chameleon_processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
    
    if self.config.mode == "zero-shot-eval":
        # flickr cider
        self.cider_score = CIDErScore(sync_on_compute=False)
        
        # winoground
        self.win_text_accuracy = Accuracy(sync_on_compute=False)
        self.win_image_accuracy = Accuracy(sync_on_compute=False)
        self.win_group_accuracy = Accuracy(sync_on_compute=False)
        
        
        self.datacomp_img_acc = Accuracy(sync_on_compute=False)
        self.datacomp_txt_acc = Accuracy(sync_on_compute=False)
    
    self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
        self.eval_model_tokenizer.pad_token = self.eval_model_tokenizer.eos_token
        self.eval_model_tokenizer.pad_token_id = self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
    if self.config.trainer.ema > 0:
        if self.config.trainer.use_custom_ema:
            from copy import deepcopy
            self.ema = deepcopy(self.backbone).eval()
            self.ema.to(self.device)
        else:
            self.ema = models.ema.EMAModel(self.get_params(), decay=self.config.trainer.ema)
        rprint(f"Using EMA with decay {self.config.trainer.ema}")
    else:
        self.ema = None

    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.trainer.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

    self.fid_eval = False

    if ((self.config.slurm or self.config.trainer.restart_on_failure) and not self.config.trainer.force_disable_signal_handler) and self.config.mode == 'train':
        self.register_signal_handler()

    if getattr(self.config.model, "image_model_fid_eval", False) or getattr(self.config.trainer, "disable_strict_load", False):
        self.strict_loading = False

    if self.config.backbone != 'dit' and self.config.backbone != 'chameleon':
        assert self.config.model.force_argmax_valid_indices is False

    if self.config.parameterization == "ar":
        assert self.config.trainer.ar_shift

    self.trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
    self.frozen_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
    self.non_embedding_params = count_parameters(self.backbone)
    rprint(f"Total trainable parameters (excluding embeddings): {self.non_embedding_params:,}, Total trainable parameters: {self.trainable_params:,}, Total frozen parameters: {self.frozen_params:,}")
    self._validate_configuration()

    if not self.config.trainer.low_precision_params:
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                    raise ValueError(f"Parameter {name} is not in fp32. It is in {param.dtype}")
                
    if self.config.eval.test_eval_speed:
        rprint("WARNING!!!! Running eval speed test")
        
    self.use_kv_cache = getattr(self.config.model, "use_kv_cache", False)
    if not getattr(self.config.eval, 'enable_gen_pplx_cleanup', True):
        assert self.config.mode == 'eval' # shouldn't really be on in train mode
        rprint(f"WARNING!!!! Disabling gen pplx cleanup, having eval model {self.gen_ppl_eval_model_name_or_path} in memory always!!!!") 
        self.gen_pplx_eval_model = transformers.AutoModelForCausalLM.from_pretrained(self.gen_ppl_eval_model_name_or_path).eval()
        
    if self.config.eval.compute_standalone_mauve and not getattr(self.config.eval, "global_disable_mauve", False):
        self.mauve_predictions = []
        self.mauve_references = []
        
    if self.config.mode == "zero-shot-eval":
        self.cider_score_metric = CiderScorer()

    if self.config.mode == "eval":
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    if self.config.trainer.awr:
        breakpoint()
        config = transformers.AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M", **tf_kwargs)
        config.vocab_size = self.vocab_size
        config.full_attention = True
        self.awr_policy = llama_custom.LlamaForCausalLM(
            config=config,
        )
        

def to(self, device):
    self.device = device
    self.backbone.to(device)
    self.train_metrics.to(device)
    self.test_metrics.to(device)
    if hasattr(self, "txt_metrics"):
        self.txt_metrics.to(device)
    if hasattr(self, "img_metrics"):
        self.img_metrics.to(device)

    if self.ema is not None:
        self.ema.to(device)

def reset_validation_metrics(self):
    metrics = MetricCollection(
        {
            "nll": NLL(sync_on_compute=False),
            "bpd": BPD(sync_on_compute=False),
            "ppl": Perplexity(sync_on_compute=False),
        },
        compute_groups=(not is_torch_xla_available() and not getattr(self.config.trainer, "disable_distributed_torchmetrics", False))
    )
    metrics.set_dtype(torch.float64)
    
    if getattr(self.config.trainer, "disable_torchmetrics", False) is False or hasattr(self, "valid_metrics"):
        self.valid_metrics = metrics.clone(prefix="val/").to(self.device)

    if getattr(self.config.trainer, "log_seperate_modal_losses", False):
        self.valid_txt_metrics = metrics.clone(prefix="val/").to(self.device)
        self.valid_img_metrics = metrics.clone(prefix="val/").to(self.device)

    self.gen_ppl_metric = Perplexity(sync_on_compute=False).to(self.device)
    self.gt_gen_ppl_metric = Perplexity(sync_on_compute=False).to(self.device)

def get_params(self):
    return itertools.chain(self.backbone.parameters())

def get_vae(self):
    if getattr(self, "vae", None) is not None:
        return self.vae

    empty_device_cache()
    
    self.vae = tokenizer_get_vae(self.config, self.device)

    return self.vae

def get_cond_vae(self):
    if getattr(self, "cond_vae", None) is not None:
        return self.cond_vae

    torch.cuda.empty_cache()
    self.cond_vae = get_vae(self.config, self.device, use_cond=True)
    return self.cond_vae


def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    kwargs = dict(
        betas=(self.config.optim.beta1, self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay,
    )
    if getattr(self.config.trainer, "adafactor", False):
        optim_cls = Adafactor
        kwargs = dict()
        kwargs.update({"scale_parameter": False, "relative_step": False})
        rprint("Using Adafactor")
    if getattr(self.config.trainer, "ademamix", False):
        from unidisc.utils.ademamix import AdEMAMix
        optim_cls = AdEMAMix
        rprint("Using AdEMAMix")
    elif is_xla_available:
        from torch_xla.amp.syncfree import AdamW
        optim_cls = AdamW
        rprint("Using XLA AdamW")
    elif getattr(self.config.trainer, "is_deepspeed", False):
        import deepspeed
        optim_cls = deepspeed.ops.adam.FusedAdam
        kwargs["set_grad_none"] = True
    else:
        optim_cls = torch.optim.AdamW
        kwargs["fused"] = self.config.optim.fused

    if self.config.model.mup:
        from mup import MuAdam
        optim_cls = partial(MuAdam, impl=optim_cls)
        
    optimizer = optim_cls(
        self.get_params(),
        lr=self.config.optim.lr,
        **kwargs,
    )

    scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
        "scheduler": scheduler,
        "interval": "step",
        "monitor": "val/loss",
        "name": "trainer/lr",
    }
    return [optimizer], [scheduler_dict]

def _validate_configuration(self):
    assert not (self.change_of_variables and self.importance_sampling)
    if self.parameterization == "sedd":
        assert not self.importance_sampling
        assert not self.change_of_variables
    if self.parameterization == "d3pm":
        assert self.T > 0
    if self.T > 0:
        assert self.parameterization in {"d3pm", "subs"}
    if self.subs_masking:
        assert self.parameterization == "d3pm"

    if hasattr(self.config.model, "text_vocab_size"):
        assert self.config.model.text_vocab_size == self.text_vocab_size, f"text_vocab_size {self.config.model.text_vocab_size} != {self.text_vocab_size}"

    if getattr(self.config.trainer, "first_token_dropout", None) is not None:
        assert self.config.data.allow_label is True
        assert self.config.trainer.add_label is True
        assert self.config.model.add_labels > 0
        assert self.config.trainer.joint_ar_nar_prob is None
        assert self.config.trainer.mask_entire_modality is None

    if getattr(self.config.eval, "class_conditional_fid", False):
        assert self.config.eval.fid_mode == "inline"

    assert getattr(self.config.model, "mask_entire_modality", None) is None

    if self.config.trainer.interleaved and not getattr(self.config.eval, "auto_enhance", False) and not getattr(self.config.trainer, "bypass_interleaved_check", False):
        assert self.config.data.use_packing_collate or self.config.mode == 'eval'
        assert self.config.data.dynamic_packing_lengths
        assert self.config.data.require_sample_ids
        assert self.config.trainer.interleaved_training_flex_attention
        assert self.config.data.use_slow_tokenizer and self.config.data.add_image_token
        assert not getattr(self.config.trainer, "force_full_attention_mask_loss_only", False)

    assert self.config.sampling.steps == self.config.sampling.max_sampling_steps

def register_signal_handler(self):
    def _handler(sig, frame: FrameType | None, prior_handler=None):
        rprint(f"Called sig handler with {sig=} {self.global_step=}")
        if sig == signal.SIGUSR1:
            signal.signal(sig, signal.SIG_IGN)

        checkpoint_path = Path(self.config.output_dir) / "checkpoints"
        timeout_minutes = self.config.trainer.ckpt_recent_timeout_minutes

        # Don't re-save checkpoint within this interval to avoid unecessary re-writing.
        # If we checkpoint on SIGUSR2, we don't need to do it on SIGTERM
        recent_ckpt_exists = checkpoint_path.exists() and any(
            (time.time() - p.stat().st_mtime) < (timeout_minutes * 60) for p in checkpoint_path.iterdir() if p.is_dir()
        )
        if (self.current_run_global_step > 100 and recent_ckpt_exists is False) or self.config.trainer.skip_early_checkpointing is False:
            rprint(f"Saving checkpoint due to {sig}")
            self.checkpoint()
            rprint(f"Finished saving checkpoint due to {sig}")
        else:
            rprint(f"Checkpoint already saved within {timeout_minutes} minutes, called by {sig}. Current run global step: {self.current_run_global_step}")

        job_str = get_slurm_job_id()
        if is_main_process():
            if sig == signal.SIGTERM:
                if self.current_run_global_step > 100 and self.config.devices >= 4:
                    wandb.alert(title="Terminated", text=f"Terminated by SIGTERM at {self.global_step}")
                rprint("Marking experiment as preempting")
                wandb.mark_preempting()

            rprint(f"Prior handler on rank: {prior_handler}")
            is_custom_sbatch_launcher = os.environ.get("CUSTOM_SBATCH_LAUNCHER", "0") == "1"
            if is_custom_sbatch_launcher:
                rprint("Using custom sbatch launcher, requeueing job manually")
                subprocess.check_call(["scontrol", "requeue", job_str])
                rprint("Finished requeueing job")
            elif prior_handler is not None and callable(prior_handler):
                rprint("Calling prior signal handler")
                prior_handler(sig, frame, exit_on_requeue=False)
                rprint(f"Returned from prior signal handler")
        else:
            # TODO: For some unknown reason, sometimes the main process [and a few others] hangs doesn't properly receive the signal.
            # Generally, we want to let the main process checkpoint/exit but if it fails, we let any rank re-queue.
            if self.config.slurm:
                time.sleep(180)
                rprint(f"WARNING: Not on rank zero!  Timed out waiting for main process to exit...Requeuing job...")
                rprint(f"WARNING: Not on rank zero! Using prior signal handler: {prior_handler}. ")
            else:
                time.sleep(5)

            try:
                if prior_handler is not None and callable(prior_handler):
                    rprint("WARNING: Not on rank zero!  Returning to prior handler")
                    prior_handler(sig, frame, exit_on_requeue=False)
                    rprint(f"WARNING: Not on rank zero!  Returned from prior handler")
            except:
                rprint(f"WARNING: Not on rank zero!  Failed to return to prior handler")

            if self.config.slurm:
                time.sleep(5)  # Should be enough time for SLURM to send a SIGTERM to all ranks. If not, we resort to manual requeueing.
                rprint(f"WARNING: Not on rank zero!  Failed to requeue using prior handler, requeuing job ourselves...  {job_str}")
                subprocess.check_call(["scontrol", "requeue", job_str])
                rprint(f"WARNING: Not on rank zero! Requeued job: {job_str}")

        if self.config.slurm:
            if torch.distributed.is_initialized():
                rprint(f"Destroying process group...")
                torch.distributed.destroy_process_group()
            return sys.exit(0)
        else:
            rprint(f"Not on SLURM, not exiting")

    prior_sigterm_handler = signal.getsignal(signal.SIGTERM)
    prior_sigusr1_handler = signal.getsignal(signal.SIGUSR1)
    prior_sigusr2_handler = signal.getsignal(signal.SIGUSR2)

    rprint(f"Found Prior SIGTERM handler: {prior_sigterm_handler}, type: {type(prior_sigterm_handler)}")
    rprint(f"Found Prior SIGUSR1 handler: {prior_sigusr1_handler}, type: {type(prior_sigusr1_handler)}")
    rprint(f"Found Prior SIGUSR2 handler: {prior_sigusr2_handler}, type: {type(prior_sigusr2_handler)}")

    signal.signal(signal.SIGTERM, functools.partial(_handler, prior_handler=prior_sigterm_handler))
    signal.signal(signal.SIGUSR2, functools.partial(_handler, prior_handler=prior_sigusr2_handler))
    signal.signal(signal.SIGUSR1, functools.partial(_handler, prior_handler=prior_sigusr1_handler))

def on_train_start(self):
    gprint(f"Starting train at step: {self.global_step}")

    if is_main_process() and getattr(self.config.trainer, "compile", None) is None and getattr(self.config.trainer, "watch_gradients", True):
        wandb.watch(
            self.backbone,
            log=("all" if getattr(self.config.trainer, "watch_all", False) else "gradients"),
            log_freq=getattr(self.config.trainer, "watch_gradients_freq", 500),
        )

    if getattr(self.config.trainer, "attach_oom_observer_train", False):
        from torchtnt.utils.oom import attach_oom_observer
        attach_oom_observer(output_dir=str(self.config.output_dir), trace_max_entries=500000)
        gprint(f"Attached OOM observer to {self.config.output_dir}")

    if self.config.trainer.nvtx_profile and self.is_compiled is False:
        torch.cuda.cudart().cudaProfilerStart()

    # TODO: Make sure we don't need the code below with the new accelerate code.
    return

def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema is not None:
        self.ema.update(self.get_params())

def init_dataloader(self, train_dataloader, val_dataloader):
    rprint("Creating train_dataset + self.train_dataloader")
    self.train_dataloader = train_dataloader
    self.validation_dataloader = val_dataloader
    if not self.config.data.iterable and not self.config.data.webdataset_indexed: assert len(self.validation_dataloader) > 0

def init_optimizer_lr_scheduler(self):
    [optimizer], [scheduler_dict] = self.configure_optimizers()
    self.optimizer = optimizer
    self.lr_scheduler = scheduler_dict["scheduler"]

def set_accelerator(self, accelerator, ckpt_path=None):
    if ckpt_path is not None:
        rprint(f"Set accelerator with ckpt path {ckpt_path}")

    self.accelerator = accelerator
    self.device = accelerator.device
    self.dtype = getattr(torch, self.config.trainer.dtype.split(".")[-1])

    def _load(obj, path, update_fn=None, key="model"):
        _ckpt_path = Path(path)
        if _ckpt_path.is_dir() and (_ckpt_path / "model.safetensors").exists():
            _ckpt_path = _ckpt_path / "model.safetensors"
            path = str(_ckpt_path)
            
        print(f"Loading from {_ckpt_path}, {_ckpt_path.suffix}, {_ckpt_path.is_dir()}")
        if _ckpt_path.suffix == ".safetensors":
            state_dict = load_file(path)
        elif _ckpt_path.is_dir():
            if getattr(self.config.trainer, 'dynamic_convert_to_normal_state_dict', False):
                gprint(f"Converting distributed checkpoint to normal state dict")
                from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
                import hashlib
                ckpt_hash = hashlib.md5(str(path).encode()).hexdigest()[:8] + "_" + Path(path).stem
                new_path = str(Path("/dev/shm") / os.getenv("USER", "aswerdlo") / f"tmp_ckpt_{ckpt_hash}.pth")
                dcp_to_torch_save(path, new_path)
                gprint(f"Converted distributed checkpoint to normal state dict at {new_path}")
                state_dict = torch.load(new_path)
                gprint(f"Loaded state dict from {path}")
            else:
                gprint(f"Loading from distributed checkpoint directory {path}")
                import torch.distributed.checkpoint as dcp
                state_dict = {
                    key: obj.state_dict(),
                }
                if getattr(self.config.trainer, 'ignore_chameleon_embed', False):
                    for k in list(state_dict[key].keys()):
                        if "embed_tokens" in k:
                            state_dict[key].pop(k)
                            gprint(f"Ignoring {k}")
                dcp.load(
                    state_dict=state_dict,
                    checkpoint_id=path,
                )
                gprint(f"Loaded state dict from {path}")
                # obj.load_state_dict(state_dict[key])
        else:
            state_dict = torch.load(path)
        
        if 'model' in state_dict and len(state_dict) < 10:
            state_dict = state_dict['model']

        state_dict = {k.replace("_orig_module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if self.config.backbone == 'llama' and "lm_head.weight" in state_dict and "model.embed_tokens.weight" not in state_dict:
            # LLaMa ties weights
            state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"].clone()

        if update_fn is not None:
            state_dict = update_fn(state_dict)
        elif getattr(self.config.trainer, 'use_orig_unidisc_dit', False):
            # loading from the original .ckpt files from unidisc repo
            state_dict = state_dict['state_dict']
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            
        try:
            kwargs = {}
            kwargs['strict'] = self.config.trainer.disable_strict_load
            if '.bin' in str(path):
                kwargs = {}
            obj.load_state_dict(state_dict, **kwargs)
        except Exception as e:
            rprint(f"Failed to load state dict: {e}")
            rprint(f"State dict keys: {state_dict.keys()}")
            rprint(f"Model state dict keys: {obj.state_dict().keys()}")
            raise e
        
    if self.config.mode != 'eval':
        self.init_optimizer_lr_scheduler()

    if getattr(self.config.trainer, "bypass_load_from_state_dicts_if_resuming", False) and ckpt_path is not None:
        rprint(f"Skipping load from state dicts since we are resuming from: {ckpt_path}")
    else:
        if self.config.trainer.load_from_state_dict is not None:
            rprint(f"Loading model state dict from {self.config.trainer.load_from_state_dict}")
            _load(self.backbone, self.config.trainer.load_from_state_dict)
            rprint(f"Loaded model state dict from {self.config.trainer.load_from_state_dict}")

        if getattr(self.config.trainer, "load_from_optimizer_state_dict", None) is not None:
            # TODO: Optimizer.bin from accelerate is the wrong format here. Look into this. The keys/are different and need to be mapped.
            def update_param_group(state_dict):
                rprint(f"len(self.optimizer.param_groups): {len(self.optimizer.param_groups[0]['params'])}, len(state_dict['param_groups']): {len(state_dict['param_groups'][0]['params'])}")
                rprint(f"self.optimizer.param_groups: {self.optimizer.param_groups[0]['params']}")
                rprint(f"state_dict['param_groups']: {state_dict['param_groups'][0]['params']}")
                state_dict["param_groups"] = self.optimizer.param_groups
                return state_dict

            _load(self.optimizer, self.config.trainer.load_from_optimizer_state_dict, update_fn=update_param_group, key="optim")
            rprint(f"Loaded optimizer state dict from {self.config.trainer.load_from_optimizer_state_dict}")

    if self.config.mode == 'eval':
        rprint(f"Moving model to {self.device}")

    self.backbone.to(self.device)
    if getattr(self.config.trainer, 'force_bf16_eval', False) and self.config.mode == 'eval':
        self.backbone.to(torch.bfloat16)

    # Model needs to be wrapped before optimizer is created for fsdp
    if self.config.trainer.xla_spmd and is_xla_available:
        self.backbone = wrap_xla_fsdp(self.config, self.backbone)

    self.backbone, self.ema = self.accelerator.prepare(self.backbone, self.ema)

    if self.config.trainer.compile and not is_xla_available:
        rprint("Compiling entire model...")
        self.backbone = compile_model(self.config, self.backbone)

    if getattr(self.config.trainer, 'mup_coord_plot', False):
        self.get_coord_plot()

    if self.config.mode == 'eval':
        return

    if not self.config.data.iterable and not self.config.data.webdataset_indexed and self.train_dataloader is not None and self.config.data.wrap_dataloaders:
        rprint(f"Before prepare: Train len: {len(self.train_dataloader)}, Validation len: {len(self.validation_dataloader)}")

    if getattr(self.config.eval, 'test_eval_speed', False):
        self.optimizer, self.lr_scheduler = None, None
    else:
        if getattr(self.config.trainer, 'force_disable_wrap_optimizer', False) is False and self.config.mode != 'eval':
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.optimizer, self.lr_scheduler
            )
        elif self.config.mode != 'eval':
            rprint("WARNING: Not wrapping optimizer with accelerator.prepare()")

    if self.config.data.webdataset_iterable is False and self.config.data.wrap_dataloaders:
        self.train_dataloader, self.validation_dataloader = self.accelerator.prepare(self.train_dataloader, self.validation_dataloader)
    else:
        rprint("WARNING: Not wrapping dataloaders with accelerator.prepare()")

    if is_xla_available and self.config.trainer.fsdp:
        self.train_dataloader = tpu_spmd_dataloader(self.train_dataloader, self.device)
        self.validation_dataloader = tpu_spmd_dataloader(self.validation_dataloader, self.device)

    if not self.config.data.iterable and not self.config.data.webdataset_indexed and self.train_dataloader is not None:
        rprint(f"After prepare: Train len: {len(self.train_dataloader)}, Validation len: {len(self.validation_dataloader)}")

    if (self.config.trainer.use_spmd_distributed_checkpointing or self.config.trainer.use_simple_spmd_distributed_checkpointing) and is_xla_available:
        gprint("Initializing distributed process group")
        import torch.distributed as dist
        import torch_xla.distributed.xla_backend
        import torch_xla.runtime as xr
        dist.init_process_group('gloo', init_method='xla://')
        gprint("Distributed process group initialized, before creating checkpoint manager")

    if (self.config.trainer.use_spmd_distributed_checkpointing and self.config.trainer.disable_all_checkpointing is False) and is_xla_available:
        gprint("Initializing checkpoint manager")
        from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer
        self.chkpt_mgr = CheckpointManager(self.config.checkpointing.save_dir, self.config.trainer.ckpt_steps)
        gprint(f"Checkpoint manager created")

    if getattr(self.config.trainer, "force_from_ckpt", None) is not None:
        ckpt_path = getattr(self.config.trainer, "force_from_ckpt")
        if ckpt_path == "":
            ckpt_path = None

    if ckpt_path is not None and Path(ckpt_path).exists():
        rprint(f"Loading checkpoint {ckpt_path}")
        if self.config.trainer.use_spmd_distributed_checkpointing and self.config.trainer.disable_all_checkpointing is False:
            gprint("Loading checkpoint for XLA")
            from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer
            tracked_steps = self.chkpt_mgr.all_steps()
            if tracked_steps:
                rprint(f"Found tracked steps: {tracked_steps}")
                best_step = max(tracked_steps) # Choose the highest step
                prime_optimizer(self.optimizer) # Before restoring the checkpoint, the optimizer state must be primed to allow state to be loaded into it.
                state_dict = {'model': self.accelerator.unwrap_model(self.backbone).state_dict(), 'optim': self.optimizer.state_dict()}
                self.chkpt_mgr.restore(best_step, state_dict)
                self.backbone.load_state_dict(state_dict['model'])
                self.optimizer.load_state_dict(state_dict['optim'])
        else:
            import os
            folder_contents = os.listdir(ckpt_path)
            gprint(f"Contents of the folder {ckpt_path}: {folder_contents}")
            self.accelerator.load_state(ckpt_path, strict=self.config.trainer.disable_strict_load is False)

    elif ckpt_path is not None:
        rprint(f"WARNING: Checkpoint {ckpt_path} does not exist")

    if getattr(self.config.trainer, "reset_lr_scheduler_step", False):
        with open_dict(self.config):
            with read_write(self.config):
                rprint(f"Resetting lr scheduler")
                if getattr(self.config.trainer, "global_num_warmup_steps", None) is not None:
                    self.config.lr_scheduler.num_warmup_steps = self.config.trainer.global_num_warmup_steps
                    rprint(f"Set num_warmup_steps to {self.config.lr_scheduler.num_warmup_steps}")

                if getattr(self.config.trainer, "global_num_training_steps", None) is not None:
                    self.config.lr_scheduler.num_training_steps = self.config.trainer.global_num_training_steps
                    rprint(f"Set num_training_steps to {self.config.lr_scheduler.num_training_steps}")
                if not self.config.trainer.disable_adjust_num_warmup_steps:
                    _world_size = 1 if (is_xla_available and self.config.trainer.xla_spmd) else self.world_size
                    rprint(f"Warmup steps was {self.config.lr_scheduler.num_warmup_steps}")
                    self.config.lr_scheduler.num_warmup_steps = self.config.lr_scheduler.num_warmup_steps * _world_size
                    rprint(f"Warmup steps is now {self.config.lr_scheduler.num_warmup_steps}, world size is {_world_size}")

                    if hasattr(self.config.lr_scheduler, "num_training_steps"):
                        rprint(f"num_training_steps was: {self.config.lr_scheduler.num_training_steps}. Applying to num_training_steps")
                        self.config.lr_scheduler.num_training_steps = self.config.trainer.global_num_training_steps * _world_size

        rprint(f"Set num_warmup_steps to {self.config.lr_scheduler.num_warmup_steps}")
        if getattr(self.config.trainer, "global_num_training_steps", None) is not None:
            rprint(f"Set num_training_steps to {self.config.lr_scheduler.num_training_steps}")
        self.lr_scheduler.scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=self.lr_scheduler.scheduler.optimizer)
        rprint(self.lr_scheduler.scheduler.__dict__)
        rprint(self.lr_scheduler.scheduler.state_dict())
        rprint("WARNING!!! Resetting lr scheduler")
    elif getattr(self.config.trainer, "force_reset_optimizer_lr_scheduler", False):
        self.init_optimizer_lr_scheduler()
        self.lr_scheduler, self.optimizer = self.accelerator.prepare(self.lr_scheduler, self.optimizer)

def set_callbacks(self):
    from torchtnt.framework._callback_handler import CallbackHandler

    from unidisc.utils.throughput_monitor import ThroughputMonitor

    precomputed_flops_per_sample = {}
    _flops_per_sample = precomputed_flops_per_sample.get(self.config.model.name, 0)
    if _flops_per_sample == 0 or self.config.backbone != 'dit':
        # Assume approx 6ND for decoder transformer model
        _flops_per_sample = 6 * self.config.model.length * self.non_embedding_params

    if self.config.trainer.xla_spmd and is_xla_available:
        _flops_per_sample /= self.world_size

    callbacks = []
    callbacks.append(
        ThroughputMonitor(
            batch_size_fn=None,
            length_fn=None,
            log_every_n_steps=50,
            window_size=2,
            separator="_",
            world_size=1 if self.config.trainer.xla_spmd else self.world_size,
            device=self.device,
            dtype=self.dtype,
            flops_per_sample=_flops_per_sample
        )
    )

    self.cb_handler = CallbackHandler(callbacks)

@try_except(write_error_to_file=True)
def checkpoint(self, state: TrainingState = None):
    if is_torch_xla_available():
        gprint("Saving checkpoint on XLA...")

    self.on_train_resume() # In case we start checkpointing in the middle of validation

    checkpoint_all_ranks = self.config.trainer.checkpoint_all_ranks
    if (not is_main_process()) and checkpoint_all_ranks is False:
        return
    
    if self.current_run_global_step < 200 and self.config.trainer.skip_early_checkpointing:
        rprint("Skipping checkpointing for the first 200 steps...")
        return
    
    if self.config.trainer.disable_all_checkpointing:
        rprint("Disabled all checkpointing...")
        return

    start_time = time.time()
    if self.config.trainer.use_simple_spmd_distributed_checkpointing and is_xla_available:
        import torch.distributed.checkpoint as dist_cp
        import torch_xla.experimental.distributed_checkpoint as xc
        gprint("Saving checkpoint...0")
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        gprint("Saving checkpoint...1")
        xm.wait_device_ops()
        gprint("Saving checkpoint...2")
        CHECKPOINT_DIR = Path(self.config.checkpointing.save_dir) / f"checkpoint_{self.global_step}"
        gprint("Saving checkpoint...4")

        if is_main_process():
            gprint(f"Clearing old checkpoints")
            handle_checkpointing_dirs(self.config, prefix="checkpoint")
            gprint(f"Finished clearing old checkpoints")

        state_dict = {
            "model": self.backbone.state_dict(),
        }
        if not self.config.trainer.ckpt_model_only:
            gprint("Saving optimizer state dict")
            state_dict["optim"] = self.optimizer.state_dict()

        gprint(f"Saving checkpoint...5 to {CHECKPOINT_DIR}")
        dist_cp.save(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=xc.SPMDSavePlanner(),
        )
        if is_main_process():
            from main import save_config_to_ckpt
            save_config_to_ckpt(self.config, CHECKPOINT_DIR, self)
        gprint("Saving checkpoint...6")
    elif self.config.checkpointing.use_automatic_naming:
        rprint("Saving checkpoint...")
        self.accelerator.save_state()
        rprint("Saved checkpoint...")
    else:
        rprint(f"Saving checkpoint...")
        prefix = "checkpoint"
        Path(self.config.checkpointing.save_dir).mkdir(exist_ok=True, parents=True)

        if is_main_process():
            handle_checkpointing_dirs(self.config, prefix="checkpoint")

        save_path = Path(self.config.checkpointing.save_dir) / f"{prefix}_{self.global_step}"
        save_path.mkdir(exist_ok=True, parents=True)

        if checkpoint_all_ranks:
            barrier()

        if self.config.trainer.ckpt_model_only:
            from safetensors.torch import save_file, save_model
            try:
                self.accelerator.save_model(self.backbone, save_path)
            except Exception as e:
                rprint(f"Failed to save model with 'save_file': {e}")
                if getattr(self.config.trainer, 'finetuning_mode', False):
                    rprint("Fallback to 'save_model' instead")
                    if is_main_process():
                        save_model(self.backbone, save_path / "model.safetensors")
        else:
            try:
                self.accelerator.save_state(save_path)
            except Exception as e:
                from traceback import print_exc
                print_exc()
                gprint(f"Failed to save state: {e}, saving model instead")
                self.accelerator.save_model(self.backbone, save_path)
                gprint("Saved model instead")

        if checkpoint_all_ranks:
            barrier()

        rprint(f"Saved checkpoint to: {save_path}")
        with try_except(write_error_to_file=True, clear_cuda_cache=True):
            self.print_hashes()
        
    rprint(f"Checkpointing took: {time.time() - start_time} seconds")

def print_hashes(self):
    if self.config.trainer.fsdp:
        rprint('Skipping module hash for FSDP')
        return
    
    rprint(f"Module hash: {module_hash(self.backbone)}")
    if self.ema is not None:
        if self.config.trainer.use_custom_ema:
            rprint(f"EMA hash: {module_hash(self.ema)}")
        else:
            rprint(f"EMA hash: {parameter_hash(self.ema.state_dict()['shadow_params'])}")

@try_except(write_error_to_file=True)
def on_train_step_end(self, state: TrainingState):
    self.cb_handler.on_train_step_end(state=state, unit=self)
    del state.batch
    tr = self.config.trainer
    if check_every_n_steps(
        state, tr.val_check_interval, run_first=tr.eval_on_start, all_processes=True, decay_steps=tr.eval_decay_steps
    ) or check_every_n_epochs(state, tr.eval_epochs, all_processes=True):
        rprint(f"Starting validation at {state.global_step}...")
        with show_memory_usage():
            with try_except(write_error_to_file=True, clear_cuda_cache=True):
                with nullcontext() if is_xla_available else (torch.no_grad() if getattr(self.config.trainer, "force_disable_inference_mode", False) else torch.inference_mode()):
                    self.validate(state)
                    self.on_validation_epoch_cleanup()
                    self.num_evals += 1
                    self.on_train_resume()
        dprint("All processes finished validation")

    xla_spmd = self.config.trainer.use_spmd_distributed_checkpointing
    if xla_spmd and self.config.trainer.disable_all_checkpointing is False and self.global_step > 10:
        # Call every step, but only runs after n steps internally
        gprint("Might save async checkpoint...")
        if getattr(self.config.checkpointing, "save_optimizer_state", True):
            state_dict = {'model': self.backbone.state_dict(), 'optim': self.optimizer.state_dict()}
        else:
            gprint("[WARNING] Not saving optimizer state")
            state_dict = {'model': self.backbone.state_dict()}
        if self.chkpt_mgr.save_async(self.global_step, state_dict):
            gprint(f'Checkpoint taken at step {self.global_step}')

    current_time = time.time()
    if not hasattr(self, "last_checkpoint_time"):
        self.last_checkpoint_time = current_time

    checkpoint_due_to_time = (current_time - self.last_checkpoint_time) >= (tr.ckpt_every_n_minutes * 60)
    checkpoint_due_to_step = check_every_n_steps(state, tr.ckpt_steps, run_first=False, all_processes=True)
    
    if is_torch_cuda_available() and tr.ckpt_every_n_minutes > 0:
        should_ckpt_all_ranks = gather_object([checkpoint_due_to_time or checkpoint_due_to_step])
    else:
        should_ckpt_all_ranks = [checkpoint_due_to_step]

    if should_ckpt_all_ranks[0] and not xla_spmd: # To avoid timing inconsistencies, we take the value from the main process
        rprint(f"Saving checkpoint at {self.global_step}...due to {'time' if checkpoint_due_to_time else 'step'}. Ranks thought: {should_ckpt_all_ranks}")
        self.last_checkpoint_time = current_time
        self.checkpoint(state)
        rprint(f"Checkpoint saved at {self.global_step}...")

def after_backward(self, state):
    freq = getattr(self.config.trainer, "log_grad_norm_every_n_steps", 200 if self.is_compiled else 50)
    if not is_xla_available and self.config.trainer.log_grad_norm and check_every_n_steps(state, freq, run_first=True, all_processes=False):
        norms, total_norm = grad_norm(self.backbone, norm_type=2, group_separator="")
        grad_norm_dict = {f"grad_norms/{k}": v for k, v in norms.items()}
        if 'text-diffusion' in self.config.wandb.project:
            grad_norm_dict = {k.replace("module.", ""): v for k, v in grad_norm_dict.items()}
        log({**grad_norm_dict, "trainer/total_grad_norm": total_norm, "trainer/global_step": self.global_step})

from model_utils import Loss
def shortcut_return(self, logprobs, output_tokens, attention_mask, prefix): # For comparing to unidisc only
    loss = -logprobs.gather( -1, output_tokens[:, :, None])[:, :, 0]
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    losses = Loss(
        loss=token_nll,
        img_loss=0,
        txt_loss=0,
        nlls=nlls,
        txt_nlls=0,
        img_nlls=0,
        token_mask=attention_mask,
        modality_mask=None,
        extra_losses=None,
    )

    if getattr(self.config.trainer, "disable_torchmetrics", False):
        raise NotImplementedError("Torchmetrics disabled")

    elif prefix == "train":
        return losses
    elif prefix == "val":
        self.valid_metrics.update(losses.nlls, losses.token_mask)
    elif prefix == "test":
        self.test_metrics.update(losses.nlls, losses.token_mask)
        metrics = self.test_metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
    else:
        raise ValueError(f"Invalid prefix: {prefix}")

def unwrap_model(self, model):
    from diffusers.utils.torch_utils import is_compiled_module
    model = self.accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model