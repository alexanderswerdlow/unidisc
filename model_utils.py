import math
import os
import random
import typing
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Dict, List, Optional, Tuple, Union

import einops
import hydra
import hydra.utils
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchmetrics
import transformers
from image_utils import Im
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.auto import tqdm

import models
import wandb
from decoupled_utils import (Profiler, barrier, dprint, get_rank,
                             get_slurm_job_id, get_world_size, gprint,
                             is_local_main_process, is_main_process,
                             is_torch_cuda_available, is_torch_xla_available,
                             module_hash, mprint, parameter_hash, print_memory,
                             rank_zero_fn, rprint, save_memory_profile,
                             show_memory_usage, try_except, use_dist)

is_xla_available = is_torch_xla_available()
if is_xla_available:
    from unidisc.utils.standalone_metrics import MeanMetric, MetricCollection
else:
    from torchmetrics import MetricCollection
    from torchmetrics.aggregation import MeanMetric

LOG2 = math.log(2)

@try_except(write_error_to_file=True)
def log(*arg, **kwargs):
    for key, value in arg[0].items():
        if isinstance(value, torch.Tensor):
            arg[0][key] = value.detach().cpu().float()
    
    if is_main_process():
        wandb.log(*arg, **kwargs)

def replace_nan_dict(x):
    return {k: v.nan_to_num(0) for k, v in x.items()}

def ddprint(*args, **kwargs):
    mprint(*args, **kwargs)

def empty_device_cache():
    if is_torch_cuda_available():
        torch.cuda.empty_cache()
    else:
        dprint("Not using cuda, skipping cache clear")

def update_logs(_logs, _extra_logs):
    _logs.update(_extra_logs())
    for k, v in _logs.items():
        if isinstance(v, torch.Tensor):
            _logs[k] = v.detach().cpu().item()
            gprint(f"Converting {k} to item: {v}")

    log(_logs)

def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        if p_name not in param_dict_src:
            print(f"Parameter {p_name} not found in src: {param_dict_src}")
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)

def identity(x):
    return x

def remap_image_torch(image):
    image_torch = image * 255
    image_torch = torch.clip(image_torch, 0, 255).to(torch.uint8)
    return image_torch

def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def wrapped_batch_decode(tokenizer, tokens, disable_mask_after_eos=False, **kwargs):
    tokens = tokens.clone()
    if (tokenizer.bos_token_id != tokenizer.eos_token_id) and not disable_mask_after_eos:
        after_first_five = torch.cumsum(tokens == tokenizer.eos_token_id, dim=1).bool()
        tokens[after_first_five.cumsum(dim=1) > 1] = tokenizer.pad_token_id
    return tokenizer.batch_decode(tokens, **kwargs)

def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    img_loss: torch.FloatTensor = None
    txt_loss: torch.FloatTensor = None
    nlls: torch.FloatTensor = None
    token_mask: torch.FloatTensor = None
    txt_nlls: torch.FloatTensor = None
    img_nlls: torch.FloatTensor = None
    extra_losses: dict = None
    modality_mask: torch.FloatTensor = None


class NLL(MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)
    
class Entropy(NLL):
    def compute(self) -> Tensor:
        """Computes the Entropy.

        Returns:
         Entropy
        """
        return self.mean_value / self.weight
    
class MauveScore(NLL):
    def compute(self) -> Tensor:
        """Computes the Mauve Score.

        Returns:
         Mauve Score
        """
        return self.mean_value / self.weight
    
class CIDErScore(NLL):
    def compute(self) -> Tensor:
        """Computes the CIDEr Score.

        Returns:
         CIDEr Score
        """
        return self.mean_value / self.weight

class Accuracy(NLL):
    def compute(self) -> Tensor:
        """Computes the Accuracy.

        Returns:
         Accuracy
        """
        return self.mean_value / self.weight

def get_coord_plot(self):
    from mup.coord_check import get_coord_data, plot_coord_data
    def gen(w):
        def f():
            from copy import deepcopy

            from omegaconf import read_write

            import models as _models
            _conf = deepcopy(self.config)
            with read_write(_conf):
                _conf.model.hidden_size = _conf.model.n_heads * w
            
            _backbone = _models.dit.DIT(
                _conf, vocab_size=self.vocab_size, mask_index=self.mask_index, text_vocab_size=self.text_vocab_size, dtype=self.dtype
            )
            self.get_base_shapes_for_mup(_backbone)
            return _backbone
        return f

    optimizer = 'adamw'
    widths = np.array([2**i for i in range(2, 6)])
    models = {int(w) * self.config.model.n_heads: gen(int(w)) for w in widths}

    fake_dataloader = []
    self.validation_dataloader.num_workers = 0
    nsteps = 30
    for i, dataloader_batch in enumerate(self.validation_dataloader):
        fake_batch = self.update_batch(dataloader_batch)
        fake_batch['x0'] = fake_batch["input_ids"]
        t = self._sample_t(fake_batch['x0'].shape[0], fake_batch['x0'].device)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = self.q_xt(fake_batch['x0'], move_chance)
        fake_batch['xt'] = xt

        fake_dataloader.append(fake_batch)
        if i >= nsteps:
            break

    def loss_fn(_batch, _logits):
        attention_mask = _batch['attention_mask']
        model_output = self._subs_parameterization(logits=_logits, xt=_batch['xt'])
        log_p_theta = torch.gather(input=model_output, dim=-1, index=_batch['x0'][:, :, None]).squeeze(-1)
        std_weighting = (dsigma / torch.expm1(sigma))[:, None]
        loss = -log_p_theta * std_weighting
        loss = (loss * attention_mask).sum() / attention_mask.sum()
        return loss

    mup = True
    lr = 1e-2
    prm = 'μP' if mup else 'SP'
    nseeds = 2
    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
        df = get_coord_data(
            models,
            fake_dataloader,
            lr=lr,
            optimizer=optimizer,
            nsteps=nsteps,
            nseeds=nseeds,
            dict_in_out=True,
            lossfn=loss_fn,
            mup=mup,
        )

    output_path = Path(__file__).parent / 'output' / f'{prm.lower()}_trsfmr_{optimizer}_coord.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_coord_data(
        df,
        legend='brief',
        save_to=str(output_path.resolve()),
        suptitle=f'{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}',
        face_color='xkcd:light grey' if not mup else None,
        loglog=True
    )
    rprint(f"Saved coord plot to {output_path.resolve()}")

    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    rprint(f"DataFrame saved as CSV to {csv_path.resolve()}")
    
    result = df[df['t'] == 1].nsmallest(100, 'l1').sort_values('l1', ascending=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(result[['module', 'width', 'l1']])
    exit()

def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
        log_score: float torch.Tensor with shape (batch_size,
            diffusion_model_input_length, vocab_size),
            log score, output of the denoising network.
        xt: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        x0: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
        loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(log_score[masked_indices], -1, words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
        pos_term = score[:, :-1].sum(dim=-1)
    else:
        pos_term = score[:, : self.mask_index].sum(dim=-1) + score[:, self.mask_index + 1 :].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(*xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

@torch.no_grad
def sample_subs_guidance(self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
        p_x0_cache = None
        x = self._sample_prior(n_samples, self.config.model.length).to(self.device)
        if target is not None:
            x[:, :-stride_length] = target
        for i in range(num_steps + 1):
            p_x0_cache, x_next, nfe_cnt = self._ddpm_caching_update(x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
            if not torch.allclose(x_next, x) or self.time_conditioning:
                p_x0_cache = None
                sampling_steps += 1
            x = x_next
        x = self.forward(x, 0 * ones).argmax(dim=-1)
        intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
        target = x[:, stride_length:]

    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((np.concatenate(intermediate_tokens, axis=1)[:, 1:] == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
        intermediate_text_samples.append(self.tokenizer.batch_decode(np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples, sequence_lengths)

def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
        self.ema.store(self.get_params())
        self.ema.copy_to(self.get_params())
    self.backbone.eval()
    (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
        n_samples=self.config.loader.eval_batch_size, stride_length=stride_length, num_strides=num_strides, dt=dt
    )
    if self.ema:
        self.ema.restore(self.get_params())
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths

def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype, device=self.device)
    assert self.config.noise.type == "loglinear"
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return -torch.gather(input=model_output_t0, dim=-1, index=x0[:, :, None]).squeeze(-1)

def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema is not None:
        self.ema.store(self.get_params())
        self.ema.copy_to(self.get_params())
    self.backbone.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema is not None:
        self.ema.restore(self.get_params())
    self.backbone.train()
    return samples

def get_score(self, x, sigma, **kwargs):
    model_output = self.forward(x, sigma, **kwargs)
    if self.parameterization == "subs":
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = -(log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (1 - masked_indices)
    return model_output.exp()

def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    nfe_cnt = 0
    score = self.get_score(x, curr_sigma)
    nfe_cnt += 1
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs), nfe_cnt

def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[..., None]
    return edge

@torch.no_grad()
def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.

    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if "llama2" in self.gen_ppl_eval_model_name_or_path:
        tokenizer_kwargs = {
            "text_samples": text_samples,
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "return_attention_mask": True,
            "truncation": True,
            "padding": True,
            "max_length": max_length,
        }
        eval_context_size = 4096
    else:
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "return_attention_mask": True,
            "truncation": True,
            "padding": True,
            "max_length": max_length,
        }
        eval_context_size = 1024

    if getattr(self.config.eval, "force_eval_context_size_match_model", False):
        eval_context_size = self.config.model.txt_length

    samples = self.eval_model_tokenizer(text_samples, **tokenizer_kwargs)
    attn_mask = samples["attention_mask"]
    samples = samples["input_ids"]
    if "llama2" not in self.gen_ppl_eval_model_name_or_path:
        attn_mask = attn_mask.to(self.device)
        samples = samples.to(self.device)
    return samples, attn_mask, eval_context_size

    
@try_except(write_error_to_file=True)
@torch.no_grad()
def compute_cider(self, text_samples, gt_text_samples):
    """Compute the CIDEr score for the generated text.
    Args:
        text_samples: List of sentences generated by the model.
        gt_text_samples: List of ground truth sentences.
    Returns:
        CIDEr score for the generated text.
    """
    for text_sample, gt_text_sample in zip(text_samples, gt_text_samples):
        self.cider_score_metric += (text_sample, gt_text_sample)
    score = self.cider_score_metric.compute_cider() # list of np.float64
    avg_score = sum(score) / len(score)
    self.cider_score.update(avg_score.item()) # weight=len(text_samples))
    

def get_anole_data(model, processor, prompt, image, device):
    inputs = processor(prompt, [image], padding=True, return_tensors="pt").to(device=device, dtype=dtype)
    image_tokens = model.model.get_image_tokens(inputs["pixel_values"])
    special_image_mask = inputs["input_ids"] == model.model.vocabulary_mapping.image_token_id
    image_tokens = image_tokens.to(inputs["input_ids"].device, inputs["input_ids"].dtype)
    inputs["input_ids"] = inputs["input_ids"].masked_scatter(special_image_mask, image_tokens)
    inputs.pop("pixel_values")
    inputs['input_ids'] = torch.load('save.pth').to(device)
    return inputs

@try_except(write_error_to_file=True)
@torch.inference_mode()
def compute_generative_perplexity(self, text_samples: typing.List[str], retokenize: bool = True, max_length: typing.Optional[int] = None, gt: bool = False, return_raw_score: bool = False) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
        retokenize: Whether to retokenize using eval model's tokenizer
        max_length: Maximum sequence length for tokenization
        gt: Whether these are ground truth samples
        return_raw_score: Whether to return raw NLL scores instead of updating metrics

    Returns:
        If return_raw_score is True, returns tensor of NLL scores.
        Otherwise updates internal perplexity metrics.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not getattr(self.config.eval, 'enable_gen_pplx_cleanup', True):
        eval_model = self.gen_pplx_eval_model
    elif getattr(self.config.eval, 'gen_ppl_use_chameleon', False):
        from transformers import (ChameleonForConditionalGeneration,
                                  ChameleonProcessor)
        model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", torch_dtype=torch.bfloat16).to("cuda")
        processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
        image = Im(Im("https://cdn.outsideonline.com/wp-content/uploads/2023/03/Funny_Dog_H.jpg").np[50:-150, 550:-900, :]).resize(256, 256).pil
        prompt = "A picture of a cat.<image>"
        device = "cuda:0"
        inputs = get_anole_data(model, processor, prompt, image, self.dtype, device)
        output = model(input_ids=inputs['input_ids'].to(device))
        attention_mask = torch.ones_like(inputs["input_ids"])
        logits = output.logits
        logits = logits.transpose(-1, -2)
        sample_chunk = inputs["input_ids"]
        nlls = F.cross_entropy(logits[..., :-1].to(device), sample_chunk[..., 1:].to(device), reduction="none")
        nlls = nlls * attention_mask[..., 1:].to(nlls.dtype)
        nlls = nlls.sum(-1) / attention_mask[..., 1:].sum(-1)
        print(torch.exp(nlls))
    else:
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
        max_length = self.config.model.txt_length

    if "llama2" not in self.gen_ppl_eval_model_name_or_path:
        eval_model = eval_model.to(self.device)
    
    # Re-tokenize using eval model's tokenizer
    if retokenize:
        (samples, attn_mask, eval_context_size) = self.eval_retokenize(text_samples, max_length=max_length)
    else:
        samples = text_samples
        attn_mask = torch.ones(samples.shape).to(self.device)
        eval_context_size = samples.shape[-1]

    batch_size = min(self.config.eval.perplexity_batch_size, samples.shape[0])
    num_batches = (samples.shape[0] + batch_size - 1) // batch_size
    all_nlls = []
    all_valid_mask = []
    for i in range(num_batches):
        batch_samples = samples[i * batch_size : (i + 1) * batch_size]
        batch_attn_mask = attn_mask[i * batch_size : (i + 1) * batch_size]

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            logits = eval_model(batch_samples, attention_mask=batch_attn_mask)[0]
            
        logits = logits.transpose(-1, -2)
        nlls = F.cross_entropy(logits[..., :-1], batch_samples[..., 1:], reduction="none")
        
        # Only consider tokens up to first EOS or padding
        first_eos = (batch_samples == self.eval_model_tokenizer.eos_token_id).cumsum(-1) <= 1
        token_mask = batch_attn_mask[..., 1:] > 0
        valid_mask = first_eos[..., 1:] * token_mask

        if not return_raw_score:
            if gt:
                self.gt_gen_ppl_metric.update(nlls, valid_mask)
            else:
                self.gen_ppl_metric.update(nlls, valid_mask)
        else:
            all_nlls.append(nlls)
            all_valid_mask.append(valid_mask)

    if getattr(self.config.eval, 'enable_gen_pplx_cleanup', True):
        eval_model.to(torch.device('cpu'))
        del eval_model

    if return_raw_score:
        all_nlls = torch.cat(all_nlls)
        all_valid_mask = torch.cat(all_valid_mask)
        # Compute mean NLL per sequence, ignoring padding/post-EOS tokens
        nll = (all_nlls * all_valid_mask).sum(-1) / all_valid_mask.sum(-1)
        return nll

def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
        t = t[:, None]
        assert t.ndim == 2
        t = t.clamp(0.0, 1.0 - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()

    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0

    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (term_2_log_nr - term_2_log_dr)

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

def _d3pm_parameterization(self, logits):
    if self.subs_masking:
        logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logits

def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None], torch.zeros_like(logits[..., :1]))
    return logits

def get_base_shapes_for_mup(self, _model):
    from copy import deepcopy

    from mup import set_base_shapes
    from omegaconf import read_write

    base_config = deepcopy(self.config)
    with read_write(base_config):
        base_config.model.hidden_size = base_config.model.n_heads # We need at least n_heads dim

    delta_config = deepcopy(base_config)
    with read_write(delta_config):
        delta_config.model.hidden_size = base_config.model.n_heads * 2

    base_model = models.dit.DIT(
        base_config, vocab_size=self.vocab_size, mask_index=self.mask_index, text_vocab_size=self.text_vocab_size, dtype=self.dtype
    )

    delta_model = models.dit.DIT(
        delta_config, vocab_size=self.vocab_size, mask_index=self.mask_index, text_vocab_size=self.text_vocab_size, dtype=self.dtype
    )

    set_base_shapes(_model, base_model, delta=delta_model)


def update_histogram(histogram, timesteps: torch.Tensor, losses: torch.Tensor):
    for t, l in zip(timesteps, losses):
        if t.item() in histogram:
            histogram[t.item()].append(l.item())
        else:
            histogram[t.item()] = [l.item()]

def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
        if not getattr(self.config.eval, 'big_seq_len_eval', False):
            assert seqlen == 2 * self.config.model.length
            # cropping is needed for text8-crop dataset
        # try the same starting point for now
        start = np.random.choice(self.config.model.length)
        end = start + self.config.model.length
        input_tokens = x0[:, start:end]
        output_tokens = x0[:, start + 1 : end + 1]
        new_attention_mask = attention_mask[:, start:end]

        # Helps with validation PPL, since the val
        # examples will all start and end with BOS/EOS
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    else:
        input_tokens = x0
        output_tokens = None
        new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

from unidisc.tokenizers.image_tokenizers import decode_latents


def viz_images_from_dataloader(self):
    _iter = iter(self.train_dataloader)
    random_elements = [next(_iter) for _ in range(10)]
    # random_elements[0]['input_ids'] - self.text_vocab_size
    out = decode_latents(self.config, self.get_vae(), torch.cat([torch.zeros_like(random_elements[0]['input_ids'][:, :1]), (random_elements[0]['input_ids'] - self.text_vocab_size)], dim=-1))
    from image_utils import Im
    print(Im(out[:16]).save())
    breakpoint()
    return random_elements

try:
    from torch.nn.attention.flex_attention import create_block_mask
except:
    pass

def _attn_mask(txt_batch_dropout, img_batch_dropout, txt_length):
    def mask_mod(b, h, q_idx, kv_idx):
        txt_sees_txt = (q_idx < txt_length) & (kv_idx < txt_length)
        img_sees_img_and_txt = (q_idx >= txt_length)
        txt_dropout_case = ~txt_batch_dropout[b] | (txt_sees_txt | img_sees_img_and_txt)

        img_sees_img = ((q_idx >= txt_length) & (kv_idx >= txt_length))
        txt_sees_txt_and_img = (q_idx < txt_length)
        img_dropout_case = ~img_batch_dropout[b] | (img_sees_img | txt_sees_txt_and_img)
        return txt_dropout_case & img_dropout_case
    return mask_mod


def get_block_mask(txt_batch_attn_dropout, img_batch_attn_dropout, txt_length, batch_size, seq_len, device):
    return create_block_mask(
        _attn_mask(txt_batch_attn_dropout, img_batch_attn_dropout, txt_length), 
        B = batch_size, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device
    )

def _interleaved_attn_mask(interleaved_sample_ids):
    def mask_mod(b, h, q_idx, kv_idx):
        return (interleaved_sample_ids[b, q_idx] == interleaved_sample_ids[b, kv_idx]) & (interleaved_sample_ids[b, q_idx] != -1)
    return mask_mod

def visualize_flex_attention(mask_mod, B, SEQ_LEN, H=16, HEAD_DIM=64, device="cuda"):
    from models.archived.utils import visualize_attention_scores
    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    visualize_attention_scores(
        query,
        key,
        mask_mod=mask_mod,
        device=device,
        name="interleaved_attn_mask",
    )
    
def get_interleaved_block_mask(interleaved_sample_ids, batch_size, seq_len, device, visualize=False):
    # Uncomment this to visualize the mask
    if visualize:
        visualize_flex_attention(_interleaved_attn_mask(interleaved_sample_ids), batch_size, seq_len, device=device)
    if (interleaved_sample_ids == -1).all(dim=-1).any():
        gprint(f"WARNING: Found all -1s in interleaved_sample_ids, setting one to 0")
        interleaved_sample_ids = interleaved_sample_ids.clone()
        interleaved_sample_ids[(interleaved_sample_ids == -1).all(dim=-1), 0] = 0

    return create_block_mask(
        _interleaved_attn_mask(interleaved_sample_ids),
        B = batch_size, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device
    )

def calculate_clip_score(
    image_paths: List[str],
    captions_mapping: Dict[str, str],
    device: torch.device = "cuda",
    seed: Optional[int] = 42,
    batch_size: int = 128,
    dataloader_workers: int = 16,
    verbose: bool = True,
):
    import clip
    from T2IBenchmark.feature_extractors import (BaseFeatureExtractor,
                                                 InceptionV3FE)
    from T2IBenchmark.loaders import CaptionImageDataset
    from T2IBenchmark.model_wrapper import (ModelWrapperDataloader,
                                            T2IModelWrapper)
    from T2IBenchmark.utils import dprint, set_all_seeds

    if seed:
        set_all_seeds(seed)

    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = CaptionImageDataset(
        images_paths=image_paths,
        captions=list(map(lambda x: captions_mapping[x], image_paths)),
        preprocess_fn=preprocess,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=dataloader_workers,
    )

    score_acc = 0.0
    num_samples = 0.0

    for image, caption in tqdm(dataloader):
        image_embedding = model.encode_image(image.to(device))
        caption_embedding = model.encode_text(clip.tokenize(caption, truncate=True).to(device))

        image_features = image_embedding / image_embedding.norm(dim=1, keepdim=True).to(
            torch.float32
        )
        caption_features = caption_embedding / caption_embedding.norm(
            dim=1, keepdim=True
        ).to(torch.float32)

        score = (image_features * caption_features).sum()
        score_acc += score
        num_samples += image.shape[0]

    clip_score = score_acc / num_samples
    dprint(verbose, f"CLIP score is {clip_score}")

    return clip_score

def get_chameleon_txt_indices(vae, include_special_tokens=True):
    image_indices = set(vae.chameleon_ori_translation.bpe2img.keys())
    if include_special_tokens:
        h_grids, w_grids = 32, 32
        image_start_token = vae.token2id(vae.image_start_token)
        n_grids_token = vae.token2id(vae.get_n_grids_token(h_grids))
        image_end_token = vae.token2id(vae.image_end_token)
        image_indices.add(image_start_token)
        image_indices.add(n_grids_token)
        image_indices.add(image_end_token)
        image_indices.add(-100)
        image_indices.add(1)
        image_indices.update(range(8192, 8820 + 1))

    return image_indices