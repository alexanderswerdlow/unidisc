import ast
from copy import deepcopy
import json
import math
import os
import pickle
import random
import shutil
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from constants import UNIDISC_DIR
from data_defs import InterleavedBatch
import einops
import numpy as np
from unidisc.utils.simple_llm import get_llm
from unidisc.utils.viz_utils import augment_image_with_random_object_coco, create_text_image
import torch
import torch.utils.checkpoint
from accelerate.utils import gather, gather_object
from image_utils import Im
from jaxtyping import Bool, Float, Integer
from PIL import Image
from tensordict import TensorDict, tensorclass
from torch import Tensor
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import utils
import wandb
from decoupled_utils import (barrier, dprint, get_num_gpus, get_rank, get_world_size,
                             gprint, is_main_process, print_memory_summary,
                             rprint, save_memory_profile, show_memory_usage, try_except, sanitize_filename)
from unidisc.tokenizers.chameleon_tokenizers import (decode_ids_batched,
                                                  get_chameleon_images)
from unidisc.tokenizers.image_tokenizers import decode_latents, get_image_batch
from unidisc.utils.throughput_monitor import get_available_flops
from model_utils import (_sample_categorical, empty_device_cache, get_chameleon_txt_indices, get_interleaved_block_mask, log,
                         remap_image_torch, replace_nan_dict,
                         wrapped_batch_decode)
from torch import nn
from model_utils import get_block_mask, MauveScore, Entropy

def get_anole_data(self, model, processor, prompt, image, dtype, device):
    inputs = processor(text=prompt, images=[image], padding=True, return_tensors="pt").to(device=device, dtype=dtype)
    image_tokens = model.model.get_image_tokens(inputs["pixel_values"])
    special_image_mask = inputs["input_ids"] == model.model.vocabulary_mapping.image_token_id
    image_tokens = image_tokens.to(inputs["input_ids"].device, inputs["input_ids"].dtype)
    inputs["input_ids"] = inputs["input_ids"].masked_scatter(special_image_mask, image_tokens)
    inputs.pop("pixel_values")
    return inputs

def calculate_chameleon_perplexity(self, model, processor, prompts, images, dtype=torch.bfloat16, return_all=False, standalone=False):
    """
    Calculate perplexities for multiple prompts and images using the Chameleon model.

    Args:
        model (ChameleonForConditionalGeneration): The Chameleon model.
        processor (ChameleonProcessor): The Chameleon processor.
        prompts (List[str]): List of prompt strings.
        images (List[Image.Image]): List of PIL Image objects.
        device (str): The device to use for computation (default: "cuda:0").
        dtype (torch.dtype): The data type to use (default: torch.bfloat16).

    Returns:
        List[float]: List of perplexities for each prompt-image pair.
    """
    device = self.device
    if model is None or processor is None:
        model = getattr(self, "chameleon_model", None)
        processor = getattr(self, "chameleon_processor", None)
        if model is None:
            from image_utils import Im
            from transformers import (ChameleonForConditionalGeneration, ChameleonProcessor)
            self.chameleon_model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", torch_dtype=torch.bfloat16).to("cuda")
            self.chameleon_processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
        
        model = self.chameleon_model
        processor = self.chameleon_processor
    assert len(prompts) == len(images), "Number of prompts and images must match"
    
    perplexities = []

    for prompt, image in zip(prompts, images):
        if not standalone:
            txt_first_prompt = f"{prompt} <image>"
            img_first_prompt = f"<image> {prompt}"
        else:
            txt_first_prompt = prompt
            img_first_prompt = "<image>"
        tot_ppl = 0.0
        tot_loss = 0.0
        img_loss = 0.0
        txt_loss = 0.0
        for i, _prompt in enumerate([txt_first_prompt, img_first_prompt]):
            inputs = self.get_anole_data(model, processor, _prompt, image, dtype, device)
            img_start_tok_id = self.chameleon_processor.tokenizer(self.chameleon_processor.image_start_token)['input_ids'][1]
            img_end_tok_id = self.chameleon_processor.tokenizer(self.chameleon_processor.image_end_token)['input_ids'][1]
            if i == 0:
                # text first
                mod_mask = torch.cumsum(inputs['input_ids'] == img_start_tok_id, dim=1).bool()
            else:
                # img first
                mod_mask = torch.cumsum(inputs['input_ids'] == img_end_tok_id, dim=1).bool()
            mod_mask = mod_mask.cumsum(dim=1) > 1
            output = model(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                labels=inputs['input_ids'].to(device)
            )
            loss = output.loss
            perplexity = torch.exp(loss).item()
            tot_ppl += perplexity
            logits = output.logits
            logits = logits.transpose(-1, -2)
            sample_chunk = inputs["input_ids"]
            nlls = F.cross_entropy(logits[..., :-1].to(self.device), sample_chunk[..., 1:].to(self.device), reduction="none")
            mod_mask = mod_mask[:, 1:]
            # img nll is where mod_mask == 1
            zeros = torch.zeros_like(nlls)
            img_nll = torch.where(mod_mask, nlls, zeros).mean().item()
            txt_nll = torch.where(~mod_mask, nlls, zeros).mean().item()
            tot_loss += loss.item()
            if not standalone:
                txt_loss += txt_nll
                img_loss += img_nll
            else:
                if i == 0:
                    txt_loss += loss.item()
                else:
                    img_loss += loss.item()

        if not standalone:
            tot_ppl /= 2
            tot_loss /= 2
            img_loss /= 2
            txt_loss /= 2

        if return_all:
            perplexities.append((tot_ppl, tot_loss, img_loss, txt_loss))
        else:
            perplexities.append(tot_ppl)

        print(f"Total PPL: {tot_ppl} | Total Loss: {tot_loss} | Img Loss: {img_loss} | Txt Loss: {txt_loss}")
    return perplexities

def get_every_n_evals(self, n):
    return (
        self.config.mode == "eval"
        or ((self.num_evals > 0 or getattr(self.config.eval, "log_on_start", False)) and n > 0 and self.num_evals % n == 0)
    ) and n != -1

@try_except(write_error_to_file=True)
def on_validation_epoch_start(self):
    rprint("on_validation_epoch_start")
    # EMA (Exponential Moving Average) is a technique used to maintain a moving average of model parameters
    # It can help stabilize training and potentially improve model performance
    if self.ema is not None and not self.config.trainer.use_custom_ema:
        # Store the current model parameters in the EMA object
        rprint(" [WARNING] USING EMA IN on_validation_epoch_start - THIS MIGHT RESET LOADED WEIGHTS ".center(100, "!"))
        self.ema.store(self.get_params())
        # Copy the EMA parameters to the current model
        self.ema.copy_to(self.get_params())

    self.backbone.eval()
    self.reset_validation_metrics()

    if getattr(self.config.trainer, "disable_torchmetrics", False) is False:
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0
    if self.non_embedding_params < 1e9:
        self.print_hashes()
    if (
        self.image_model
        and getattr(self.config.model, "image_model_fid_eval", False)
        and self.get_every_n_evals(getattr(self.config.eval, "log_every_n_fid", 10))
    ):
        
        self.fid_eval = True
        if self.config.eval.fid_mode == "inline":
            from vqgan.inception_metrics import MultiInceptionMetrics
            self.inception_metrics = MultiInceptionMetrics(
                reset_real_features=False,
                compute_unconditional_metrics=True,
                compute_conditional_metrics=False,
                compute_conditional_metrics_per_class=False,
                num_classes=1000,
                num_inception_chunks=10,
                manifold_k=3,
            )
            if self.config.mode == "eval":
                self.computed_tokens = []
        else:
            if getattr(self.config.eval, "force_fid_output_dir", None) is None:
                shm_path = Path("/dev/shm") / os.getenv("USER")
                fid_save_path = shm_path / Path(self.config.output_dir).parent.stem / Path(self.config.output_dir).stem / f"{self.num_evals}_{self.global_step}" / "fid_gen"
            else:
                fid_save_path = Path(getattr(self.config.eval, "force_fid_output_dir", None)) / "fid_gen"
            fid_save_path.mkdir(parents=True, exist_ok=True)
            fid_gt_path = fid_save_path.parent / (fid_save_path.name.replace("gen", "gt"))
            fid_gt_path.mkdir(parents=True, exist_ok=True)
            self.fid_gen_dir = fid_save_path
            self.fid_gt_dir = fid_gt_path
            rprint(f"FID eval output dir: {self.fid_gen_dir}, FID GT dir: {self.fid_gt_dir}")

        rprint(f"Setting FID eval for epoch {self.num_evals}")
    else:
        self.fid_eval = False
        if self.image_model and getattr(self.config.model, "image_model_fid_eval", False):
            rprint(f"Not setting FID eval: num_evals: {self.num_evals} % {getattr(self.config.eval, 'log_every_n_fid', 10)}")

    if self.config.eval.compute_img_to_txt_mauve_clip:
        shm_path = Path("/dev/shm") / os.getenv("USER")
        img_to_txt_mauve_save_path = shm_path / Path(self.config.output_dir).parent.stem / Path(self.config.output_dir).stem / f"{self.num_evals}_{self.global_step}" / "img_to_txt_mauve_gen"
        img_to_txt_mauve_save_path.mkdir(parents=True, exist_ok=True)
        img_to_txt_mauve_gt_path = img_to_txt_mauve_save_path.parent / (img_to_txt_mauve_save_path.name.replace("gen", "gt"))
        img_to_txt_mauve_gt_path.mkdir(parents=True, exist_ok=True)
        self.img_to_txt_mauve_gen_dir = img_to_txt_mauve_save_path
        self.img_to_txt_mauve_gt_dir = img_to_txt_mauve_gt_path
        rprint(f"Img to txt mauve eval gen dir: {self.img_to_txt_mauve_gen_dir}, gt dir: {self.img_to_txt_mauve_gt_dir}")

    self.saved_tokens = defaultdict(list)
    self.validation_start_time = time.time()

    if getattr(self.config.trainer, "attach_oom_observer_eval", False):
        from torchtnt.utils.oom import attach_oom_observer
        attach_oom_observer(output_dir=str(self.config.output_dir), trace_max_entries=1000000)
        rprint(f"Attached OOM observer to {self.config.output_dir}")
        self.gpu_memory_reserved = torch.cuda.memory_reserved()


def sample(self, return_input_ids=False, **kwargs):
    continuous_mode = self.config.trainer.image_mode == "continuous"
    text_only = kwargs.get("text_only", False)
    kwargs.pop("text_only", None)
    assert not continuous_mode
    txt_tokens, img_tokens = self._sample(text_only=text_only, **kwargs)
    if img_tokens is not None:
        img_pred = decode_latents(self.config, self.get_vae(), img_tokens)
    else:
        img_pred = None
    if txt_tokens is not None:
        txt_pred = wrapped_batch_decode(self.tokenizer, txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    else:
        txt_pred = None
    if return_input_ids:
        return txt_pred, img_pred, txt_tokens, img_tokens
    else:
        return txt_pred, img_pred


@torch.no_grad()
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    batch = self.update_batch(batch)
    assert (batch["input_ids"][~batch["x0_unmask"]] == self.mask_index).all()
    txt_pred, img_pred, txt_tokens, img_tokens = self.sample(x0=batch["input_ids"], x0_unmask=batch["x0_unmask"], return_input_ids=True)
    batch.update(dict(txt_pred=txt_pred, img_pred=img_pred, txt_tokens=txt_tokens, img_tokens=(img_tokens + self.text_vocab_size)))
    return batch

@torch.no_grad()
def zero_shot_eval_step(self, batch, batch_idx):
    batch = self.zero_shot_update_batch(batch)
    dataset_name = self.config.data.train
    
    def get_similarity(x0, batch, num_timesteps=None, txt_cond=True, return_unweighed=False, do_unconditional=False):
        # NOTE - this function assume [txt, img] order with self.config.model.txt_length + self.config.model.img_length
        # given a batch of img+text, get the similarity score
        return_unweighed = return_unweighed or getattr(self.config.eval, "return_unweighed_sim", False)
        class_log_probs = []
        unweighed_class_log_probs = []
        num_timesteps = num_timesteps or self.config.sampling.steps
        effective_batch_size = batch['modality'].shape[0]
        empty_device_cache()
        times = torch.linspace(0, 1, steps=num_timesteps + 2)[1:-1].to(self.device).to(torch.float32)
        
        if getattr(self.config.eval, "use_random_timesteps_same_batch", False):
            times = torch.rand(num_timesteps, device=x0.device)
            times = torch.sort(times)[0]
        
        if getattr(self.config.eval, "use_random_timesteps_diff_batch", False):
            # get a (B, num_timesteps) random timesteps
            times = torch.rand(effective_batch_size, num_timesteps, device=x0.device)
            times = torch.sort(times)[0]
            print(f'Times: {times}')
        
        do_unconditional = do_unconditional or getattr(self.config.eval, "do_unconditional", False)
        # unweighed/weighed, randomized but different over batch, randomized but same over batch, 
        cond_mask = torch.full_like(x0, False, device=x0.device).bool()
        if txt_cond:
            cond_mask[:, :self.config.model.txt_length] = True
        else:
            # img conditioned
            cond_mask[:, self.config.model.txt_length:] = True
        full_mask = torch.full_like(x0, self.mask_index, device=x0.device)
        pad_mask = x0 == self.tokenizer.pad_token_id
        rprint(f'Getting similarity with {times.shape[0]} timesteps, {effective_batch_size} samples, {do_unconditional} unconditional, {self.parameterization} parameterization, {self.config.eval.cfg} cfg, {num_timesteps} num_timesteps, {txt_cond} txt_cond')
        # for t in times:
        #     # t = self._sample_t(1, x0.device).expand(effective_batch_size)
        #     breakpoint()
        #     if getattr(self.config.eval, "`use_random_timesteps_diff_batch`", False):
        #         t = t.expand(effective_batch_size)
        #     else:
        #         t = t.expand(1)
        for i in range(num_timesteps):
            empty_device_cache()
            if getattr(self.config.eval, "use_random_timesteps_diff_batch", False):
                t = times[:, i]
            else:
                t = times[i]
                t = t.expand(effective_batch_size)
            sigma, dsigma = self.noise(t)
            # print(sigma, t)
            unet_conditioning = None # sigma[:, None] -> This causes CUDA OOM
            move_chance = 1 - torch.exp(-sigma[:, None])

            xt, ignore_batch_mask_for_metrics, joint_ar_nar_mask, _, __ = self.q_xt(x0, move_chance, return_ignore_batch_mask_for_metrics=True, batch=batch)
            if not do_unconditional:
                cond = torch.where(cond_mask, x0, xt)
                if self.config.eval.cfg is not None:
                    uncond = torch.where(cond_mask, full_mask, xt)
                    cond_output = self.forward(
                        cond, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=None, joint_ar_nar_mask=joint_ar_nar_mask, modality=batch['modality'], return_logits=True
                    )
                    uncond_output = self.forward(
                        uncond, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=None, joint_ar_nar_mask=joint_ar_nar_mask, modality=batch['modality'], return_logits=True
                    )
                    cat_output = torch.stack([cond_output, uncond_output])
                    logits = cfg(self.config, t, cat_output).squeeze(0)
                    model_output = self._subs_parameterization(logits, xt=xt, batch=batch, modality=batch['modality'])
                else:
                    # return logits false so already done with subs parameterization
                    model_output = self.forward(
                        cond, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=None, joint_ar_nar_mask=joint_ar_nar_mask, modality=batch['modality']
                    )
            else:
                if self.config.eval.cfg is not None:
                    uncond = torch.where(cond_mask, full_mask, xt)
                    cond_output = self.forward(
                        xt, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=None, joint_ar_nar_mask=joint_ar_nar_mask, modality=batch['modality'], return_logits=True
                    )
                    uncond_output = self.forward(
                        uncond, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=None, joint_ar_nar_mask=joint_ar_nar_mask, modality=batch['modality'], return_logits=True
                    )
                    cat_output = torch.stack([cond_output, uncond_output])
                    logits = cfg(self.config, t, cat_output).squeeze(0)
                    model_output = self._subs_parameterization(logits, xt=xt, batch=batch, modality=batch['modality'])
                else:
                    # return logits false so already done with subs parameterization
                    model_output = self.forward(
                        xt, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=None, joint_ar_nar_mask=joint_ar_nar_mask, modality=batch['modality']
                    )

                
            # print(f'Time: {t[0]}')
            log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
            # print(f'Log P Theta before pad remove: {-log_p_theta.mean()} | {(log_p_theta == 0).sum()}')
            zeros = torch.zeros_like(log_p_theta)
            log_p_theta = torch.where(pad_mask, zeros, log_p_theta)
            # zero out the loss on conditioned part
            if not do_unconditional:
                log_p_theta = torch.where(cond_mask, zeros, log_p_theta)
            # print(f'Log P Theta after pad remove: {-log_p_theta.mean()} | {(log_p_theta == 0).sum()}')
            std_weighting = (dsigma / torch.expm1(sigma))[:, None]
            unweighed_log_p_theta = -log_p_theta
            loss = -log_p_theta * std_weighting
            log_probs = loss.sum(dim=-1) / (~pad_mask).sum(dim=-1)
            unweighed_log_probs = unweighed_log_p_theta.sum(dim=-1) / (~pad_mask).sum(dim=-1)
            # print(f'Weighed loss: {log_probs.mean()} | Log P Theta: {-log_p_theta.mean()} | Std Weighting: {std_weighting.mean()}')
            class_log_probs.append(log_probs)
            unweighed_class_log_probs.append(unweighed_log_probs)
        overall_time_log_probs = torch.stack(class_log_probs) # (num_time, B)
        unweighed_overall_time_log_probs = torch.stack(unweighed_class_log_probs) # (num_time, B)
        if return_unweighed:
            return unweighed_overall_time_log_probs.mean(dim=0) # (B)
        return overall_time_log_probs.mean(dim=0) # (B)

    def get_similarity_ar(x0, batch, txt_cond=True, do_unconditional=False, **kwargs):
        # get likelihood for each token and then average
        img_first = kwargs.get("img_first", False)
        if img_first:
            x0 = torch.cat([x0[:, self.config.model.txt_length:], x0[:, :self.config.model.txt_length]], dim=1)
            mod = batch['modality']
            mod = torch.cat([mod[:, self.config.model.txt_length:], mod[:, :self.config.model.txt_length]], dim=1)
        else:
            mod = batch['modality']
        empty_device_cache()
        do_unconditional = do_unconditional or getattr(self.config.eval, "do_unconditional", False)

        if getattr(self.config.eval, "cfg", None):
            rprint('NOT SETTING CFG for AR')
        # if getattr(self.config.eval, "cfg", None):
        #     cat_mod_input_ids = torch.cat([x0, torch.where(batch['modality'] == 1, self.mask_index, x0)], dim=0)
        #     _modality = torch.cat([batch['modality'], batch['modality']], dim=0)
        #     cat_p_x0 = self.forward(
        #             cat_mod_input_ids, 
        #             sigma=None, 
        #             batch=dict(modality=_modality), modality=_modality
        #         )
        #     logit_c, logit_u = cat_p_x0.chunk(2, dim=0)
        #     _w = getattr(self.config.eval, "cfg", None)
        #     model_output = (1 + _w) * logit_c - _w * logit_u
        # else:
        model_output = self.forward(x=x0, sigma=None, modality=mod)
        x0 = x0[:, 1:]
        # attention_mask = batch['attention_mask'][0][None, :].repeat(x0.shape[0], 1)[:, 1:]
        attention_mask = x0 != self.tokenizer.pad_token_id
        log_p_theta = model_output.gather(-1, x0[:, :, None])[:, :, 0]
        if img_first:
            txt_sl = slice(self.config.model.img_length-1, None)
            img_sl = slice(None, self.config.model.img_length-1)
        else:
            txt_sl = slice(None, self.config.model.txt_length - 1)
            img_sl = slice(self.config.model.txt_length - 1, None)
        nll = (-log_p_theta * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)
        txt_nll = (-log_p_theta[:, txt_sl] * attention_mask[:, txt_sl]).sum(dim=-1) / attention_mask[:, txt_sl].sum(dim=-1)
        img_nll = (-log_p_theta[:, img_sl] * attention_mask[:, img_sl]).sum(dim=-1) / attention_mask[:, img_sl].sum(dim=-1)
        if do_unconditional:
            return nll
        return img_nll if txt_cond else txt_nll
    
    def get_similarity_chameleon(zipp, batch, txt_cond=True, do_unconditional=False, prompts=None, images=None, **kwargs):
        # get likelihood for each token and then average
        empty_device_cache()
        img_first = kwargs.get("img_first", False)
        img_start_tok_id = self.chameleon_processor.tokenizer(self.chameleon_processor.image_start_token)['input_ids'][1]
        img_end_tok_id = self.chameleon_processor.tokenizer(self.chameleon_processor.image_end_token)['input_ids'][1]
        do_unconditional = do_unconditional or getattr(self.config.eval, "do_unconditional", False)
        if not prompts and not images:
            prompt, image = zipp
            if img_first:
                _prompt = f"<image> {prompt}"
            else:
                _prompt = f"{prompt} <image>"
            inputs = self.get_anole_data(self.chameleon_model, self.chameleon_processor, _prompt, image, dtype=self.dtype, device=self.device)

        else:
            inputs = self.get_anole_data(self.chameleon_model, self.chameleon_processor, prompts, images, dtype=self.dtype, device=self.device)
        # mod mask which is one for image tokens from the indx we see img_start_tok_id to img_end_tok_id
        
        if img_first:
            mod_mask = torch.cumsum(inputs['input_ids'] == img_end_tok_id, dim=1).bool()
        else:
            mod_mask = torch.cumsum(inputs['input_ids'] == img_start_tok_id, dim=1).bool()
        
        mod_mask = mod_mask.cumsum(dim=1) > 1
        output = self.chameleon_model(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            labels=inputs['input_ids'].to(self.device)
        )   
        loss = output.loss
        logits = output.logits
        logits = logits.transpose(-1, -2)
        sample_chunk = inputs["input_ids"]
        nlls = F.cross_entropy(logits[..., :-1].to(self.device), sample_chunk[..., 1:].to(self.device), reduction="none")
        mod_mask = mod_mask[:, 1:]
        # img nll is where mod_mask == 1
        zeros = torch.zeros_like(nlls)
        img_nll = torch.where(mod_mask, nlls, zeros)
        txt_nll = torch.where(~mod_mask, nlls, zeros)
        if do_unconditional:
            return nlls.mean(dim=-1)
        return img_nll.mean(dim=-1) if txt_cond else txt_nll.mean(dim=-1)
    
    if dataset_name == "nlphuji/flickr30k":
        txt_tokens, img_tokens = self._sample(
            text_only=False,
            x0=batch["input_ids"],
            x0_unmask=batch["attention_mask"],
            modality=batch["modality"],
        )
        img_samples = decode_latents(self.config, self.get_vae(), img_tokens[:, :self.config.model.img_length])
        txt_samples = wrapped_batch_decode(self.tokenizer, txt_tokens[:, self.config.model.img_length:], clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        gt_text_samples = wrapped_batch_decode(self.tokenizer, batch['gt_input_ids'][:, :self.config.model.txt_length], skip_special_tokens=True, clean_up_tokenization_spaces=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        self.compute_cider(txt_samples, gt_text_samples)
    elif dataset_name == "facebook/winoground":
        # breakpoint()
        # if batch_idx <= 15:
        #     return
        a0_0 = batch["input_ids_0_0"] # a
        a0_1 = batch["input_ids_0_1"] # d
        a1_0 = batch["input_ids_1_0"] # b
        a1_1 = batch["input_ids_1_1"] # c
        
        text_correct_count = 0
        image_correct_count = 0
        group_correct_count = 0
        
        wino_chameleon = getattr(self.config.eval, "wino_chameleon", False)

        s0_0, s0_1, s1_0, s1_1 = None, None, None, None
        modes = ['image', 'text', 'group']
        
        if wino_chameleon:
            txt0 = wrapped_batch_decode(tokens=batch['caption_0_input_ids'], tokenizer=self.tokenizer, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)[0]
            txt1 = wrapped_batch_decode(tokens=batch['caption_1_input_ids'], tokenizer=self.tokenizer, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)[0]
            img0 = Im(batch['img_0']).pil
            img1 = Im(batch['img_1']).pil
            prompts = [txt0, txt0, txt1, txt1]
            images = [img0, img1, img0, img1]
            zipp = list(zip(prompts, images))

        # note - signs are reversed since we have loss, so want to minimize instead of maximize
        def text_correct(result):
            return torch.logical_and(result["s0_i0"] < result["s1_i0"], result["s1_i1"] < result["s0_i1"])
        
        def image_correct(result):
            return torch.logical_and(result["s0_i0"] < result["s0_i1"], result["s1_i1"] < result["s1_i0"])
        
        def group_correct(result):
            return torch.logical_and(image_correct(result), text_correct(result))
        results_cond = {}
        for mode in modes:
            do_unconditional = (mode == 'group')
            txt_cond = not (mode == 'text')
            img_first = mode == 'text'
            if wino_chameleon:
                do_unconditional = True
                s0_0 = get_similarity_chameleon(zipp[0], batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
                s0_1 = get_similarity_chameleon(zipp[1], batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
                s1_0 = get_similarity_chameleon(zipp[2], batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
                s1_1 = get_similarity_chameleon(zipp[3], batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
            elif self.parameterization == "ar":
                s0_0 = get_similarity_ar(a0_0, batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
                s0_1 = get_similarity_ar(a0_1, batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
                s1_0 = get_similarity_ar(a1_0, batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
                s1_1 = get_similarity_ar(a1_1, batch, txt_cond=False, do_unconditional=do_unconditional, img_first=img_first)
            else:
                s0_0 = get_similarity(a0_0, batch, txt_cond=txt_cond, do_unconditional=do_unconditional)
                s0_1 = get_similarity(a0_1, batch, txt_cond=txt_cond, do_unconditional=do_unconditional)
                s1_0 = get_similarity(a1_0, batch, txt_cond=txt_cond, do_unconditional=do_unconditional)
                s1_1 = get_similarity(a1_1, batch, txt_cond=txt_cond, do_unconditional=do_unconditional)
            result = {
                "s0_i0": s0_0,
                "s0_i1": s0_1,
                "s1_i0": s1_0,
                "s1_i1": s1_1,
            }
            if mode == 'text':
                results_cond['text'] = text_correct(result)
                text_correct_count += text_correct(result).sum().item()
            elif mode == 'image':
                results_cond['image'] = image_correct(result)
                image_correct_count += image_correct(result).sum().item()
            elif mode == 'group':
                if getattr(self.config.eval, "wino_group_conditional", False):
                    rprint('[Winoground] Using conditional group accuracy')
                    group_correct_count = (torch.logical_and(results_cond['text'], results_cond['image'])).sum().item()
                else:
                    rprint('[Winoground] Using unconditional group accuracy')
                    group_correct_count += group_correct(result).sum().item()
        bsz = a0_0.shape[0]
        txt_acc = text_correct_count / bsz
        img_acc = image_correct_count / bsz
        group_acc = group_correct_count / bsz        
        
        self.win_text_accuracy.update(txt_acc)
        self.win_image_accuracy.update(img_acc)
        self.win_group_accuracy.update(group_acc)
        running_avg_txt = self.win_text_accuracy.compute()
        running_avg_img = self.win_image_accuracy.compute()
        running_avg_group = self.win_group_accuracy.compute()
        rprint(f"[{batch_idx}] Winoground Text Accuracy: {txt_acc} ({running_avg_txt}), Image Accuracy: {img_acc} ({running_avg_img}), Group Accuracy: {group_acc} ({running_avg_group})")
    else:
        # def randomize_batch - input is a batch. for the batch['input_ids'] which contains self.config.model.txt_length txt tokens + self.config.model.img_length img tokens which are PAIRED
        # we want to randomly swap the img/txt tokens between each other
        x0 = batch['input_ids']
        img_first = getattr(self.config.model, "img_first", False)
        only_one_correct = getattr(self.config.eval, "only_one_correct", False)
        wino_chameleon = getattr(self.config.eval, "wino_chameleon", False)
        # todo check attn mask for text retrieval
        x0_txt = x0.clone()
        x0_img = x0.clone()
        if only_one_correct:
            # for each sample from 1st batch onwards, shuffle the img/txt tokens, as in map randomly
            x0c = x0.clone()
            if img_first:
                second_half = x0c[1:, self.config.model.img_length:]
            else:
                second_half = x0c[1:, self.config.model.txt_length:]
            # shuffle second half
            # second_half = second_half[torch.randperm(second_half.size(0))]
            second_half = torch.cat([second_half[1:], second_half[0].unsqueeze(0)], dim=0)
            # replace img tokens with txt tokens
            if img_first:
                x0c[1:, self.config.model.img_length:] = second_half
            else:
                x0c[1:, self.config.model.txt_length:] = second_half
            if wino_chameleon:
                if img_first:
                    img_tokens = x0c[:, :self.config.model.img_length]
                    txt_tokens = x0c[:, self.config.model.img_length:]
                else:
                    txt_tokens = x0c[:, :self.config.model.txt_length]
                    img_tokens = x0c[:, self.config.model.txt_length:]
                dec_txt = wrapped_batch_decode(self.tokenizer, txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
                dec_imgs = decode_latents(self.config, self.get_vae(), img_tokens - self.text_vocab_size)
                dec_imgs = [Im(img).pil for img in dec_imgs]
                if img_first:
                    # append '<image>' to beginning of each txt sample
                    dec_txt = ['<image> ' + txt for txt in dec_txt]
                else:
                    dec_txt = [txt + ' <image>' for txt in dec_txt]
                class_sim = get_similarity_chameleon(None, batch, do_unconditional=True, img_first=img_first, prompts=dec_txt, images=dec_imgs)
                if torch.isinf(class_sim).any():
                    rprint(f'[Chameleon] Inf found in class_sim, check transformers version')
                    breakpoint()
            elif self.parameterization == "ar":
                class_sim = get_similarity_ar(x0c, batch, do_unconditional=True)
            else:
                class_sim = get_similarity(x0c, batch, do_unconditional=True)
                
            topk = class_sim.topk(k=1, dim=0, largest=False)
            topk_indices = topk.indices
            topk_acc = (topk_indices == 0).float().mean().item()
            rprint(f"[{batch_idx}] Datacomp Correct Pair Retrieval Acc: {topk_acc} ({self.datacomp_img_acc.compute()})")
            self.datacomp_img_acc.update(topk_acc)
        else:
            if img_first:
                # image retrieval given text, so fix text
                x0_txt[:, self.config.model.img_length:] = x0[0, self.config.model.img_length:] # make all texts the first text  
                
                # text retrieval given image
                x0_img[:, :self.config.model.img_length] = x0[0, :self.config.model.img_length] # make all images the first image
            else:
                # image retrieval given text, so fix text
                x0_txt[:, :self.config.model.txt_length] = x0[0, :self.config.model.txt_length] # make all texts the first text 
            
                # text retrieval given image
                x0_img[:, self.config.model.txt_length:] = x0[0, self.config.model.txt_length:] # make all images the first image

            if self.parameterization == "ar":
                txt_class_sim = get_similarity_ar(x0_txt, batch, txt_cond=True)
                img_class_sim = get_similarity_ar(x0_img, batch, txt_cond=True) # TODO MAYBE REVERT?
            else:
                txt_class_sim = get_similarity(x0_txt, batch, txt_cond=True)
                img_class_sim = get_similarity(x0_img, batch, txt_cond=False)
                
            img_topk = img_class_sim.topk(k=1, dim=0, largest=False)
            txt_topk = txt_class_sim.topk(k=1, dim=0, largest=False)
            
            img_topk_indices = img_topk.indices
            txt_topk_indices = txt_topk.indices
            
            img_acc = (img_topk_indices == 0).float().mean().item()
            txt_acc = (txt_topk_indices == 0).float().mean().item()
            rprint(f"[{batch_idx}] Datacomp Text Retrieval Acc: {img_acc}, Datacomp Image Retrieval Accuracy: {txt_acc}")
            self.datacomp_img_acc.update(img_acc)
            self.datacomp_txt_acc.update(txt_acc)
        # img_class_sim is (B) - argmin since loss txt_conds
        
@torch.no_grad()
def validation_step(self, batch, batch_idx):
    batch = self.update_batch(batch)
    continuous_mode = self.config.trainer.image_mode == "continuous"

    if self.config.mode == "eval":
        logs = dict()
        logs["gpu_max_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
        logs["gpu_cur_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        logs["gpu_max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        logs["gpu_cur_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        log({**logs, **self.get_step_metrics()})

    if self.get_every_n_evals(getattr(self.config.eval, "log_every_n_evals", 10)) \
        and self.image_model \
        and (batch_idx == 0 or self.config.eval.visualize_data_only) \
        and not continuous_mode:
        self.visualize_samples(batch, batch_idx)
        if self.config.eval.visualize_data_only: return
    
    if batch_idx < self.config.eval.num_sample_batches and self.config.eval.compute_generative_perplexity:
        if continuous_mode:
            # todo update to use modality once multimodal batches update is done by alex
            gt_text_samples = wrapped_batch_decode(self.tokenizer, batch['text_tokens'][:, :self.config.model.txt_length], skip_special_tokens=True, clean_up_tokenization_spaces=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos) # since input_ids is for images
        else:
            input_ids = batch["input_ids"]
            pad_tokens = torch.full_like(input_ids, self.tokenizer.pad_token_id)
            text_tokens = torch.where(batch["modality"] == 0, input_ids, pad_tokens)
            gt_text_samples = wrapped_batch_decode(self.tokenizer, text_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        if getattr(self.config.trainer, "disable_text_modality", False):
            gt_text_samples = [' ']
        self.compute_generative_perplexity(gt_text_samples, gt=True)
    
    if getattr(self.config.trainer, "log_flops", False) \
        and batch_idx == 0 \
        and self.current_run_global_step <= 1 \
        and self.config.trainer.fsdp is False:
        self.log_flops(batch=batch, batch_idx=batch_idx)
    if self.fid_eval:
        if self.config.eval.fid_mode == "inline":
            self.update_inline_fid(batch, batch_idx)
        elif self.config.eval.fid_mode == "clean":
            self.update_clean_fid(batch, batch_idx)
        else:
            raise ValueError(f"Invalid FID mode: {self.config.eval.fid_mode}")

    if getattr(self.config.eval, "get_top_k", False) and self.config.parameterization == "ar":
        self.get_top_k(batch, batch_idx)

    try:
        if self.config.eval.compute_img_to_txt_mauve_clip and not self.config.eval.unconditional_fid:
            self.update_img_to_txt_mauve_clip(batch, batch_idx)
    except Exception as e:
        empty_device_cache()
        rprint(f"Error in update_img_to_txt_mauve_clip: {e}")

    if (self.get_every_n_evals(getattr(self.config.eval, "log_every_n_evals", 10)) \
        and continuous_mode \
        and self.config.eval.generate_samples \
        and not self.config.eval.test_eval_speed):
        # todo remove this from here and move to on_validation_epoch_end
        data = self.sample_transfusion(batch_size_per_gpu=batch['input_ids'].shape[0])
        # TODO @sid support batching. prob pass list of lists to be general.
        rec_embs = [data.xt_img_embed[i, data.modality[i] == 1] for i in range(data.shape[0])] 
        # stack and transpose
        rec_embs = torch.stack(rec_embs)
        rec_txt = data.xt_ids[data.modality == 0][None]
        recon_image = decode_latents(self.config, self.get_vae(), rec_embs, batched=True) # TODO @sid support batching e.g. not just first element. prob pass list of lists to be general.
        txt = wrapped_batch_decode(self.tokenizer, rec_txt, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        rprint(f"Sampled {len(txt)} text samples:\n {txt[:1][:50]}")
        image_list = [wandb.Image(img) for img in recon_image]
        val_loss = self.compute_loss(batch, prefix="val")
        log({"val/gen_img": image_list, "val/loss": val_loss, **self.get_step_metrics()})

    if (
        self.get_every_n_evals(getattr(self.config.eval, "log_every_n_evals", 10))
        and (self.unified_model or self.cub_model or self.vggface_model)
        and batch_idx < getattr(self.config.eval, "num_masking_viz_batches", 1)
        and not continuous_mode # todo add masking val support s
    ):
        self.sample_masking(batch=batch, batch_idx=batch_idx)

    return self.compute_loss(batch, prefix="val", batch_idx=batch_idx)

@try_except(write_error_to_file=True)
@torch.no_grad()
def zero_shot_eval_epoch_end(self, example_batch=None):
    dataset_name = self.config.data.train
    dprint("zero_shot_eval_epoch_end")
    if dataset_name == "nlphuji/flickr30k":
        cider_score = self.cider_score.compute()
        rprint('Flickr30k CIDEr score: ', cider_score)
        # log it
        log({
            'val/cider_score': cider_score
        })
    elif dataset_name == "facebook/winoground":
        win_text_accuracy = self.win_text_accuracy.compute()
        win_image_accuracy = self.win_image_accuracy.compute()
        win_group_accuracy = self.win_group_accuracy.compute()
        rprint(f'Winoground Text Accuracy: {win_text_accuracy}')
        rprint(f'Winoground Image Accuracy: {win_image_accuracy}')
        rprint(f'Winoground Group Accuracy: {win_group_accuracy}')
        # log it
        log({
            'val/win_text_accuracy': win_text_accuracy,
            'val/win_image_accuracy': win_image_accuracy,
            'val/win_group_accuracy': win_group_accuracy
        })
    else:
        datacomp_img_acc = self.datacomp_img_acc.compute()
        datacomp_txt_acc = self.datacomp_txt_acc.compute()
        rprint(f'Datacomp Text Accuracy: {datacomp_img_acc}')
        rprint(f'Datacomp Image Accuracy: {datacomp_txt_acc}')
        # log it
        log({
            'val/datacomp_text_retr_acc': datacomp_img_acc,
            'val/datacomp_img_retr_acc': datacomp_txt_acc
        })
        
@try_except(write_error_to_file=True)
@torch.no_grad()
def get_img_text_saturation_batch(self, example_batch):
    max_sampling_steps = self.config.model.length
    batch_size_per_gpu = example_batch["input_ids"].shape[0]
    do_standalone = getattr(self.config.eval, "cham_standalone", False)
    pplx_per_step = []
    # make stpes linspace between 1 and max_sampling_steps with 100 steps
    # steps = np.linspace(1, max_sampling_steps, 10).astype(int)
    # steps = [1,2,4,8,16,32,64,128,256,512,1024]
    steps = [1,2,4,8,16,32,64] # todo revert

    rprint(f"do_standalone: {do_standalone} with steps: {steps}")
    dec_txt_list = []
    dec_img_list = []
    for step in steps:
        rprint(f"Step: {step}")
        (txt_tokens, img_tokens), nfe_cnt = self._sample(text_only=False, batch_size_per_gpu=batch_size_per_gpu, sample_modality=example_batch["modality"], return_nfe=True, num_steps=step)
        decoded_img = Im(decode_latents(self.config, self.get_vae(), img_tokens)).pil
        decoded_txt = wrapped_batch_decode(self.tokenizer, txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        if not isinstance(decoded_img, list):
            decoded_img = [decoded_img]
        if not isinstance(decoded_txt, list):
            decoded_txt = [decoded_txt]
        dec_txt_list.append(decoded_txt)
        dec_img_list.append(decoded_img)
        tot_ppl, tot_loss, img_loss, txt_loss = self.calculate_chameleon_perplexity(self.chameleon_model, self.chameleon_processor, prompts=decoded_txt, images=decoded_img, return_all=True)[0]
        rprint(f"Step {step} - Total PPL: {tot_ppl} | Total Loss: {tot_loss} | Img Loss: {img_loss} | Txt Loss: {txt_loss}")
        pplx_per_step.append((step, tot_ppl, tot_loss, img_loss, txt_loss))
        empty_device_cache()
    return dec_txt_list, dec_img_list, pplx_per_step

@torch.no_grad()
@try_except(write_error_to_file=True)
@torch.no_grad()
def on_validation_epoch_end(self, example_batch=None):
    dprint("on_validation_epoch_end")

    if self.config.eval.compute_val_metrics_standalone:
        self.compute_val_metrics_standalone()

    all_val_metrics = self.get_step_metrics()
    all_val_metrics.update(self.valid_metrics.compute())
    if hasattr(self, "valid_txt_metrics"):
        valid_txt_metrics = self.valid_txt_metrics.compute()
        valid_img_metrics = self.valid_img_metrics.compute()
        all_val_metrics.update({
            **{f"val/txt_{k.split('/')[-1]}": v for k, v in replace_nan_dict(valid_txt_metrics).items()},
            **{f"val/img_{k.split('/')[-1]}": v for k, v in replace_nan_dict(valid_img_metrics).items()},
        })

    log(all_val_metrics)

    gprint("example_batch['input_ids'].ndim: ", example_batch['input_ids'].ndim)
    if example_batch['input_ids'].ndim == 3:
        combined_batches = example_batch
        example_batch = self.update_batch(example_batch[0])
    else:
        example_batch = self.update_batch(example_batch)

    if self.config.eval.auto_enhance:
        self.auto_enhance(combined_batches)

    continuous_mode = self.config.trainer.image_mode == "continuous"
    compute_chameleon_perplexity = getattr(self.config.eval, "compute_chameleon_perplexity", False)
    all_images = []
    with try_except(write_error_to_file=True, clear_cuda_cache=True):
        if self.fid_eval:
            if self.config.eval.fid_mode == "inline":
                self.compute_inline_fid_eval()
            elif self.config.eval.fid_mode == "clean":
                self.compute_clean_fid_eval()
            else:
                raise ValueError(f"Invalid FID mode: {self.config.eval.fid_mode}")
            
            if self.config.eval.calculate_clip_score:
                prefix = "unconditional" if self.config.eval.unconditional_fid else "fid"
                self.compute_clip_score(self.fid_gen_dir, f"{prefix}_gen")
                self.compute_clip_score(self.fid_gt_dir, f"{prefix}_gt")
                if self.config.trainer.ar_inpainting:
                    import shutil
                    target_dir = Path(self.fid_gt_dir).parent / "fid_inpainting"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    for img_file in Path(self.fid_gt_dir).rglob("*.png"):
                        shutil.copy2(img_file, target_dir / img_file.name)

                    for json_file in Path(self.fid_gen_dir).rglob("*.json"):
                        shutil.copy2(json_file, target_dir / json_file.name)

                    self.compute_clip_score(target_dir, f"{prefix}_inpainting")

        if self.config.eval.unconditional_fid and \
            self.config.eval.compute_img_to_txt_mauve_during_unconditional_fid and self.config.eval.compute_img_to_txt_mauve_clip:
            rprint("Computing img to txt mauve during unconditional fid")
            # CLIP score is the same as the fid clip score so we don't need to compute it again
            gen_txt_tokens = self.gather_tokens(self.saved_tokens["unconditional_gen_txt_tokens"])
            gt_txt_tokens = self.gather_tokens(self.saved_tokens["unconditional_gt_txt_tokens"])
            if not getattr(self.config.eval, "global_disable_mauve", False):
                self.compute_mauve_entropy(self.fid_gen_dir, self.fid_gt_dir, gen_txt_tokens, gt_txt_tokens, "unconditional")
        elif self.config.eval.compute_img_to_txt_mauve_clip:
            gen_txt_tokens = self.gather_tokens(self.saved_tokens["img_to_txt_gen_txt_tokens"])
            gt_txt_tokens = self.gather_tokens(self.saved_tokens["img_to_txt_gt_txt_tokens"])
            if not getattr(self.config.eval, "global_disable_mauve", False):
                self.compute_mauve_entropy(self.img_to_txt_mauve_gen_dir, self.img_to_txt_mauve_gt_dir, gen_txt_tokens, gt_txt_tokens, "img_to_txt")
            if self.config.eval.calculate_clip_score:
                self.compute_clip_score(self.img_to_txt_mauve_gen_dir, "img_to_txt_mauve_gen")
                self.compute_clip_score(self.img_to_txt_mauve_gt_dir, "img_to_txt_mauve_gt")
            self.compute_mauve_entropy(self.img_to_txt_mauve_gen_dir, self.img_to_txt_mauve_gt_dir, gen_txt_tokens, gt_txt_tokens, "img_to_txt")

    should_eval_speed = getattr(self.config.eval, "test_eval_speed", False)
    if self.config.eval.generate_samples:
        with try_except(write_error_to_file=True):
            empty_device_cache()
            if getattr(self.config.eval, 'set_random_gen_seed', False):
                new_seed = get_rank() * 10 + 32
                torch.manual_seed(new_seed)
                torch.cuda.manual_seed(new_seed)
                random.seed(new_seed)
                np.random.seed(new_seed)

            tot_time_per_sample = []
            tot_token_time_per_token = []
            tot_nfe_cnt = 0
            batch_size_per_gpu = self.config.loader.eval_batch_size
            sampling_steps = self.config.sampling.steps
            num_batches = self.config.eval.num_sample_batches
            gen_ppl_max_batches = 1e8
            compute_entropy = getattr(self.config.eval, "compute_entropy", False)
            compute_gen_ppl =  self.config.eval.compute_generative_perplexity
            entropies = []
            
            if self.config.eval.compute_standalone_mauve and not getattr(self.config.eval, "global_disable_mauve", False):
                mauve_N = self.config.eval.mauve_num_samples
                # we need to generate this many samples distributed over the batch size * num_gpus
                # if not clean division, generate one extra batch we can discard later
                num_batches = math.ceil(mauve_N / (batch_size_per_gpu * get_num_gpus()))
                should_eval_speed = True # if we are generating this many samples might as well time it
                gen_ppl_max_batches = getattr(self.config.eval, "gen_ppl_max_batches", 1e8) # since we are generating a lot of samples, we can compute gen ppl for a few batches but not all since that'll be slow with eval_mode = llama
                compute_entropy = True
                compute_gen_ppl = True
                rprint(f"[MAUVE] Generating {mauve_N} samples with batch size {batch_size_per_gpu}, sampling steps {sampling_steps}, total length {self.config.model.length}, num_batches: {num_batches}, max_gen_ppl_batches: {gen_ppl_max_batches}")
                
            rprint(f"Generating {num_batches} samples with batch size {batch_size_per_gpu}, sampling steps {sampling_steps}, total length {self.config.model.length}, compute_entropy: {compute_entropy}, compute_gen_ppl: {compute_gen_ppl}")
            all_samples = []
            get_img_text_saturation = getattr(self.config.eval, "get_img_text_saturation", False)
            for i in tqdm(range(num_batches), desc="Generating samples"):
                if get_img_text_saturation:
                    dec_txt_list, dec_img_list, all_vals = self.get_img_text_saturation_batch(example_batch)
                    # Prepare data for logging
                    df = pd.DataFrame(all_vals, columns=["step", "tot_ppl", "tot_loss", "img_loss", "txt_loss"])
                    df.to_csv(Path(self.config.output_dir) / f"img_text_saturation_batch_{i}.csv", index=False)
                    rprint(f"Saved img_text_saturation_batch_{i}.csv to {Path(self.config.output_dir) / f'img_text_saturation_batch_{i}.csv'}")

                    log_data = []
                    for (step, tot_ppl, tot_loss, img_loss, txt_loss), dec_txt, dec_img in zip(all_vals, dec_txt_list, dec_img_list):
                        concatenated_text = ' | '.join(dec_txt)
                        concatenated_image = dec_img[0]
                        log_data.append([step, tot_ppl, tot_loss, img_loss, txt_loss, concatenated_text, wandb.Image(concatenated_image)])

                    # Log to wandb
                    log_table = wandb.Table(columns=["Step", "Total PPL", "Total Loss", "Image Loss", "Text Loss", "Generated Text", "Generated Image"], data=log_data)
                    wandb.log({"img_text_saturation": log_table, "trainer/global_step": self.global_step})
                    rprint("Logged img_text_saturation table to wandb")
                    # log (step, Im)
                    # make it into pd df and store in output_dir
                    break
                if should_eval_speed:
                    start_time = start_timing(sync=True, enable=True, message="Evaluating inference speed")

                if self.parameterization == "ar" and continuous_mode:
                    data = self.sample_transfusion(text_only=True, batch_size_per_gpu=batch_size_per_gpu)
                    txt_tokens = data.xt_ids[:, self.static_txt_sl]
                else:
                    (txt_tokens, img_tokens), nfe_cnt = self._sample(
                        text_only=False,
                        batch_size_per_gpu=batch_size_per_gpu, 
                        sample_modality=example_batch["modality"], 
                        return_nfe=True,
                    )
                    tot_nfe_cnt += nfe_cnt
                if should_eval_speed:
                    tot_time = end_timing(start_time, enable=True, sync=True)
                    if continuous_mode: assert (data.modality == 0).all()
                    tot_time_per_sample.append(tot_time)
                    tot_token_time_per_token.append((tot_time) / self.config.model.length)
                        
                if compute_entropy:
                    entropies.append(self.compute_entropy(txt_tokens).item())
                    
                if compute_chameleon_perplexity:
                    all_images.extend(Im(decode_latents(self.config, self.get_vae(), img_tokens)).pil)
                text_samples = wrapped_batch_decode(self.tokenizer, txt_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                if self.config.eval.compute_standalone_mauve and not getattr(self.config.eval, "global_disable_mauve", False):
                    self.mauve_predictions.extend(text_samples)
                if len(text_samples) > 0 and len(text_samples[0]) > 0 and self.config.eval.compute_generative_perplexity and i <= gen_ppl_max_batches:
                    self.compute_generative_perplexity(text_samples)

                rprint(f"Generated {len(text_samples)} samples - {[text_samples[i][:200] for i in range(min(len(text_samples), 5))]}")
                all_samples.extend(text_samples)

            # TODO: @ssg2 is this needed?
            # Log the last generated samples
            # if not compute_chameleon_perplexity:
            #     text_samples = all_samples[:self.config.sampling.num_sample_log]
            #     all_images = all_images[:self.config.sampling.num_sample_log]

            avg_nfe_cnt = tot_nfe_cnt / num_batches
            if should_eval_speed:
                
                # TODO: @ssg2 is this needed?
                # data_dict = {
                #     f"samples": wandb.Table(columns=["Generated Samples", "Time per sample", "Time per token", "Generated Images"], data=[[s, t, tt, wandb.Image(img)] for s, t, tt, img in zip(text_samples, tot_time_per_sample, tot_token_time_per_token, all_images )]),
                #     "trainer/global_step": self.global_step,
                # }

                data_dict = {
                    f"samples": wandb.Table(columns=["Generated Samples", "Generated Images"], data=[[s, wandb.Image(img)] for s, img in zip(all_samples[:self.config.sampling.num_sample_log], all_images[:self.config.sampling.num_sample_log])]),
                    "trainer/global_step": self.global_step,
                }
                assert len(tot_time_per_sample) == len(tot_token_time_per_token)
                if len(tot_time_per_sample) > 1:
                    tot_time_per_sample = tot_time_per_sample[1:] # exclude warmup
                    tot_token_time_per_token = tot_token_time_per_token[1:]
                print(f'Have {len(tot_time_per_sample)} samples')
                print(f'tot_time_per_sample: {tot_time_per_sample}')
                print(f'tot_token_time_per_token: {tot_token_time_per_token}')
                avg_time_per_sample = sum(tot_time_per_sample) / len(tot_time_per_sample)
                avg_time_per_token = sum(tot_token_time_per_token) / len(tot_token_time_per_token)
                data_dict["val/avg_time_per_sample"] = avg_time_per_sample
                data_dict["val/avg_time_per_token"] = avg_time_per_token
                data_dict["val/avg_nfe_cnt"] = avg_nfe_cnt
                rprint(f"Time per sample: avg (excluding warmup): {avg_time_per_sample} - {tot_time_per_sample} ")
                rprint(f"Time per token: avg (excluding warmup): {avg_time_per_token} - {tot_token_time_per_token} ")
                with open(Path(self.config.output_dir) / "times.txt", "a") as f:
                    f.write(f"{avg_time_per_sample}, {avg_time_per_token}\n")
                    f.write(f"{tot_time_per_sample}\n")
                    f.write(f"{tot_token_time_per_token}\n")
                rprint(f"Logged time per sample and time per token to {Path(self.config.output_dir) / 'times.txt'}")
            else:
                if len(text_samples) > 0 and isinstance(text_samples[0], list):
                    text_samples = [[item] for sublist in text_samples for item in sublist]
                else:
                    text_samples = [[item] for item in text_samples]

                data_dict = {
                    "samples": wandb.Table(columns=["Generated Samples"], data=text_samples),
                    **self.get_step_metrics()
                }

            if compute_gen_ppl:
                data_dict["val/gen_ppl"] = self.gen_ppl_metric.compute()
                data_dict["val/gt_gen_ppl"] = self.gt_gen_ppl_metric.compute()
                self.gen_ppl_metric.reset()
                self.gt_gen_ppl_metric.reset()
                
            if compute_entropy:
                data_dict["val/val_entropy"] = sum(entropies) / len(entropies) if len(entropies) > 0 else 0
                
            if compute_chameleon_perplexity:
                if getattr(self.config.eval, "max_chameleon_samples", False):
                    all_images = all_images[:self.config.eval.max_chameleon_samples]
                    all_samples = all_samples[:self.config.eval.max_chameleon_samples]
                pplxs = self.calculate_chameleon_perplexity(self.chameleon_model, self.chameleon_processor, images=all_images, prompts=all_samples)
                
                # take average of pplxs
                avg_pplx = sum(pplxs) / len(pplxs)
                data_dict["val/chameleon_ppl"] = avg_pplx
                
            if self.config.eval.compute_standalone_mauve and not getattr(self.config.eval, "global_disable_mauve", False):
                all_mauve_preds = gather_object(self.mauve_predictions)
                all_mauve_refs = gather_object(self.mauve_references)
                data_dict["val/mauve_score"] = self.get_mauve_score(all_mauve_preds, all_mauve_refs, "standalone")

            log(data_dict)
    
    # Note: the above function got a little complicated due to the use in scoring/speed evals, etc. so we use the below function
    # for both unconditional *and* conditional sampling. 
    if (
        ((self.get_every_n_evals(getattr(self.config.eval, "log_every_n_evals", 10))
        and (self.image_model or self.config.trainer.multimodal_batches)
        and not getattr(self.config.model, "img_cond", False)
        and not should_eval_speed) or getattr(self.config.eval, "force_eval_uncond", False)) and not getattr(self.config.eval, "force_disable_eval_uncond", False)
    ):
        dprint("Generating samples")
        with try_except(write_error_to_file=True):
            has_label = getattr(self.config.model, "cond_label", False)
            sample_kwargs = dict()

            if has_label:
                label = torch.randint(0, self.config.model.label_vocab_size, (self.config.loader.eval_batch_size,)).to(device=self.device, dtype=torch.int64)
                sample_kwargs["label"] = label
            else:
                label = torch.randint(0, 1, (self.config.loader.eval_batch_size * 20,))

            text_samples_list = []
            img_samples_list = []
            for j in range(getattr(self.config.eval, "num_uncond_sample_batches", 1)):
                if continuous_mode:
                    data = self.sample_transfusion(batch_size_per_gpu=self.config.loader.eval_batch_size)
                    text_samples = data.xt_ids[:, self.static_txt_sl]
                    img_samples = data.xt_img_embed[:, self.static_img_sl]
                    img_samples = decode_latents(self.config, self.get_vae(), img_samples)
                else:
                    if getattr(self.config.eval, "eval_large_batch", None) is not None:
                        data = combined_batches[j]
                        data = self.update_batch(data)
                        rprint(f"Taken slice {j} of {getattr(self.config.eval, 'eval_large_batch', None)}")
                    else:
                        data = example_batch

                    _modality = data.get("modality", None)
                    _bs = min(self.config.eval.perplexity_batch_size, self.config.loader.eval_batch_size)
                    if _bs < _modality.shape[0]:
                        _modality = _modality[:_bs]

                    text_samples, img_samples = self._sample(
                        text_only=False,
                        num_steps=self.config.sampling.max_sampling_steps,
                        batch_size_per_gpu=_bs,
                        example_batch=data,
                        sample_batch_idx=j,
                        modality=_modality,
                        sample_ids=data.get("sample_ids", None),
                        allow_interleaved_conditional=True,
                        **sample_kwargs
                    )
                    num_text_tokens = self.config.model.txt_length if self.config.model.txt_length > 0 else 128
                    if text_samples is None:
                        text_samples = [torch.zeros((self.config.loader.eval_batch_size, num_text_tokens), dtype=torch.int64, device=self.device)]
                    elif isinstance(text_samples, list):
                        new_text_samples = []
                        for text_sample in text_samples:
                            text_samples_padded = torch.nn.functional.pad(text_sample, (0, num_text_tokens - text_sample.shape[-1]), value=self.tokenizer.pad_token_id) if text_sample.shape[-1] < num_text_tokens else text_sample[..., :num_text_tokens]
                            new_text_samples.append(text_samples_padded)
                        text_samples = new_text_samples
                    else:
                        text_samples = [torch.nn.functional.pad(text_samples, (0, num_text_tokens - text_samples.shape[-1]), value=self.tokenizer.pad_token_id) if text_samples.shape[-1] < num_text_tokens else text_samples[..., :num_text_tokens]]
                    
                    text_samples_list.extend(text_samples)
                    if img_samples is not None:
                        if isinstance(img_samples, list):
                            img_samples_list.extend(img_samples)
                        else:
                            img_samples_list.append(img_samples)

            if len(text_samples_list) > 0 and any(text_samples is not None for text_samples in text_samples_list):
                text_samples = torch.cat(text_samples_list, dim=0)
            else:
                text_samples = None
            has_img = any(img_samples is not None for img_samples in img_samples_list)
            log_dict = {}
            try:
                if has_img:
                    if isinstance(img_samples_list[0], Tensor):
                        img_samples = torch.cat(img_samples_list, dim=0)
                        if img_samples.ndim == 2:
                            pred_img = decode_latents(self.config, self.get_vae(), img_samples)
                        else:
                            pred_img = img_samples

                        log_dict.update({"val/gen_images": wandb.Image(pred_img)})
                    else:
                        pred_img = img_samples_list
                        for i, img in enumerate(img_samples_list):
                            log_dict[f"val/gen_images_{i}"] = wandb.Image(img)
                else:
                    pred_img = img_samples_list
            except Exception as e:
                rprint(f"Error during gather: {e}")
                pred_img = [None] * len(img_samples_list)
                has_img = False
            with try_except(write_error_to_file=True):
                if text_samples is not None:
                    text_samples = gather(text_samples)
                    pred_txt = wrapped_batch_decode(self.tokenizer, text_samples, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
                    prefix = "class_cond" if has_label else ("cond" if self.config.trainer.interleaved else "uncond")

                    if isinstance(pred_img, Tensor):
                        pred_img = pred_img.float().cpu()

                    pred_img = gather_object(pred_img)
                    gen_table = wandb.Table(columns=[*([f"{prefix}_sampled_image"] if has_img else []), f"{prefix}_sampled_caption", *(["Label"] if has_label else [])])
                    for img, caption, label in zip(pred_img, pred_txt, label):
                        gen_table.add_data(*([wandb.Image(img)] if has_img else []), caption, *([label] if has_label else []))
                    log_dict[f"{prefix}_sample_table"] = gen_table
            log({**log_dict, **self.get_step_metrics()})

    if getattr(self.config.trainer, "print_llm_loss", False) and hasattr(self, 'histogram') and not should_eval_speed:
        avg_losses = {t: sum(l) / len(l) for t, l in self.histogram.items()}
        timesteps, avg_losses = zip(*sorted(avg_losses.items()))

        from io import BytesIO

        import matplotlib.pyplot as plt

        plt.plot(timesteps, avg_losses)
        plt.xlabel('Timesteps')
        plt.ylabel('Average Loss')
        plt.title('Loss over Time')
        plt.show()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        log({"loss_over_time": wandb.Image(img)})
        rprint("Logged loss over time")

    if hasattr(self, "valid_txt_metrics"):
        self.valid_metrics.reset()
        self.valid_txt_metrics.reset()
        self.valid_img_metrics.reset()

    if (time.time() - getattr(self, "validation_start_time", time.time())) > 15:
        rprint(f"Validation took: {time.time() - self.validation_start_time} seconds")

    dprint("on_validation_epoch_end finished")

def on_validation_epoch_cleanup(self):
    self.reset_validation_metrics()
    self.fid_eval = False
    self.saved_tokens = defaultdict(list)
    if hasattr(self, "inception_metrics"): del self.inception_metrics

    if "tokens" in self.config.data.train and hasattr(self, "vae"):
            del self.vae
            self.vae = None

    if is_main_process() and not getattr(self.config.eval, "disable_fid_cleanup", False): self.cleanup_fid_output()
    empty_device_cache()

    if getattr(self.config.trainer, "attach_oom_observer_eval", False):
        if hasattr(self, "gpu_memory_reserved") and self.gpu_memory_reserved is not None:
            cur_gpu_memory_reserved = torch.cuda.memory_reserved()
            if getattr(self.config.trainer, "force_save_eval_memory_profile", False) or (cur_gpu_memory_reserved - self.gpu_memory_reserved > 4 * 1024**3):  # 4GB in bytes
                rprint(f"Warning: GPU memory usage increased by more than 4GB during validation. Initial: {self.gpu_memory_reserved / 1024**3:.2f}GB, Current: {cur_gpu_memory_reserved / 1024**3:.2f}GB")
                oom_dir = Path(self.config.output_dir) / "oom_profile"
                oom_dir.mkdir(parents=True, exist_ok=True)
                save_memory_profile(oom_dir)
            self.gpu_memory_reserved = None
        dprint("Disabled memory history")
        torch.cuda.memory._record_memory_history(enabled=None)

    dprint("on_validation_epoch_cleanup finished")

def gather_tokens(self, tokens):
    tokens = torch.cat(tokens, dim=0).to(device=self.device, dtype=torch.int64)
    tokens = gather(tokens)
    return tokens

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def get_top_k(self, batch, batch_idx):
    if batch_idx == 0:
        all_top_k = {1: [], 2: [], 5: []}
        for i in range(16):
            mod_input_ids = batch['input_ids'].clone()
            mod_input_ids[:, self.static_txt_sl] = mod_input_ids[i, self.static_txt_sl]
            mod_attention_mask = batch['attention_mask'].clone()
            mod_attention_mask[:, self.static_txt_sl] = mod_attention_mask[i, self.static_txt_sl]

            if getattr(self.config.eval, "cfg", None):
                cat_mod_input_ids = torch.cat([mod_input_ids, torch.where(batch['modality'] == 1, self.mask_index, mod_input_ids)], dim=0)
                cat_p_x0 = self.forward(
                    cat_mod_input_ids, 
                    sigma=None, 
                    attention_mask=mod_attention_mask,
                    batch=dict(modality=batch['modality']), modality=batch['modality']
                )
                logit_c, logit_u = cat_p_x0.chunk(2, dim=0)
                _w = getattr(self.config.eval, "cfg", None)
                model_output = (1 + _w) * logit_c - _w * logit_u
            else:
                model_output = self.forward(mod_input_ids, sigma=None, attention_mask=mod_attention_mask, batch=dict(modality=batch['modality']), modality=batch['modality'])
            
            log_p_theta = torch.gather(input=model_output, dim=-1, index=mod_input_ids[:, 1:, None]).squeeze(-1)
            mean_nll = (-log_p_theta * mod_attention_mask[:, 1:]).sum(dim=-1) / mod_attention_mask[:, 1:].sum(dim=-1)
            
            for k in [1, 2, 5]:
                topk_values, topk_indices = torch.topk(mean_nll, k, dim=0)
                all_top_k[k].append(0 in topk_indices.tolist())

        for k in [1, 2, 5]:
            retrieval_rate = sum(all_top_k[k]) / len(all_top_k[k])
            rprint(f"{retrieval_rate:.2%} retrieved in top {k}")
            log({f"val/top_{k}": retrieval_rate})

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def compute_clip_score(self, output_dir, prefix):
    from model_utils import calculate_clip_score
    caption_paths = [str(x.as_posix()) for x in Path(output_dir).glob('*.png') if x.is_file() and x.with_suffix('.json').exists()]
    captions_mapping = {str(x): json.load(Path(x).with_suffix('.json').open())['caption'] for x in caption_paths}
    clip_score = calculate_clip_score(caption_paths, captions_mapping=captions_mapping)
    clip_score *= 100 # For some reason people scale cosine sim
    rprint(f"{prefix} CLIP score: {clip_score}")
    log({f"val/{prefix}_clip_score": clip_score, **self.get_step_metrics()})

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def compute_inline_fid(self):
    rprint(f"FID Eval. We have {len(self.inception_metrics.fake_uncond_features)} batches.")
    try:
        if self.config.mode == "eval" and not self.config.trainer.image_mode == "continuous":
            output_dir = Path("eval_tokens").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            dataset_size = sum(x[-1].shape[0] for x in self.computed_tokens)
            data = TensorDict(
                {
                    "txt_input_ids": torch.cat([x[1] for x in self.computed_tokens]).to(device="cpu", dtype=torch.int32),
                    "img_input_ids": torch.cat([x[2] for x in self.computed_tokens]).to(device="cpu", dtype=torch.int16),
                    "gt_img_input_ids": torch.cat([x[3] for x in self.computed_tokens]).to(device="cpu", dtype=torch.int16),
                },
                batch_size=[dataset_size],
            )
            save_loc = str(output_dir / f"{get_rank()}")
            data.memmap(save_loc)
            gprint(f"Saved tokens to {save_loc}")

            rank = get_rank()
            output_folder = Path("fid_metrics")
            output_folder.mkdir(parents=True, exist_ok=True)
            torch.save(self.inception_metrics.fake_uncond_features, output_folder / f"rank_{rank}_fake_uncond_features.pt")
            torch.save(self.inception_metrics.fake_uncond_logits, output_folder / f"rank_{rank}_fake_uncond_logits.pt")
            torch.save(self.inception_metrics.real_features, output_folder / f"rank_{rank}_real_features.pt")
            rprint(f"Saved rank_{rank} tensors.")
    except Exception as e:
        gprint(f"Error during all_gather_object or saving tensors: {e}")

    with torch.autocast(device_type=self.device.type, enabled=False):
        metrics = self.inception_metrics.compute() # Gather is done internally

    rprint(f"Computed metrics: {metrics}")
    metrics = {f"val/{k}": v for k, v in metrics.items()}
    log({**metrics, "trainer/global_step": self.global_step})
    output_folder = Path("fid_metrics")
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / f'metrics_{get_rank()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
        for k, v in metrics.items():
            f.write(f"val/{k}: {v}\n")

    self.fid_eval = False
    del self.inception_metrics
    rprint("Finished FID eval")

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def compute_clean_fid_eval(self):
    with try_except(write_error_to_file=True):
        images = []
        for i, filename in enumerate(sorted(Path(self.fid_gen_dir).iterdir(), key=lambda x: random.random())):
            if i >= self.config.loader.eval_batch_size * get_world_size():
                break
            if filename.is_file() and filename.suffix == ".png":
                for i in range(3):
                    try:
                        img = Image.open(filename)
                    except Exception as e:
                        time.sleep(0.1)
                        rprint(f"Error opening image {filename}: {e}")
                images.append(np.array(img))
        images = np.stack(images)
        log({"val/fid_gen_img_at_compute": wandb.Image(Im(images).torch)})

    from cleanfid import fid
    kwargs = dict()
    if self.config.eval.clean_fid_use_precomputed_stats:
        kwargs.update(dict(
            dataset_name=self.config.eval.clean_fid_precomputed_name,
            dataset_res=self.config.eval.clean_fid_precomputed_res,
            dataset_split=self.config.eval.clean_fid_precomputed_split,
        ))
    else:
        kwargs.update(dict(fdir2=str(self.fid_gt_dir)))
    
    score = fid.compute_fid(
        fdir1=str(self.fid_gen_dir),
        use_dataparallel=False,
        **kwargs
    )

    rprint(f"FID score: {score}")
    metrics = {"val/fid_unconditional": score, **self.get_step_metrics()}
    log(metrics)

    metrics = {f"val/{k}": v for k, v in metrics.items()}
    output_folder = Path("fid_metrics")
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / f'metrics_{get_rank()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    self.fid_eval = False
    
def sample_for_fid(self, batch, batch_idx, return_gt_img=False, return_gt_txt=False, img_to_txt_gen=False):
    """This function is also used for img -> txt generation."""
    continuous_mode = self.config.trainer.image_mode == "continuous"
    sample_kwargs = self.get_cond_dict(batch)
    orig_modality, orig_input_ids = None, None
    if img_to_txt_gen:
        if self.config.parameterization == "ar":
            txt_first_sl = slice(None, self.config.model.txt_length)
            img_first_sl = slice(None, self.config.model.img_length)
            if (batch["modality"][:, txt_first_sl] == 0).all(): # Flip [txt, img] -> [img, txt]
                assert (batch["modality"][:, :self.config.model.txt_length] == 0).all() and (batch["modality"][:, self.config.model.txt_length:] == 1).all()
                flipped_batch = dict()
                img_slice = slice(-self.config.model.img_length, None)
                txt_slice = slice(None, self.config.model.txt_length)
                for key in ["modality", "attention_mask", "input_ids"]:
                    flipped_batch[key] = torch.cat([batch[key][:, img_slice], batch[key][:, txt_slice]], dim=1)

                batch = flipped_batch
            else:
                assert (batch["modality"][:, img_first_sl] == 1).all() # We already have [img, txt]

            assert (batch["modality"][:, :self.config.model.img_length] == 1).all(), "Img tokens should be 0"
        else:
            assert (batch["modality"][:, :self.config.model.txt_length] == 0).all() # We already have [txt, img]

        sample_kwargs["sample_modality"] = batch["modality"]
        _x0_unmask = (batch["modality"] == 1)
    elif getattr(self.config.eval, "unconditional_fid", False):
        sample_kwargs["x0_unmask"] = None
        sample_kwargs["x0"] = None
        sample_kwargs["sample_modality"] = batch["modality"]
    elif self.config.trainer.ar_inpainting:
        assert getattr(self.config.eval, "txt_conditional_fid", False)
        min_val, max_val = getattr(self.config.eval, "ar_inpainting_min_val", 0.9), getattr(self.config.eval, "ar_inpainting_max_val", 1.0)
        n = batch["modality"].shape[0]
        _eps_t = torch.rand(n, device=self.device)
        t = (max_val - min_val) * _eps_t + min_val
        if getattr(self.config.eval, "ar_inpainting_force_val", None) is not None:
            t = torch.full_like(t, getattr(self.config.eval, "ar_inpainting_force_val"), dtype=t.dtype, device=t.device)
        if self.config.parameterization == "ar":
            orig_modality, orig_input_ids = batch["modality"].clone(), batch["input_ids"].clone()
            del batch["batch_contains_img"]
            batch.auto_batch_size_()
            batch = torch.cat([batch, batch], dim=1)
            x0 = batch["input_ids"]
            move_indices = torch.rand(*x0.shape, device=x0.device) < t[:, None] # Unmask so we switch sign compared to move_indices
            move_indices[:, x0.shape[1] // 2:] = False
            batch["input_ids"] = torch.where(move_indices, self.mask_index, x0)
            _x0_unmask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
            _x0_unmask[:, :batch["input_ids"].shape[1] // 2] = True
        else:
            _x0_unmask = torch.rand(*batch["modality"].shape, device=batch["modality"].device) > t[:, None] # Unmask so we switch sign compared to move_indices
        sample_kwargs["sample_modality"] = batch["modality"]
        sample_kwargs["x0_unmask"] = _x0_unmask
        sample_kwargs["x0"] = batch["input_ids"]
    elif getattr(self.config.eval, "class_conditional_fid", False) or getattr(self.config.eval, "txt_conditional_fid", False):
        sample_kwargs["x0"] = batch["input_ids"]
        if getattr(self.config.eval, "class_conditional_fid", False):
            sample_kwargs["sample_modality"] = torch.full_like(batch["modality"], 1)
            sample_kwargs["sample_modality"][:, 0] = 0
            _x0_unmask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
            _x0_unmask[..., 0] = True
        elif getattr(self.config.eval, "txt_conditional_fid", False):
            assert ((batch["modality"] == 1).sum(dim=-1) > 0).all(), "No img samples provided"
            sample_kwargs["sample_modality"] = batch["modality"]
            _x0_unmask = (batch["modality"] == 0)
        sample_kwargs["x0_unmask"] = _x0_unmask

    if continuous_mode:
        data = self.sample_transfusion(batch_size_per_gpu=self.config.loader.eval_batch_size)
        gen_txt_tokens = data.xt_ids[:, self.static_txt_sl]
        gen_img_tokens = data.xt_img_embed[:, self.static_img_sl]
        gen_img = decode_latents(self.config, self.get_vae(), gen_img_tokens)
    else:
        gen_txt_tokens, gen_img_tokens = self._sample(text_only=False, **sample_kwargs)
        gen_img = decode_latents(self.config, self.get_vae(), gen_img_tokens)

    fid_rec_img, gt_img_tokens, gt_txt_tokens = None, None, None
    if return_gt_img:
        if "img" in batch:
            fid_rec_img = batch["img"]
        else:
            if orig_modality is None:
                orig_modality = batch.get("modality", None)
            if orig_input_ids is None:
                orig_input_ids = batch["input_ids"]

            _, gt_img_tokens = self.decode_batch(orig_input_ids, text_only=False, sample_modality=orig_modality)
            if gt_img_tokens.shape[0] == 0:
                rprint(f"{gt_img_tokens.shape} {batch['input_ids'].shape}")
            fid_rec_img = decode_latents(self.config, self.get_vae(), gt_img_tokens)

    if return_gt_txt:
        if orig_input_ids is None:
            orig_input_ids = batch["input_ids"]
        if orig_modality is None:
            orig_modality = batch.get("modality", None)
        gt_txt_tokens, _ = self.decode_batch(orig_input_ids, text_only=False, sample_modality=orig_modality)

    _prefix = "img_to_txt" if img_to_txt_gen else ("unconditional" if getattr(self.config.eval, "unconditional_fid", False) else "txt_to_img")
    self.saved_tokens[_prefix + "_gen_img_tokens"].append(gen_img_tokens.detach().cpu().to(torch.int32))
    self.saved_tokens[_prefix + "_gen_txt_tokens"].append(gen_txt_tokens.detach().cpu().to(torch.int32))
    if gt_img_tokens is not None: self.saved_tokens[_prefix + "_gt_img_tokens"].append(gt_img_tokens.detach().cpu().to(torch.int32))
    if gt_txt_tokens is not None: self.saved_tokens[_prefix + "_gt_txt_tokens"].append(gt_txt_tokens.detach().cpu().to(torch.int32))

    return gen_img, gen_txt_tokens, gt_img_tokens, gt_txt_tokens, gen_img_tokens, fid_rec_img


def update_inline_fid(self, batch, batch_idx):
    gen_img, txt_tokens, gt_img_tokens, gt_txt_tokens, gen_img_tokens, fid_rec_img = self.sample_for_fid(batch, batch_idx, return_gt_img=True, return_gt_txt=True)

    if self.config.mode == "eval":
        self.computed_tokens.append((txt_tokens, gen_img_tokens, gt_img_tokens))
    with torch.autocast(device_type=self.device.type, enabled=False):
        self.inception_metrics.update(remap_image_torch(fid_rec_img).to(self.device), None, image_type="real")
        self.inception_metrics.update(remap_image_torch(gen_img).to(self.device), None, image_type="unconditional")
    
    if batch_idx == 0:
        log({"val/fid_gen": wandb.Image(gen_img), "val/fid_gt": wandb.Image(fid_rec_img), **self.get_step_metrics()})

    if batch_idx > 0 and batch_idx % 5 == 0 and self.config.mode == "eval":
        gprint(f"Saving rank_{get_rank()} tensors.")
        try:
            rank = get_rank()
            torch.save(self.inception_metrics.fake_uncond_features, f"{batch_idx}_rank_{rank}_fake_uncond_features.pt")
            torch.save(self.inception_metrics.fake_uncond_logits, f"{batch_idx}_rank_{rank}_fake_uncond_logits.pt")
            torch.save(self.inception_metrics.real_features, f"{batch_idx}_rank_{rank}_real_features.pt")
            gprint(f"Saved rank_{rank} tensors.")
        except Exception as e:
            gprint(f"Error during all_gather_object or saving tensors: {e}")

def update_clean_fid(self, batch, batch_idx):
    assert hasattr(self, "fid_gen_dir")
    save_gt_img = not self.config.eval.clean_fid_use_precomputed_stats
    gen_img, txt_tokens, gt_img_tokens, gt_txt_tokens, img_samples, fid_rec_img = self.sample_for_fid(batch, batch_idx, return_gt_img=save_gt_img, return_gt_txt=True)

    if self.config.model.image_model_fid_eval:
        txt_samples = wrapped_batch_decode(self.tokenizer, txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        gt_txt_samples = wrapped_batch_decode(self.tokenizer, gt_txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    save_loc = Path(self.fid_gen_dir)
    save_loc.mkdir(parents=True, exist_ok=True)
    quantized_img = remap_image_torch(gen_img).permute(0, 2, 3, 1).cpu().numpy()

    if save_gt_img:
        gt_quantized_img = remap_image_torch(fid_rec_img).permute(0, 2, 3, 1).cpu().numpy()
        save_loc_gt = Path(self.fid_gt_dir)
        save_loc_gt.mkdir(parents=True, exist_ok=True)
        
    for i in range(gen_img.shape[0]):
        gen_img_pil = Image.fromarray(quantized_img[i])
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        filename = f"{batch_idx}_{get_rank()}_{i}_{suffix}.png"
        out_file_path = save_loc / filename
        gen_img_pil.save(out_file_path)

        if self.config.eval.txt_conditional_fid:
            with open(out_file_path.with_suffix(".json"), 'w') as json_file:
                json.dump({"caption": txt_samples[i]}, json_file)

        if save_gt_img:
            gt_img_pil = Image.fromarray(gt_quantized_img[i])
            gt_out_file_path = save_loc_gt / filename
            gt_img_pil.save(gt_out_file_path)

            if self.config.eval.txt_conditional_fid:
                with open(gt_out_file_path.with_suffix(".json"), 'w') as json_file:
                    json.dump({"caption": gt_txt_samples[i]}, json_file)

    if batch_idx == 0:
        rprint(f"Logging at batch idx {batch_idx}")
        time.sleep(0.2)
        with try_except(write_error_to_file=True):
            images = []
            for i, filename in enumerate(sorted(Path(self.fid_gen_dir).iterdir(), key=lambda x: random.random())):
                if i >= self.config.loader.eval_batch_size * get_world_size():
                    break
                if filename.is_file() and filename.suffix == ".png":
                    img = Image.open(filename)
                    images.append(np.array(img))
            images = np.stack(images)
            log({"val/fid_gen_img": wandb.Image(Im(images).torch)})
            rprint(f"FID Txt: {txt_samples[0]}")

def update_img_to_txt_mauve_clip(self, batch, batch_idx):
    assert hasattr(self, "img_to_txt_mauve_gen_dir")
    save_gt_img = True
    empty_device_cache()
    gen_img, gen_txt_tokens, gt_img_tokens, gt_txt_tokens, gen_img_tokens, fid_rec_img = self.sample_for_fid(batch, batch_idx, return_gt_img=save_gt_img, return_gt_txt=True, img_to_txt_gen=True)

    gen_txt_samples = wrapped_batch_decode(self.tokenizer, gen_txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    gt_txt_samples = wrapped_batch_decode(self.tokenizer, gt_txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    save_loc = Path(self.img_to_txt_mauve_gen_dir)
    save_loc.mkdir(parents=True, exist_ok=True)
    quantized_img = remap_image_torch(gen_img).permute(0, 2, 3, 1).cpu().numpy()

    if save_gt_img:
        gt_quantized_img = remap_image_torch(fid_rec_img).permute(0, 2, 3, 1).cpu().numpy()
        save_loc_gt = Path(self.img_to_txt_mauve_gt_dir)
        save_loc_gt.mkdir(parents=True, exist_ok=True)
        
    for i in range(gen_img.shape[0]):
        gen_img_pil = Image.fromarray(quantized_img[i])
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        filename = f"{batch_idx}_{get_rank()}_{i}_{suffix}.png"
        out_file_path = save_loc / filename
        gen_img_pil.save(out_file_path)
        with open(out_file_path.with_suffix(".json"), 'w') as json_file:
            json.dump({"caption": gen_txt_samples[i]}, json_file)

        if save_gt_img:
            gt_img_pil = Image.fromarray(gt_quantized_img[i])
            gt_out_file_path = save_loc_gt / filename
            gt_img_pil.save(gt_out_file_path)
            with open(gt_out_file_path.with_suffix(".json"), 'w') as json_file:
                json.dump({"caption": gt_txt_samples[i]}, json_file)

    if batch_idx == 0:
        rprint(f"GT img -> txt mauve: {gt_txt_samples[0]}")
        rprint(f"Gen img -> txt mauve: {gen_txt_samples[0]}")
        
def compute_mauve_entropy(self, img_to_txt_mauve_gen_dir, img_to_txt_mauve_gt_dir, gen_txt_tokens, gt_txt_tokens, prefix):
    gt_txt = []
    gt_img = []
    gt_dir = Path(img_to_txt_mauve_gt_dir)
    gen_dir = Path(img_to_txt_mauve_gen_dir)
    stems = [f.stem for f in gt_dir.iterdir() if f.suffix == '.json' and (gen_dir / f.name.replace("gt", "gen")).exists()]
    assert len(stems) > 0, f"No stems found in {gt_dir} and {gen_dir}"
    rprint(f"Found {len(stems)} unique stems")

    gt_img = []
    gt_txt = []
    gen_txt = []
    gen_img = []
    data_dict = {}
    for stem in stems:
        gt_img_path = gt_dir / f"{stem}.png"
        gt_img.append(Image.open(gt_img_path))

        gen_img_path = gen_dir / f"{stem}.png"
        gen_img.append(Image.open(gen_img_path))

        with open(gt_dir / f"{stem}.json", 'r') as f:
            gt_txt.append(json.load(f)["caption"])

        with open(gen_dir / f"{stem}.json", 'r') as f:
            gen_txt.append(json.load(f)["caption"])

    table = wandb.Table(columns=["GT Image", "GT Text", "Generated Image", "Generated Text"])
    num_samples_to_display = min(20, len(stems))
    for i in range(num_samples_to_display):
        table.add_data(
            wandb.Image(gt_img[i]),
            gt_txt[i],
            wandb.Image(gen_img[i]),
            gen_txt[i]
        )
        
    data_dict[f"val/{prefix}_mauve_samples"] = table
    if not getattr(self.config.eval, "global_disable_mauve", False):
        data_dict[f"val/{prefix}_mauve_score"] = self.get_mauve_score(gen_txt, gt_txt, prefix)
    data_dict[f"val/{prefix}_gt_entropy"] = self.compute_entropy(gt_txt_tokens)
    data_dict[f"val/{prefix}_gen_entropy"] = self.compute_entropy(gen_txt_tokens)
    data_dict[f"val/{prefix}_percent_valid_txt_tokens"] = self.count_valid_tokens(gen_txt_tokens).float().mean(dim=-1) / gen_txt_tokens.shape[-1]
    log({**data_dict, **self.get_step_metrics()})

def count_valid_tokens(self, text_tokens):
    after_first_eos = torch.cumsum(text_tokens == self.tokenizer.eos_token_id, dim=1).bool()
    after_first_eos_mask = after_first_eos.cumsum(dim=1) > 1
    return ~after_first_eos_mask

def get_valid_seq(self, text_tokens):
    if self.tokenizer.bos_token_id == self.tokenizer.eos_token_id:
        assert False, "BOS and EOS are the same."
    
    eos_positions = (text_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        return text_tokens[..., :eos_positions[0] + 1]
    else:
        return text_tokens

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def compute_entropy(self, text_tokens):
    """Compute the entropy of the generated text.
    Definition Pg 33 of https://arxiv.org/pdf/2409.02908

    Args:
        text_tokens: Tensor of generated text tokens. (B, L)
    Returns:
        Entropy of the generated text.
    """
    val_entropy = Entropy(sync_on_compute=False).to(self.device)
    B, L = text_tokens.shape
    K = self.tokenizer.vocab_size  # Use the actual vocabulary size
    
    # Compute entropy for each sequence in the batch
    entropies = []
    for seq in text_tokens:
        seq_length = seq.numel()
        token_frequencies = torch.bincount(self.get_valid_seq(seq), minlength=K)
        p_k = token_frequencies.float() / seq_length
        p_k = p_k.to(self.device)
        nll = -torch.sum(p_k * torch.log(p_k + 1e-10))
        entropies.append(nll)
    
    # Calculate the average entropy across the batch
    avg_entropy = torch.mean(torch.tensor(entropies))
    
    # Update the validation entropy metric
    val_entropy.update(avg_entropy, weight=B)
    return val_entropy.compute()

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def get_mauve_score(self, pred, gt, prefix):
    from evaluate import load 
    mauve = load('mauve')
    
    # We require a list of strings for pred, gt
    mauve_metric = MauveScore(sync_on_compute=False).to(self.device)
    rprint(f"Generated {len(pred)} MAUVE predictions")
    assert len(pred) >= self.config.eval.mauve_num_samples
    rprint(f'Before removing duplicates: {len(pred)}')
    pred_text = list(set(pred))
    rprint(f'After removing duplicates: {len(pred_text)}')
    ref_text = list(set(gt))
    store_path = os.path.join(self.config.output_dir, f"{prefix}_mauve_predictions.pkl")
    with open(store_path, "wb") as f:
        pickle.dump(pred_text, f)
        
    rprint(f"Stored {len(pred_text)} unique MAUVE predictions to {store_path}")
    
    min_len = min(len(pred_text), len(ref_text))
    pred_text = pred_text[:min_len]
    ref_text = ref_text[:min_len]
    
    rprint(f"Computing img to txt MAUVE score for {len(pred_text)} unique predictions and {len(ref_text)} references")
    
    # compute mauve score
    device_id = 0 # this is main process
    mauve_divergence_curve_discretization_size = self.config.eval.mauve_divergence_curve_discretization_size
    mauve_scaling_factor = self.config.eval.mauve_scaling_factor
    avg_over_seed = self.config.eval.mauve_average_over_seeds
    
    # generate avg_over_seed number of seeds randomly
    random_seeds = [random.randint(0, 100000) for _ in range(avg_over_seed)]
    for seed in random_seeds:
        mauve_score = mauve.compute(
            references=ref_text,
            predictions=pred_text,
            device_id=device_id,
            divergence_curve_discretization_size=mauve_divergence_curve_discretization_size, 
            mauve_scaling_factor=mauve_scaling_factor
        )
        mauve_metric.update(mauve_score.mauve)
        rprint(f"MAUVE score for seed {seed}: {mauve_score.mauve}")
        store_path = os.path.join(self.config.output_dir, f"{prefix}_mauve_score_seed_{seed}.txt")
        with open(store_path, "w") as f:
            f.write(str(mauve_score))
            
        rprint(f"Stored MAUVE score for seed {seed} to {store_path}")
        
    avg_mauve_score = mauve_metric.compute()
    return avg_mauve_score


def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

def get_cfg_weight(self, t):
    _cfg = self.config.eval.cfg
    if not getattr(self.config.eval, "force_cfg_value", False):
        if _cfg == -1:
            _cfg = torch.linspace(0, 10, t.shape[0]).to(t.device)

        if getattr(self.config.eval, "cfg_min_timestep", None) is not None and getattr(self.config.eval, "cfg_max_timestep", None) is not None:
            _w = (_cfg * ((t - getattr(self.config.eval, "cfg_max_timestep")) / (getattr(self.config.eval, "cfg_min_timestep") - getattr(self.config.eval, "cfg_max_timestep"))))[:, None]
        else:
            _w = (_cfg * (1 - t))[:, None]
    else:
        _w = _cfg

    if getattr(self.config.eval, "cfg_min_timestep", None) is not None:
        _w = torch.where(t > getattr(self.config.eval, "cfg_min_timestep", None), _w, torch.tensor(0.0))
        
    if getattr(self.config.eval, "cfg_max_timestep", None) is not None:
        _w = torch.where(t < getattr(self.config.eval, "cfg_max_timestep", None), _w, torch.tensor(0.0))

    if not isinstance(_w, torch.Tensor):
        _w = torch.tensor(_w)

    return _w

def _ddpm_forward(self, x, t, sigma_t, x0=None, x0_unmask=None, force_cfg=None, **kwargs):    
    _w = None
    if getattr(self.config.eval, "cfg", None) is not None and x0_unmask is not None and x0_unmask.sum() > 0:
        _w = self.get_cfg_weight(t)
        
    orig_modality, orig_sample_ids = None, None
    if _w is not None and (_w > 0).any():
        x_uncond = x.clone()
        x_uncond[x0_unmask] = self.mask_index
        if getattr(self.config.eval, "split_cfg_batches", False):
            cat_p_x0 = torch.cat([
                self.forward(
                    x=x,
                    sigma=sigma_t,
                    return_logits=True,
                    **kwargs
                ),
                self.forward(
                    x=x_uncond,
                    sigma=sigma_t,
                    return_logits=True,
                    **kwargs
                )
            ], dim=0)
        else:
            orig_modality = kwargs.get("modality", None)
            if orig_modality is not None:
                orig_modality = orig_modality.clone()
                kwargs["modality"] = torch.cat([orig_modality, orig_modality], dim=0)

            orig_sample_ids = kwargs.get("sample_ids", None)
            if orig_sample_ids is not None:
                orig_sample_ids = orig_sample_ids.clone()
                kwargs["sample_ids"] = torch.cat([orig_sample_ids, orig_sample_ids], dim=0)

            if self.config.trainer.interleaved_training_flex_attention:
                assert 'sample_ids' in kwargs
                kwargs['block_mask'] = get_interleaved_block_mask(kwargs['sample_ids'], x.shape[0], x.shape[-1], self.device)

            cat_p_x0 = self.forward(
                x=torch.cat([x, x_uncond], dim=0),
                sigma=torch.cat([sigma_t, sigma_t], dim=0) if sigma_t is not None else None,
                return_logits=True,
                **kwargs
            )
            kwargs["modality"] = orig_modality
            kwargs["sample_ids"] = orig_sample_ids

        logit_c, logit_u = cat_p_x0.chunk(2, dim=0)
        if isinstance(_w, torch.Tensor) and _w.ndim == 2 and logit_c.ndim == 3:
            _w = _w.unsqueeze(-1)
        output_logits = (1 + _w) * logit_c - _w * logit_u
        _modality = kwargs.get("modality", None)
        if self.config.trainer.ar_shift:
            _modality = _modality[:, 1:]
        
        p_x0 = self._subs_parameterization(output_logits, xt=None, batch=None, modality=_modality)
        p_x0 = p_x0.exp()
        del logit_c, logit_u, cat_p_x0, output_logits, orig_modality, orig_sample_ids, x, x_uncond
    else:
        p_x0 = self.forward(x=x, sigma=sigma_t, **kwargs)
        p_x0 = p_x0.exp()

    if self.config.trainer.force_bf16_eval:
        p_x0 = p_x0.to(torch.bfloat16)

    kwargs.pop("attention_caching", None)
    kwargs.pop("block_mask", None)

    if getattr(self.config.eval, "force_empty_cache", False):
        empty_device_cache()

    return p_x0


def sample_masking(self, batch, batch_idx):
    assert (self.config.loader.batch_size == self.config.loader.eval_batch_size) or self.config.mode == 'eval' # need for modality otherwise x and modality have different batch sizes
    if getattr(self.config.model, "img_cond", False):
        text_samples, img_samples = self._sample(text_only=False, **self.get_cond_dict(batch))
        pred_img = decode_latents(self.config, self.get_vae(), img_samples)
        log({"val/gen_images_": wandb.Image(pred_img), "trainer/global_step": self.global_step})

    orig_bs = batch["input_ids"].shape[0]
    bs = min(10, max(1, int(orig_bs // 2)))
    bs = getattr(self.config.eval, "masking_batch_size", bs)
    bs = min(bs, orig_bs)
    
    if getattr(self.config.eval, "num_random_masking", None) is not None:
        num_random_masking = getattr(self.config.eval, "num_random_masking", 1)
        bs = max(bs, num_random_masking)
    else:
        num_random_masking = max((x0.shape[0] + 1) // 4, 1)
    
    _attention_mask = (batch["attention_mask"] if "attention_mask" in batch else None)[:bs]
    _input_ids = (batch["input_ids"])[:bs]
    _x_modality = (batch["modality"])[:bs] if "modality" in batch else None

    if _x_modality.shape[0] != bs:
        _x_modality = _x_modality[[0]].repeat(bs, 1)

    (input_tokens, output_tokens, _attention_mask) = self._maybe_sub_sample(_input_ids, _attention_mask)
    x0 = input_tokens
    forward_kwargs = self.get_cond_dict(batch)
    forward_kwargs['is_sample_masking'] = True
    
    if "x_cond" in forward_kwargs:
        forward_kwargs["x_cond"] = forward_kwargs["x_cond"][:bs]

    assert output_tokens is None
    assert self.T == 0 and self.change_of_variables is False

    random_masking_ratio = getattr(self.config.eval, "random_masking_ratio", 0.95)
    t = random_masking_ratio + (1 - random_masking_ratio) * torch.rand(num_random_masking, device=x0.device)
    sigma, dsigma = self.noise(t)
    unet_conditioning = sigma[:, None]
    move_chance = 1 - torch.exp(-sigma[:, None])

    unet_conditioning = torch.cat([unet_conditioning, unet_conditioning.new_full((bs - num_random_masking, 1), torch.nan)], dim=0)
    move_chance = torch.cat([move_chance, move_chance.new_full((bs - num_random_masking, move_chance.shape[1]), 1)], dim=0)

    uniform_mask = torch.full(x0.shape, True, device=x0.device, dtype=torch.bool)
    text_only_mask = uniform_mask.clone()
    text_only_mask = torch.where(_x_modality == 1, False, text_only_mask)

    image_only_mask = uniform_mask.clone()
    image_only_mask = torch.where(_x_modality == 0, False, image_only_mask)
    image_only_mask = torch.where(batch["batch_contains_img"][:bs, None], image_only_mask, True)
    mask_dict = dict(mask_all=uniform_mask, mask_text_only=text_only_mask, mask_image_only=image_only_mask)

    if getattr(self.config.eval, "mask_img_only", False):
        uniform_mask = torch.full(x0.shape, True, device=x0.device, dtype=torch.bool)
        image_only_mask = torch.where(_x_modality == 0, False, uniform_mask)
        move_chance = torch.ones_like(move_chance)
        mask_dict = dict(mask_image_only=image_only_mask)
    elif getattr(self.config.eval, "mask_img_only_keep_partial", False):
        mask_dict = dict(mask_image_only=image_only_mask)
    elif getattr(self.config.eval, "mask_all_only", False):
        mask_dict = dict(mask_all=uniform_mask)

    only_uniform_mask = getattr(self.config.eval, "only_uniform_mask", False)

    table_dict = dict()
    for mask_name, allow_move_mask in mask_dict.items():
        if mask_name == "mask_all" and not only_uniform_mask:
            _move_chance = 0.5 + (1 - 0.5) * torch.rand_like(move_chance)
        elif mask_name == "mask_text_only":
            _move_chance = torch.zeros_like(move_chance)
        else:
            _move_chance = move_chance

        xt = self.q_xt(
            x0,
            _move_chance,
            allow_move_mask,
            mask_image_square=(mask_name != "mask_text_only") and not only_uniform_mask,
            mask_text_region=(mask_name != 'mask_image_only') and not only_uniform_mask
        )

        if getattr(self.config.eval, "single_step_denoising", False):
            forward_kwargs.pop("is_sample_masking", None)
            model_output = self.forward(xt, unet_conditioning, **forward_kwargs)
            if not self.is_compiled:
                utils.print_nans(model_output, "model_output")
            model_output = model_output.exp()
            pred_tokens = model_output.argmax(dim=-1)
            pred_tokens = torch.where(xt == self.mask_index, pred_tokens, xt)
            pred_text, pred_img = self.decode_batch(pred_tokens, text_only=False, sample_modality=_x_modality)
            pred_img = decode_latents(self.config, self.get_vae(), pred_img)
            pred_txt = wrapped_batch_decode(self.tokenizer, pred_text, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        else:
            xt_unmasked = xt != self.mask_index
            pred_txt, pred_img = self.sample(x0=xt, x0_unmask=xt_unmasked, sample_modality=_x_modality, **forward_kwargs)

        gen_table = wandb.Table(columns=["GT Img", "GT Caption", "Masked Img", "Masked Caption", "Pred Img", "Pred Caption", "Move chance"])
        masked_txt, masked_img, mask_text_mask, mask_img_mask = self.decode_batch(
            xt, text_only=False, return_masks=True, allow_mask_index=True, sample_modality=_x_modality
        )

        downscale_ratio = self.config.model.downscale_ratio
        latent_dim = self.config.data.resolution // downscale_ratio

        img_mask = einops.repeat(
            einops.rearrange(mask_img_mask[:, self.static_img_sl], "b (h w) -> b h w", h=latent_dim, w=latent_dim),
            "b h w -> b (h na) (w nb)",
            na=downscale_ratio,
            nb=downscale_ratio,
        )

        gt_txt, gt_img = self.decode_batch(_input_ids, text_only=False, sample_modality=_x_modality)
        gt_txt = wrapped_batch_decode(self.tokenizer, gt_txt, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        gt_img = decode_latents(self.config, self.get_vae(), gt_img)

        masked_txt = wrapped_batch_decode(self.tokenizer, masked_txt, clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=self.config.data.disable_mask_after_eos)
        masked_img = gt_img.clone().permute(0, 2, 3, 1)
        masked_img[img_mask] = torch.tensor([0.5, 0.5, 0.5], dtype=masked_img.dtype, device=masked_img.device)
        masked_img = masked_img.permute(0, 3, 1, 2)
        for _gt_img, _gt_txt, _masked_img, _masked_txt, _pred_img, _pred_txt, _move_chance in zip(
            gt_img, gt_txt, masked_img, masked_txt, pred_img, pred_txt, move_chance
        ):
            gen_table.add_data(
                wandb.Image(_gt_img), _gt_txt, wandb.Image(_masked_img), _masked_txt, wandb.Image(_pred_img), _pred_txt, _move_chance
            )

        table_suffix = f"_{batch_idx}"
        table_dict[f"{mask_name}_sample_table{table_suffix}"] = gen_table

    log({**table_dict, "trainer/global_step": self.global_step})

def log_flops(self, batch, batch_idx):
    use_torch_tnt = False
    use_native_torch = True
    use_fvcore = False
    with torch.enable_grad():
        with torch.autocast(self.device.type, dtype=self.dtype):
            new_batch_idxs = batch["input_ids"].new_ones((self.config.loader.batch_size, self.config.model.length))
            if use_fvcore:
                # Broken due to some issue with triton
                from fvcore.nn import (ActivationCountAnalysis,
                                       FlopCountAnalysis, flop_count_str,
                                       flop_count_table)
                example_input = (new_batch_idxs, None)
                fca = FlopCountAnalysis(self.accelerator.unwrap_model(self.backbone), example_input)
                aca = ActivationCountAnalysis(self.accelerator.unwrap_model(self.backbone), example_input)
                print(flop_count_table(fca, max_depth=1))
                print(flop_count_str(fca))
                print(fca.total())

            if use_torch_tnt:
                from torchtnt.utils.module_summary import get_module_summary
                module_summary = get_module_summary(self.backbone, module_args=(new_batch_idxs, None), module_kwargs={})
                rprint(module_summary)
                rprint(f"TorchTNT Forward FLOPs: {module_summary.flops_forward / 1e12:.2f} FLOPs")
                rprint(f"TorchTNT Backward FLOPs: {module_summary.flops_backward / 1e12:.2f} FLOPs")
                rprint(f"TorchTNT Total FLOPs: {(module_summary.flops_forward + module_summary.flops_backward) / 1e12:.2f} FLOPs")

            if use_native_torch:
                from torch.utils.flop_counter import FlopCounterMode
                flop_counter = FlopCounterMode(self.backbone, display=True, depth=3)
                with flop_counter:
                    fake_batch = {}
                    fake_batch["input_ids"] = new_batch_idxs
                    fake_batch['attention_mask'] = batch['attention_mask'].new_ones(new_batch_idxs.shape)
                    if 'modality' in batch:
                        fake_batch['modality'] = batch['modality'].new_ones(new_batch_idxs.shape)
                    fake_batch['x0'] = fake_batch["input_ids"]
                    t = self._sample_t(fake_batch['x0'].shape[0], fake_batch['x0'].device)
                    sigma, dsigma = self.noise(t)
                    move_chance = 1 - torch.exp(-sigma[:, None])
                    xt = self.q_xt(fake_batch['x0'], move_chance)
                    fake_batch['xt'] = xt
                    if self.config.trainer.image_mode == "continuous":
                        B, T = fake_batch["input_ids"].shape
                        indices = fake_batch["input_ids"].to(batch['text_tokens'].dtype)
                        fake_sigma = torch.ones(B, T, device=self.device).long()
                        fake_x_img_emb = torch.randn(B, T, 4 * (self.config.model.patching_downscale ** 2), device=self.device)
                        fake_modality = torch.zeros(B, T, device=self.device, dtype=torch.long)
                        fake_modality[:, self.config.model.txt_length:] = True
                        logits = self.backbone(indices=indices, sigma=fake_sigma, continuous_mode=True, x_img_emb=fake_x_img_emb, modality=fake_modality) # todo remove hardcoding 4
                    else:
                        logits = self.backbone(fake_batch["input_ids"], sigma=None, modality=fake_batch.get("modality", None))
                    from transformers.modeling_outputs import \
                        CausalLMOutputWithPast
                    if isinstance(logits, torch.Tensor):
                        logits = logits
                    elif isinstance(logits, tuple):
                        logits = logits[0]
                    elif isinstance(logits, CausalLMOutputWithPast):
                        logits = logits.logits

                    loss = logits.mean().to(torch.float32)
                    loss.backward()

                total_flops = flop_counter.get_total_flops()
                rprint(f"Total FLOPs Per Sample Fwd+Bwd: {(total_flops / self.config.loader.batch_size) / 1e12:.2f} TFLOPs")
                rprint(f"Total FLOPs Per Fwd+Bwd: {total_flops / 1e12:.2f} TFLOPs")
                rprint(f"Total FLOPs Per Global Step: {(total_flops / 1e12) * self.world_size * self.gradient_accumulation_steps:.2f} TFLOPs")

            rprint(f"GPU available FLOP/s: {get_available_flops(new_batch_idxs.device, self.dtype) / 1e12:.2f} TFLOP/s")
            rprint(f"Total available FLOP/s: {(get_available_flops(new_batch_idxs.device, self.dtype) / 1e12) * self.world_size * self.gradient_accumulation_steps:.2f} TFLOP/s")
            rprint(f"Used Batch Size: {self.config.loader.batch_size} for FLOP Calculations")

@torch.inference_mode()
def _ddpm_update(self, x, t, dt, **kwargs):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    nfe_cnt = 0
    _sigma = None if getattr(self.config.trainer, "force_null_sigma", False) else sigma_t
    p_x0 = self._ddpm_forward(x, t, _sigma, **kwargs)
    nfe_cnt += 1
    assert move_chance_t.ndim == p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    del p_x0, q_xs, move_chance_t, move_chance_s
    return copy_flag * x + (1 - copy_flag) * _x, nfe_cnt

@torch.inference_mode()
def _ddpm_caching_update(self, x, t, dt, p_x0=None, x0=None, x0_unmask=None, modality=None,**kwargs):
    assert self.config.noise.type == "loglinear"
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
        t = t.squeeze(-1)
        
    nfe_cnt = 0
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape

    if p_x0 is None:
        _sigma = None if getattr(self.config.trainer, "force_null_sigma", False) else sigma_t
        p_x0 = self._ddpm_forward(x, t, _sigma, x0=x0, x0_unmask=x0_unmask, modality=modality, **kwargs)
        nfe_cnt += 1
    assert move_chance_t.ndim == p_x0.ndim
    if self.config.trainer.force_bf16_eval: empty_device_cache()
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    if self.config.trainer.force_bf16_eval: empty_device_cache()

    if self.config.trainer.ar_shift:
        if x0 is not None:
            _x = torch.cat([x0[:, [0]], _x], dim=1)
        else:
            _x = torch.cat([torch.full_like(_x[..., :1], fill_value=self.tokenizer.pad_token_id), _x], dim=1)

    del q_xs, move_chance_t, move_chance_s
    return p_x0, copy_flag * x + (1 - copy_flag) * _x, nfe_cnt


@try_except(write_error_to_file=True, clear_cuda_cache=True)
@torch.inference_mode()
def _sample(
    self,
    num_steps=None,
    eps=1e-5,
    text_only=True,
    x0=None,
    x0_unmask=None,
    batch_size_per_gpu=None,
    example_batch=None,
    sample_batch_idx=None,
    sample_modality=None,
    sample_ids=None,
    return_raw_data=False,
    **kwargs,
):
    """Generate samples from the model."""
    if not (x0 is None) == (x0_unmask is None):
        breakpoint()
    assert (x0 is None) == (x0_unmask is None), f"x0: {x0} x0_unmask: {x0_unmask}"
    batch_size_per_gpu = (x0.shape[0] if x0 is not None else self.config.loader.eval_batch_size) if batch_size_per_gpu is None else batch_size_per_gpu
    sample_modality = kwargs.get("modality", None) if sample_modality is None else sample_modality
    kwargs['modality'] = sample_modality
    kwargs['sample_ids'] = sample_ids
    return_nfe = kwargs.pop('return_nfe', False)
    is_sample_masking = kwargs.pop('is_sample_masking', False)
    allow_interleaved_conditional = kwargs.pop('allow_interleaved_conditional', False)
    nfe_cnt = 0
    assert batch_size_per_gpu > 0
    if num_steps is None:
        num_steps = self.config.sampling.steps
    if getattr(self.config.eval, "test_eval_speed", False) and getattr(self.config.eval, 'eval_at_ratio_length', False):
        num_steps = self.config.model.length
        if getattr(self.config.eval, "num_steps_ratio", None) is not None:
            num_steps = int(num_steps * self.config.eval.num_steps_ratio)
    
    decode_kwargs = dict(sample_modality=sample_modality, return_raw_data=return_raw_data, is_sample_masking=is_sample_masking)

    if x0 is not None and x0_unmask is not None:
        x = self._sample_prior(batch_size_per_gpu, x0.shape[1]).to(self.device)
        decode_kwargs['x0_unmask'] = x0_unmask
        if getattr(self.config.eval, "visualize_sample", False):
            x_viz = x.clone()
            x_viz = torch.where(x0_unmask, x0, x)
            _mask_id = self.tokenizer("mask")['input_ids']
            assert len(_mask_id) == 3
            x_viz[x_viz == self.mask_index] = _mask_id[1]
            ret_txt, ret_img = self.decode_sampling(x_viz, text_only, **kwargs, **decode_kwargs, image_save_postfix="_masked_input")
            print(ret_txt)

    elif (self.config.trainer.interleaved and not self.config.backbone == "chameleon") and allow_interleaved_conditional:
        assert self.config.trainer.interleaved_training_flex_attention
        x0 = example_batch['input_ids'].to(self.device)
        total_samples = getattr(self.config.eval, "num_uncond_sample_batches", 1) - 1
        half_uncond = getattr(self.config.eval, "half_uncond", False)
        if not half_uncond or sample_batch_idx >= total_samples // 2:
            unmask_modality = getattr(self.config.eval, "unmask_modality", sample_batch_idx % 2)
            x0_unmask = sample_modality == unmask_modality
            if x0_unmask.sum() == x0.numel():
                unmask_modality = 1 - unmask_modality
                x0_unmask = sample_modality == unmask_modality

            if x0.shape != sample_modality.shape:
                breakpoint()
                
            if unmask_modality == 1:
                x0_unmask = torch.zeros_like(x0_unmask)
                for i in range(x0.shape[0]):
                    eos_pos = (x0[i] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        idx = random.randint(0, len(eos_pos) - 2)
                        x0_unmask[i, :] = True
                        if len(eos_pos) >= idx + 1:
                            _sl = slice(eos_pos[idx], None)
                        else:
                            _sl = slice(eos_pos[idx] + 2, eos_pos[idx+1] - 1)
                            
                        x0_unmask[i, _sl] = (sample_modality[i, _sl] == 1)

            # Set first sentence to be unmasked
            for i in range(x0.shape[0]):
                eos_pos = (x0[i] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    assert (eos_pos[0] < 48) or (sample_modality[i].sum() == 0), f"eos_pos: {eos_pos}"
                    x0_unmask[i, :eos_pos[0]+1] = True

            if unmask_modality == 1 and x0_unmask.sum() == 0:
                x0_unmask = torch.ones_like(x0_unmask)
                print(f"Found no umasked tokens, unmasking random sequences")
                for i in range(x0.shape[0]):
                    seq_len = (x0[i] != self.tokenizer.pad_token_id).sum()
                    if seq_len == 0:
                        continue
                        
                    start_pos = random.randint(0, seq_len-1)
                    max_len = min(seq_len - start_pos, 200)
                    unmask_len = random.randint(1, max_len)
                    x0_unmask[i, start_pos:start_pos+unmask_len] = False

            gprint(f"Unmasking modality: {unmask_modality}, Unmasking {(x0_unmask.sum() / x0_unmask.numel()):.2%} of image tokens. Txt tokens: {(sample_modality == 0).sum()}, Img tokens: {(sample_modality == 1).sum()}")

        x0_unmask[~example_batch['attention_mask']] = True
        x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
        decode_kwargs['x0_unmask'] = x0_unmask
        x = torch.where(x0_unmask, x0, x)

        if getattr(self.config.eval, "visualize_sample", False):
            x_viz = x.clone()
            _mask_id = self.tokenizer("mask")['input_ids']
            assert len(_mask_id) == 3
            _mask_id = _mask_id[1]
            x_viz[x == self.mask_index] = _mask_id
            self.decode_sampling(x_viz, text_only, **kwargs, **decode_kwargs, image_save_postfix="_x0_unmasked")

        if self.parameterization == "ar" or getattr(self.config.eval, "eval_large_batch", None) is not None:
            rprint(f"Masking all tokens by default.")
            x0_unmask = torch.zeros(*x0.shape, device=x0.device).to(torch.bool)
        else:
            rprint(f"Hit chamelon sample")
            if sample_batch_idx == getattr(self.config.eval, "num_uncond_sample_batches", 1) - 1:
                x0_unmask = torch.zeros(*x0.shape, device=x0.device, dtype=torch.bool)
                x0_unmask[..., -20:] = True
                rprint(f"Unmasking first {x0_unmask.shape[-1] // 2} tokens")
            else:
                x0_unmask = torch.rand(*x0.shape, device=x0.device) < (sample_batch_idx / 60)
                rprint(f"Unmasking {(sample_batch_idx / 60)} of image_tokens, {x0_unmask.sum()}")

        x = self._sample_prior(batch_size_per_gpu, x0.shape[1]).to(self.device)
        _img_indices = torch.isin(x0, torch.tensor(list(image_indices), device=self.device))
        if getattr(self.config.eval, "unmask_chameleon_txt", False):
            rprint(f"Unmasking all text tokens")
            x0_unmask |= _img_indices
            x0_unmask[:, :4] = True
            rprint(f"All tokens: {x0_unmask.tolist()}")
            # assert sample_modality is None
            # decode_kwargs['sample_modality'] = torch.isin(x0, torch.tensor(list(image_indices), device=self.device)).to(torch.long)
        else:
            x0_unmask |= (~_img_indices)

        kwargs['forward_attention_mask'] = attention_mask
        decode_kwargs['image_indices'] = image_indices
        decode_kwargs['x0_unmask'] = x0_unmask
        rprint(f"Unmasking: {torch.sum(x0_unmask)}")
    else:
        x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
        decode_kwargs['x0_unmask'] = x0_unmask

    if self.config.trainer.interleaved_training_flex_attention:
        assert 'sample_ids' in kwargs
        kwargs['block_mask'] = get_interleaved_block_mask(kwargs['sample_ids'], x.shape[0], x.shape[-1], self.device)

    if num_steps > (~x0_unmask).sum(dim=-1).min():
        rprint(f"num_steps {num_steps} > sequence length {(~x0_unmask).sum(dim=-1).min()}, setting num_steps to sequence length")
        num_steps = (~x0_unmask).sum(dim=-1).min()

    if self.parameterization == "ar":
        with show_memory_usage(empty_cache=True):
            out, nfe_cnt = self._ar_sampler(batch_size_per_gpu, x0=x0, x0_unmask=x0_unmask, **kwargs)
        res = self.decode_sampling(out, text_only, **kwargs, **decode_kwargs)
        if return_nfe:
            return res, nfe_cnt
        return res

    if x0 is not None and x0_unmask is not None:
        x = torch.where(x0_unmask, x0, x)

    if self.sampler == "maskgit" or self.sampler == "first_hitting" or self.sampler == "maskgit_nucleus":
        sampling_schedule = 'arccos' if self.sampler in ['maskgit', 'maskgit_nucleus'] else 'linear'
        
        # v1
        # schedule = adap_sche(num_steps, mode=sampling_schedule, seq_len=x.shape[-1], leave=False)
        
        # v2
        # make seq length equal to max number of masked tokens in any sample in the batch
        # Calculate the number of masked tokens for each sample in the batch
        # num_masked = (x == self.mask_index).sum(dim=-1)
        # Get the maximum number of masked tokens across all samples
        # min_masked = num_masked.min().item()
        # schedule = adap_sche(num_steps, mode=sampling_schedule, seq_len=min_masked, leave=False)
        
        # v3 - use x shape
        schedule = adap_sche(x=x, step=num_steps, mask_index=self.mask_index, mode=sampling_schedule)
        print(f"schedule: {schedule}")
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    
    is_x_sliced = False
    attention_caching = self.config.eval.attention_caching
    attention_caching_txt_to_img_ratio = getattr(self.config.eval, "attention_caching_txt_to_img_ratio", 10)
    if attention_caching:
        backbone = self.accelerator.unwrap_model(self.backbone)
        backbone.set_flex_attention_cache(x.shape[0], x.shape[1], self.device, self.dtype)
        full_data = dict()
        x_next = None

    # At the beginning of _sample method, after initializing variables
    if getattr(self.config.eval, "visualize_denoising", False):
        denoising_steps = [x.clone()]
    
    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype if self.config.trainer.force_bf16_eval else torch.float32)
        if attention_caching:
            if i % attention_caching_txt_to_img_ratio == 0:
                if is_x_sliced:
                    def replace_new_data(_key, _new_data):
                        if full_data[_key] is not None:
                            full_data[_key][:,self.static_txt_sl] =  _new_data
                        return full_data[_key]
                    
                    x = replace_new_data("x", x)
                    x0 = replace_new_data("x0", x0)
                    x0_unmask = replace_new_data("x0_unmask", x0_unmask)
                    p_x0_cache = replace_new_data("p_x0_cache", p_x0_cache)
                    kwargs["modality"] = replace_new_data("modality", kwargs.get("modality", None))
                    del full_data
                    full_data = dict()
                    is_x_sliced = False

                update_cache_slice = None
                block_mask = True
            elif (i - 1) % attention_caching_txt_to_img_ratio == 0:
                update_cache_slice = slice(0, x.shape[1])
                block_mask = get_block_mask(
                    txt_batch_attn_dropout=torch.zeros(x.shape[0], dtype=torch.bool, device=x.device), 
                    img_batch_attn_dropout=torch.ones(x.shape[0], dtype=torch.bool, device=x.device), 
                    txt_length=self.config.model.txt_length, 
                    batch_size=x.shape[0],
                    seq_len=x.shape[1],
                    device=x.device
                )
            else:
                update_cache_slice = self.static_txt_sl
                block_mask = True
                if not is_x_sliced:
                    is_x_sliced = True
                
                    def clone_if_valid(_data):
                        if _data is not None:
                            return _data.clone()
                        else:
                            return None
                        
                    def sl_if_valid(_data):
                        if _data is not None:
                            return _data[:, self.static_txt_sl]
                        else:
                            return None
                    
                    full_data.update(x=clone_if_valid(x), x0=clone_if_valid(x0), x0_unmask=clone_if_valid(x0_unmask), modality=clone_if_valid(kwargs.get("modality", None)), p_x0_cache=clone_if_valid(p_x0_cache))
                    x = sl_if_valid(x)
                    x0 = sl_if_valid(x0)
                    x0_unmask = sl_if_valid(x0_unmask)
                    x_next = sl_if_valid(x_next)
                    p_x0_cache = sl_if_valid(p_x0_cache)
                    kwargs["modality"] = sl_if_valid(kwargs.get("modality", None))

            kwargs["update_cache_slice"] = update_cache_slice
            kwargs["block_mask"] = block_mask

        if self.sampler == "maskgit":
            x, nfe_step_cnt = self._maskgit_update(x, t, dt, x0=x0, x0_unmask=x0_unmask, schedule=schedule, step=i, **kwargs)
        elif self.sampler == "maskgit_nucleus":
            x, nfe_step_cnt = self._maskgit_nucleus_update(x, t, dt, x0=x0, x0_unmask=x0_unmask, schedule=schedule, step=i, **kwargs)
        elif self.sampler == "first_hitting":
            x, nfe_step_cnt = self._first_hitting_update(x, t, dt, x0=x0, x0_unmask=x0_unmask, schedule=schedule, step=i, **kwargs)
        elif self.sampler == "ddpm":
            x, nfe_step_cnt = self._ddpm_update(x, t, dt, x0=x0, x0_unmask=x0_unmask, **kwargs)
        elif self.sampler == "ddpm_tweedie":
            assert not return_nfe, "Tweedie sampler does not support return_nfe"
            x = self._ddpm_update_finetune_controlled_tweedie(x, t, dt, sampling_step=i, **kwargs)
            nfe_step_cnt = 0
        elif self.sampler == "ddpm_cache":
            p_x0_cache, x_next, nfe_step_cnt = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache, x0=x0, x0_unmask=x0_unmask, **kwargs)
            if not torch.allclose(x_next, x) or self.time_conditioning:
                p_x0_cache = None  # Disable caching
            x = x_next
        else:
            x, nfe_step_cnt = self._analytic_update(x, t, dt)

        nfe_cnt += nfe_step_cnt
        if self.tokenizer.eos_token_id in x and getattr(self.config.trainer, "force_after_eos_padding", False) and (self.tokenizer.eos_token_id != self.tokenizer.bos_token_id) and not attention_caching:
            after_first_eos = torch.cumsum(x == self.tokenizer.eos_token_id, dim=1).bool()
            after_first_eos_mask = after_first_eos.cumsum(dim=1) > 1
            to_mask = ((after_first_eos_mask & (sample_modality == 0)) & (x != self.tokenizer.pad_token_id)) & (x != self.mask_index)
            x[to_mask] = self.tokenizer.pad_token_id

            if to_mask.sum() > 0:
                rprint(f"Masked an avg of {torch.sum(to_mask, dim=1).float().mean()} tokens due to EOS.")

        if x0 is not None and x0_unmask is not None: x = torch.where(x0_unmask, x0, x)
        
        # Add capture of current state for visualization
        if getattr(self.config.eval, "visualize_denoising", False) and i % getattr(self.config.eval, "visualize_step_interval", max(1, num_steps // 10)) == 0:
            denoising_steps.append(x.clone())
            
        clear_gpu_memory_if_needed()
    
    if getattr(self.config.eval, "visualize_denoising", False) and denoising_steps:
        if denoising_steps[-1] is not x:
            denoising_steps.append(x.clone())
        
        step_images = []
        for step_x in denoising_steps:
            _, step_res = self.decode_sampling(step_x, text_only=False, bypass_return_interleaved_modalities_split=True, **kwargs, **decode_kwargs)
            if not isinstance(step_res, Image.Image):
                step_res = step_res[0]
            step_images.append(step_res)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path("/dev/shm") / os.getenv("USER", 'user') / "denoise_vis" / date_folder / f"{timestamp}.png"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        Im.concat_horizontal(step_images).save(save_dir)
        rprint(f"Saved denoising visualization to {save_dir}")

    if is_x_sliced:
        def replace_new_data(_key, _new_data):
            if full_data[_key] is not None:
                full_data[_key][:,self.static_txt_sl] =  _new_data
            return full_data[_key]
        
        x = replace_new_data("x", x)
        x0 = replace_new_data("x0", x0)
        x0_unmask = replace_new_data("x0_unmask", x0_unmask)
        p_x0_cache = replace_new_data("p_x0_cache", p_x0_cache)
        kwargs["modality"] = replace_new_data("modality", kwargs.get("modality", None))
        del full_data
        full_data = dict()
        is_x_sliced = False

    if self.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        if self.sampler == "analytic":
            x = self._denoiser_update(x, t)
        else:
            unet_conditioning = self.noise(t)[0]
            x = self.forward(x=x, sigma=unet_conditioning, **kwargs).argmax(dim=-1)

    if x0 is not None and x0_unmask is not None:
        x = torch.where(x0_unmask, x0, x)
    res = self.decode_sampling(x, text_only, **kwargs, **decode_kwargs)

    if return_nfe:
        return res, nfe_cnt
    return res

def decode_sampling(self, x, text_only, is_sample_masking=False, bypass_return_interleaved_modalities_split=False, **kwargs):
    if self.config.trainer.interleaved and getattr(self.config.eval, "return_interleaved_modalities_split", False) and not bypass_return_interleaved_modalities_split:
        decoded_data = self.decode_batch({"input_ids": x, **kwargs}, text_only=False)
        image_save_postfix = kwargs.get("image_save_postfix", None)
        assert len(decoded_data) == 1
        all_imgs = []
        all_txt = []
        for i in range(min(len(decoded_data), 64)):
            sample_data, sample_modalities = decoded_data[i].to_list()
            ret = self.get_interleaved_image(sample_data, sample_modalities, image_save_postfix=image_save_postfix)
            all_txt_in_sample = []
            all_img_in_sample = []
            for j in range(len(sample_data)):
                if sample_modalities[j] == 0:
                    text_samples = sample_data[j]
                    pred_txt = wrapped_batch_decode(
                        self.tokenizer, text_samples[None], clean_up_tokenization_spaces=False, skip_special_tokens=False, disable_mask_after_eos=True
                    )
                    all_txt_in_sample.extend(pred_txt)
                else:
                    img_samples = sample_data[j]
                    pred_img = decode_latents(self.config, self.get_vae(), img_samples[None])
                    all_img_in_sample.extend([Im(x).pil for x in pred_img])

            # in case we have  text...<image>.</image>. This causes [" text...", "</image>"], which we merge below.
            if len(all_txt_in_sample) >= 2 and all_txt_in_sample[-1] == self.tokenizer.eos_token:
                all_txt_in_sample[-2] += all_txt_in_sample[-1]
                all_txt_in_sample.pop()

            all_txt.extend(all_txt_in_sample)
            all_imgs.extend(all_img_in_sample)

        print(f"Returning... all_txt: {all_txt}, all_imgs: {all_imgs}")
        for i in range(len(all_imgs)):
            filename = f"img_{get_rank()}_{str(time.time()).replace('.', '__')}.png"
            Im(all_imgs[i]).save(filename)
        return all_txt, all_imgs
    elif (self.config.trainer.interleaved and not is_sample_masking) or getattr(self.config.eval, "fake_interleaved", False):
        image_save_postfix = kwargs.get("image_save_postfix", None)
        decoded_data = self.decode_batch({"input_ids": x, **kwargs}, text_only=False)
        all_imgs = []
        all_txt_ids = []
        num_text_tokens = self.config.model.txt_length
        for i in range(min(len(decoded_data), 64)):
            sample_data, sample_modalities = decoded_data[i].to_list()
            all_imgs.append(self.get_interleaved_image(sample_data, sample_modalities, image_save_postfix=image_save_postfix))
            all_txt_ids_in_sample = []
            for j in range(len(sample_data)):
                if sample_modalities[j] == 0:
                    text_samples = sample_data[j]
                    if text_samples.shape[-1] < num_text_tokens:
                        text_samples = torch.nn.functional.pad(
                            text_samples,
                            (0, num_text_tokens - text_samples.shape[-1]),
                            value=self.tokenizer.pad_token_id
                        )
                    else:
                        text_samples = text_samples[..., :num_text_tokens]
                    all_txt_ids_in_sample.append(text_samples)

            if len(all_txt_ids_in_sample) == 0:
                all_txt_ids_in_sample.append(torch.zeros((num_text_tokens), dtype=torch.long, device=self.device))

            all_txt_ids.append(torch.cat(all_txt_ids_in_sample, dim=0))

        if kwargs.get("return_raw_data", False):
            return all_txt_ids, all_imgs, x
        
        return all_txt_ids, all_imgs
    else:
        ret = self.decode_batch(x, text_only=text_only, **kwargs)
        if getattr(self.config.eval, "visualize_sample", False):
            self.save_image_text_pair(ret[1], ret[0][:, self.static_txt_sl])
        return ret


@tensorclass
class InputData:
    # x0: Float[Tensor, "b c h w"]
    xt_ids: Integer[Tensor, "b h w c"]

    # x0_emb: Optional[Float[Tensor, "b h w 2"]] = None
    xt_img_embed: Optional[Float[Tensor, "b h w 2"]] = None
    modality: Bool[Tensor, "b h w"] = False
    sigma: Optional[Float[Tensor, "b"]] = None

@torch.no_grad()
def sample_transfusion(
    self,
    batch_size_per_gpu=None,
    text_only=False, # todo maybe make default True
):  
    """Generate samples from the model in autoregressive discrete mode for text and diffusion for image."""
    # x0 = example_batch["input_ids"] # (for img tokens?)
    # x0_emb = batch["img_emb"]
    B = batch_size_per_gpu if batch_size_per_gpu is not None else self.config.loader.eval_batch_size
    T = self.config.model.length
    C = self.config.model.downscale_ratio # = vae_latent_dim * (patching_downscale ** 2)
    # num_pred_tokens = T - 1
    num_img_tokens = self.config.model.img_length
    num_img_diffusion_steps = self.config.sampling.steps

    # TODO @sid This should be what we want? but for interleaved, we should prob add eos after text and before start img.
    xt_ids = torch.full((B, T), fill_value=self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
    xt_ids[:, 0] = self.tokenizer.bos_token_id
    xt_img_embed = torch.zeros((B, T, C), device=self.device)
    modality = torch.zeros((B, T), dtype=torch.long, device=self.device) # assuming everything is text initially
    sigma = torch.zeros((B, T), dtype=self.dtype, device=self.device)
    data = InputData(xt_ids=xt_ids, xt_img_embed=xt_img_embed, modality=modality, sigma=sigma, batch_size=[B])

    noise = torch.distributions.Gumbel(0, 1).sample((data.shape[0], T, self.vocab_size)).to(self.device)
    img_start_token_id = self.tokenizer.eos_token_id
    i = 1 # since we already have <bos>
    continuous_diffusion_mode = False
    while i < T:
        if continuous_diffusion_mode:
            # Diffusing mode
            img_sl = slice(i, i+num_img_tokens)
            data.modality[:, img_sl] = 1
            data.xt_img_embed[:, img_sl] = self.sample_continuous_image(data, img_sl=img_sl, num_steps=num_img_diffusion_steps, return_embeddings=True) # (b, n_img, latent_dim * 4)
            i += num_img_tokens
            continuous_diffusion_mode = False
            break    
        else:
            # autoregressive mode
            ar_sl = slice(None, i)
            if self.use_kv_cache:
                start_pos = i - 1
                kv_sl = slice(start_pos, i)
            else:
                kv_sl = ar_sl
                start_pos=None
            pred_logits, pred_noise = self.forward(x=data.xt_ids[:, kv_sl], sigma=data.sigma[:, ar_sl], modality=data.modality[:, ar_sl], x_img_emb=data.xt_img_embed[:, ar_sl], disable_ar_shift=True, continuous_mode=True, start_pos=start_pos)
            pred_logits = pred_logits[:, -1]
            y = (pred_logits + noise[:,  i]).argmax(-1)
            # y = (pred_logits).argmax(-1)

            data.xt_ids[:, i] = y
            # data.xt_ids[:, i + 1] = y
            i += 1
            if not text_only and (i == self.config.model.txt_length-1 or torch.all(y == img_start_token_id)): # todo make variable <boi>
                continuous_diffusion_mode = True

    if self.config.model.use_kv_cache:
        backbone = self.accelerator.unwrap_model(self.backbone)
        backbone.reset_kv_cache(batch_size=self.config.model.inference_max_batch_size, seq_len=self.config.model.inference_max_seq_len, dtype=self.dtype, device=self.device)

    return data    

def sample_continuous_image(self, data: InputData, img_sl, num_steps=None, return_embeddings=False):
    if num_steps is None:
        num_steps = self.config.sampling.steps
    B = data.xt_img_embed.shape[0]
    noise_scheduler = self.vae.scheduler
    noise_scheduler.set_timesteps(num_steps, device=self.device)
    timesteps = noise_scheduler.timesteps
    data.xt_img_embed[:, img_sl] = torch.randn_like(data.xt_img_embed[:, img_sl])

    visible_sl = slice(None, img_sl.stop)
    for i in range(num_steps+1):
        data.sigma[:, img_sl] = (timesteps[i] * torch.ones(B, device=self.device)).unsqueeze(-1)
        pred_logits, pred_noise = self.forward(
            x=data.xt_ids[:, visible_sl], sigma=data.sigma[:, visible_sl], x_img_emb=data.xt_img_embed[:, visible_sl], modality=data.modality[:, visible_sl], disable_ar_shift=True, continuous_mode=True
        )  # exp not needed since we predict noise (b,n,c) in latent space directly, not a probability distribution
        data.xt_img_embed[:, img_sl] = noise_scheduler.step(pred_noise[:, img_sl], timesteps[i], data.xt_img_embed[:, img_sl]).prev_sample

    if return_embeddings: return data.xt_img_embed[:, img_sl] # (b, n_img, latent_dim * 4)

    # x = x.transpose(1, 2)
    # data.xt_img_embed[:, img_sl] = data.xt_img_embed[:, img_sl].transpose(1, 2)
    text_tokens, img_tokens = self.decode_batch(data.xt_ids[:, img_sl], text_only=False)
    return text_tokens, img_tokens


def cfg(config, t, cat_p_x0):
    logit_c, logit_u = cat_p_x0.chunk(2, dim=0)
    _cfg = config.eval.cfg
    if not getattr(config.eval, "force_cfg_value", False):
        if _cfg == -1:
            _cfg = torch.linspace(0, 10, t.shape[0]).to(t.device)
        _w = (_cfg * (1 - t))[:, None, None]
    else:
        _w = _cfg

    return (1 + _w) * logit_c - _w * logit_u

def nucleus_sampling_batch(logits, top_p=0.9, temperature=1.0):
    """
    Perform nucleus (top-p) sampling on batched and sequenced logits.

    Args:
        logits (torch.Tensor): A tensor of shape (B, N, C) where B is the batch size,
                               N is the sequence length, and C is the number of classes.
        top_p (float): The cumulative probability threshold for nucleus sampling.
        temperature (float): Temperature value for scaling logits.

    Returns:
        torch.Tensor: Indices sampled from the filtered distribution for each position,
                      with shape (B, N).
    """
    B, N, C = logits.shape

    # Apply softmax to get probabilities
    # probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  # Shape: (B, N, C)
    probs = logits / temperature

    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # Both shape: (B, N, C)

    # Compute the cumulative sum of probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # Shape: (B, N, C)

    # Create a mask for top-p
    mask = cumulative_probs <= top_p  # Shape: (B, N, C)

    # Ensure at least one token is included
    mask[:, :, 0] = True

    # Apply the mask to the sorted probabilities
    filtered_probs = sorted_probs * mask.float()  # Shape: (B, N, C)

    # Renormalize the probabilities
    filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Shape: (B, N, C)

    # Sample from the renormalized distribution
    sampled_indices = torch.multinomial(filtered_probs.view(-1, C), num_samples=1).squeeze(-1)  # Shape: (B*N)

    # Reshape sampled_indices to (B, N)
    sampled_indices = sampled_indices.view(B, N)

    # Gather the original indices based on sorted_indices
    final_indices = torch.gather(sorted_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)  # Shape: (B, N)

    return final_indices

def nucleus_sampling(logits, top_p=0.9, temperature=1.0):
    """
    Perform nucleus (top-p) sampling on the given logits.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C) where B is the batch size
                               and C is the number of classes.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: Indices sampled from the filtered distribution.
    """
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

    # Sort the probabilities in descending order and get the sorted indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute the cumulative sum of probabilities along the last dimension
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask to filter out probabilities that contribute to top_p mass
    mask = cumulative_probs <= top_p
    
    # Ensure at least one token is always included
    mask[..., 0] = True  # Always include the most probable token

    # Zero out probabilities that are not part of the top-p mass
    filtered_probs = sorted_probs * mask.float()

    # Renormalize the filtered probabilities
    filtered_probs /= (filtered_probs.sum(dim=-1, keepdim=True))
    # Sample from the renormalized distribution
    sampled_indices = torch.multinomial(filtered_probs, num_samples=1)[:, 0]
    # Map back to original indices
    final_indices = sorted_indices.gather(dim=-1, index=sampled_indices.unsqueeze(-1)).squeeze(-1)

    return final_indices

def clear_gpu_memory_if_needed():
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory
        if current_memory >= 0.50:
            torch.cuda.empty_cache()

def _ar_sampler(self, B, x0=None, x0_unmask=None, modality=None, **kwargs):
    assert B > 0
    assert (x0 is None) == (x0_unmask is None), f"x0: {x0} x0_unmask: {x0_unmask}"
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros((B, num_pred_tokens + 1), dtype=torch.long, device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    if x0 is not None: x = torch.where(x0_unmask, x0, x)
    split_cfg_batches = getattr(self.config.eval, "split_cfg_batches", False) and not self.config.model.use_kv_cache
    effective_bs = B * 2 if ((self.config.eval.cfg is not None and x0 is not None) and split_cfg_batches is False) else B
    top_p = getattr(self.config.eval, "top_p", None)
    temperature = getattr(self.config.eval, "temperature", 1.0)
    if self.config.model.use_kv_cache:
        assert getattr(self.config.model, "inference_max_batch_size", None) is None
        assert getattr(self.config.model, "inference_max_seq_len", None) is None
        self.accelerator.unwrap_model(self.backbone).reset_kv_cache(
            batch_size=effective_bs,
            seq_len=num_pred_tokens,
            dtype=self.dtype,
            device=self.device
        )

    _x, _modality = None, None
    if self.config.eval.cfg is not None and x0 is not None:
        if split_cfg_batches is False:
            _x = torch.cat([x, torch.where(x0_unmask, self.mask_index, x)], dim=0)
            _modality = torch.cat([modality, modality], dim=0)

    nfe_cnt = 0
    noise = torch.distributions.Gumbel(0, 1).sample((B, num_pred_tokens, self.vocab_size)).to(self.device) # precompute noise
    for i in range(num_pred_tokens):
        start_pos = i if self.use_kv_cache else None
        ar_sl = slice(start_pos, i+1)

        if self.config.eval.cfg is not None and x0 is not None:
            if split_cfg_batches:
                logit_c = self.forward(
                    x=x[:, ar_sl], sigma=None, modality=modality[:, ar_sl], start_pos=start_pos, disable_ar_shift=True
                )[:, -1] 
                logit_u = self.forward(
                    x=torch.where(x0_unmask, self.mask_index, x)[:, ar_sl], sigma=None, modality=modality[:, ar_sl], start_pos=start_pos, disable_ar_shift=True
                )[:, -1]
            else:
                _x[:B] = x
                _x[B:] = torch.where(x0_unmask, self.mask_index, x)
                next_logits = self.forward(x=_x[:, ar_sl], sigma=None, modality=_modality[:, ar_sl], start_pos=start_pos, disable_ar_shift=True)[:, -1]
                logit_c, logit_u = next_logits.chunk(2, dim=0)

            _w = self.get_cfg_weight(1 - (i / num_pred_tokens))
            next_logits = (1 + _w) * logit_c - _w * logit_u
        else:
            next_logits = self.forward(x=x[:, ar_sl], sigma=None, modality=modality[:, ar_sl], start_pos=start_pos, disable_ar_shift=True)[:, -1]

        if getattr(self.config.model, "force_argmax_valid_indices", False):
            # start_pos = i
            next_sl = slice(i + 1, i + 2)
            try:
                next_logits[..., self.text_vocab_size:] = torch.where((modality[:, next_sl] == 0), torch.finfo(next_logits.dtype).min, next_logits[..., self.text_vocab_size:])
                next_logits[..., :self.text_vocab_size] = torch.where((modality[:, next_sl] == 1), torch.finfo(next_logits.dtype).min, next_logits[..., :self.text_vocab_size])
            except:
                breakpoint()
        if top_p is not None:
            # do nucleus sampling
            y = nucleus_sampling(next_logits, top_p=top_p, temperature=temperature)
        else:
            next_logits = next_logits + noise[:, i]
            nfe_cnt += 1
            y = (next_logits).argmax(-1)
        x[:, i + 1] = y
        if x0 is not None: x = torch.where(x0_unmask, x0, x)
        if not self.config.model.use_kv_cache:
            empty_device_cache()
        
        if getattr(self.config.eval, "force_empty_cache", False):
            empty_device_cache()

    if self.config.model.use_kv_cache:
        # TODO: PyTorch must have a b
        del noise, next_logits, _x, _modality
        self.accelerator.unwrap_model(self.backbone).reset_kv_cache(
            batch_size=effective_bs,
            seq_len=num_pred_tokens,
            dtype=self.dtype,
            device=self.device,
            set_to_none=True
        )

    return x, nfe_cnt

def handle_interleaved_decode(self, sample, allow_mask_index=False, new_mask_index=None, **kwargs):
    batch = sample
    sample_modality = sample.get("modality", None)
    sample = sample.get("input_ids", None)

    text_tokens = torch.where(sample_modality == 0, sample, self.tokenizer.pad_token_id)
    img_tokens = torch.where((sample_modality == 1), sample, self.mask_index)

    invalid_text_mask = (text_tokens >= self.text_vocab_size) & (sample_modality == 0)
    invalid_img_mask = (img_tokens < self.text_vocab_size) & (sample_modality == 1)
    mask_img_mask = (img_tokens == self.mask_index) & (sample_modality == 1)

    if invalid_text_mask.sum() > 0:
        assert allow_mask_index or self.config.model.force_argmax_valid_indices is False or self.config.sampling.predictor == "ddpm_tweedie" or self.config.parameterization == "ar", f"invalid_text_mask.sum(): {invalid_text_mask.sum()}, {invalid_text_mask.nonzero()[:4]}"
        text_tokens[invalid_text_mask] = self.mask_index

    if new_mask_index is not None:
        img_invalid_mask_v2 = ((img_tokens < self.text_vocab_size) & (img_tokens != self.mask_index))

    sample = torch.where(sample_modality == 1, img_tokens - self.text_vocab_size, text_tokens)
    if invalid_img_mask.sum() > 0 or mask_img_mask.sum() > 0:
        if new_mask_index is not None:
            assert img_invalid_mask_v2.sum().item() == 0
            sample[mask_img_mask] = new_mask_index
        else:
            sample[mask_img_mask] = 0
            sample[invalid_img_mask] = 0

    new_batch = {**batch, "input_ids": sample}
    new_batch = InterleavedBatch.custom_from_dict(new_batch)
    new_batch = new_batch.to_elements()
    return new_batch

def decode_batch(self,
    sample,
    text_only=True,
    return_masks: bool = False,
    allow_mask_index: bool = False,
    new_mask_index=None,
    sample_modality=None,
    **kwargs
):

    if isinstance(sample, dict) or isinstance(sample, TensorDict):
        if self.config.trainer.interleaved or getattr(self.config.eval, "fake_interleaved", False):
            return handle_interleaved_decode(self, sample, allow_mask_index=allow_mask_index, new_mask_index=new_mask_index, **kwargs)
        else:
            sample_modality = sample.get("modality", None)
            sample = sample.get("input_ids", None)

    img_tokens = None
    continuous_mode = self.config.trainer.image_mode == "continuous"
    if continuous_mode:
        text_tokens, img_tokens = sample[..., self.static_txt_sl], sample[..., self.static_img_sl]
    elif self.unified_model and self.config.trainer.multimodal_batches and sample_modality is not None:
        if (sample_modality == 0).all(dim=-1).sum() > 0:
            text_tokens = torch.where(sample_modality == 0, sample, self.tokenizer.pad_token_id)
            img_tokens = torch.where((sample_modality == 1)[:, self.static_img_sl], sample[:, self.static_img_sl], self.mask_index)
        else:
            text_tokens = torch.where(sample_modality == 0, sample, self.tokenizer.pad_token_id)
            img_tokens = torch.where((sample_modality == 1), sample, self.mask_index)

        invalid_text_mask = text_tokens >= self.text_vocab_size
        if getattr(self.config.model, "add_labels", None) is not None:
            invalid_img_mask = (img_tokens < self.text_vocab_size) | (img_tokens >= (self.vocab_size - self.config.model.add_labels))
        else:
            invalid_img_mask = (img_tokens < self.text_vocab_size)
        mask_text_mask = text_tokens == self.mask_index
        mask_img_mask = img_tokens == self.mask_index
        if invalid_text_mask.sum() > 0:
            assert allow_mask_index or self.config.model.force_argmax_valid_indices is False or self.config.sampling.predictor == "ddpm_tweedie" or self.config.parameterization == "ar", f"invalid_text_mask.sum(): {invalid_text_mask.sum()}, {invalid_text_mask.nonzero()[:4]}"
            text_tokens[invalid_text_mask] = self.mask_index

        if new_mask_index is not None:
            img_invalid_mask_v2 = ((img_tokens < self.text_vocab_size) & (img_tokens != self.mask_index))

        img_tokens = img_tokens - self.text_vocab_size
        if invalid_img_mask.sum() > 0 or mask_img_mask.sum() > 0:
            if new_mask_index is not None:
                assert img_invalid_mask_v2.sum().item() == 0
                img_tokens[mask_img_mask] = new_mask_index
            else:
                img_tokens[mask_img_mask] = 0
                img_tokens[invalid_img_mask] = 0
        
        if img_tokens.shape[-1] != self.config.model.img_length:
            if (sample_modality[:, -self.config.model.img_length:].sum(dim=-1) == self.config.model.img_length).all():
                img_tokens = img_tokens[:, -self.config.model.img_length:]
            elif (sample_modality[:, :self.config.model.img_length].sum(dim=-1) == self.config.model.img_length).all():
                img_tokens = img_tokens[:, :self.config.model.img_length]

    elif self.unified_model:
        text_tokens, img_tokens = sample[..., self.static_txt_sl], sample[..., self.static_img_sl]
        invalid_text_mask = text_tokens >= self.text_vocab_size
        invalid_img_mask = img_tokens < self.text_vocab_size
        mask_text_mask = text_tokens == self.mask_index
        mask_img_mask = img_tokens == self.mask_index

        if invalid_text_mask.sum() > 0:
            assert allow_mask_index or self.config.model.force_argmax_valid_indices is False or self.config.sampling.predictor == "ddpm_tweedie" or self.config.parameterization == "ar", f"invalid_text_mask.sum(): {invalid_text_mask.sum()}"
            text_tokens[invalid_text_mask] = self.mask_index

        if new_mask_index is not None:
            img_invalid_mask_v2 = ((img_tokens < self.text_vocab_size) & (img_tokens != self.mask_index))

        img_tokens = img_tokens - self.text_vocab_size
        if invalid_img_mask.sum() > 0 or mask_img_mask.sum() > 0:
            assert allow_mask_index or self.config.model.force_argmax_valid_indices is False or self.config.sampling.predictor == "ddpm_tweedie" or self.config.parameterization == "ar", f"invalid_img_mask.sum(): {invalid_img_mask.sum()}"
            if new_mask_index is not None:
                assert img_invalid_mask_v2.sum().item() == 0
                img_tokens[mask_img_mask] = new_mask_index
            else:
                img_tokens[mask_img_mask] = 0
                img_tokens[invalid_img_mask] = 0

        try:
            assert img_tokens.shape[-1] == self.config.model.img_length, f"img_tokens.shape[-1]: {img_tokens.shape[-1]}, config.model.img_length: {self.config.model.img_length}, sample_modality: {sample_modality}"
        except:
            breakpoint()
            
    elif self.image_model:
        text_tokens, img_tokens = None, sample
    else:
        text_tokens, img_tokens = sample, None
    if text_only:
        return text_tokens
    else:
        if return_masks:
            return text_tokens, img_tokens, mask_text_mask, mask_img_mask
        else:
            return text_tokens, img_tokens

def optional_add_bos(self, _x, x0):
    if self.config.trainer.ar_shift:
        if x0 is not None:
            _x = torch.cat([x0[:, [0]], _x], dim=1)
        else:
            _x = torch.cat([torch.full_like(_x[..., :1], fill_value=self.tokenizer.pad_token_id), _x], dim=1)
    return _x

def adap_sche(x, step, mask_index, mode="arccos"):
    """ Create a 2D sampling scheduler
        :param
        x     -> torch.Tensor: input tensor with shape (B, seq_len)
        step  -> int: number of prediction steps during inference
        mode  -> str: the rate of value to unmask
        leave -> bool: tqdm arg on either to keep the bar or not
        :return
        scheduler -> torch.LongTensor(): 2D tensor of shape (B, max_seq_len) with schedules for each sample
    """
    num_masked = (x == mask_index).sum(dim=-1).to(x.device)
    
    r = torch.linspace(1, 0, step)
    
    if mode == "root":
        val_to_mask = 1 - (r ** .5)
    elif mode == "linear":
        val_to_mask = 1 - r
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return None
    val_to_mask = val_to_mask.to(x.device)
    schedules = []
    for seq_len in num_masked:
        print(f"seq_len: {seq_len}")
        sche = (val_to_mask / val_to_mask.sum()) * seq_len
        sche = sche.round()
        sche[sche == 0] = 1
        sche[-1] += seq_len - sche.sum()
        sche[-1] = max(sche[-1], 0)
        schedules.append(sche.int())

    return torch.stack(schedules, dim=0)

    
@torch.no_grad()
def _first_hitting_update(self, x, t, dt, schedule=None, step=None, **kwargs):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]

    _sigma = None if getattr(self.config.trainer, "force_null_sigma", False) else sigma_t
    nfe_cnt = 0
    p_x0 = self._ddpm_forward(x, t, _sigma, **kwargs)
    nfe_cnt += 1

    copy_flag = (x != self.mask_index) # [B, N]

    # TODO: inefficient that we sample all tokens even if we only want to unmask a few
    _x = _sample_categorical(p_x0)

    num_unmask = schedule[:, step]
    num_unmask = torch.minimum(num_unmask, (~copy_flag).sum(dim=-1))
    if torch.all(num_unmask <= 0):
        return x, nfe_cnt

    random_values = torch.rand_like(copy_flag, dtype=torch.float32)
    random_values = torch.where(~copy_flag, random_values, -1)
    _, indices = torch.sort(random_values, dim=-1, descending=True)
    range_tensor = torch.arange(copy_flag.shape[-1], device=copy_flag.device).expand(copy_flag.shape)
    final_mask = range_tensor < num_unmask[:, None]

    result = torch.zeros_like(copy_flag)
    result.scatter_(-1, indices, final_mask)

    return torch.where(result, _x, x), nfe_cnt

@torch.no_grad()
def _maskgit_update(self, x, t, dt, schedule=None, step=None, **kwargs):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    nfe_cnt = 0
    _sigma = None if getattr(self.config.trainer, "force_null_sigma", False) else sigma_t

    copy_flag = (x != self.mask_index)
    r_temp = getattr(self.config.eval, 'maskgit_r_temp', 10)
    num_unmask = schedule[:, step]
    # rprint(f"num_unmask: {num_unmask}, (~copy_flag).sum(dim=-1).max().item(): {(~copy_flag).sum(dim=-1).max().item()}")
    num_unmask = torch.minimum(num_unmask, (~copy_flag).sum(dim=-1))
    if torch.all(num_unmask <= 0):
        return x, nfe_cnt

    p_x0 = self._ddpm_forward(x, t, _sigma, **kwargs)
    nfe_cnt += 1
    pred_code = torch.multinomial(p_x0.view(-1, p_x0.shape[-1]), 1)[:, 0].view(p_x0.shape[:-1])
    conf = torch.gather(p_x0, -1, pred_code.unsqueeze(-1)).squeeze(-1)
    
    rand = r_temp * torch.from_numpy(np.random.gumbel(size=pred_code.shape)).to(self.device) * t
    conf = torch.log(conf.squeeze()) + rand

    if self.config.trainer.ar_shift:
        copy_flag = copy_flag[:, 1:]

    # do not predict on already predicted tokens
    conf = torch.where(copy_flag, -torch.inf, conf)

    # Choose the predicted tokens with the highest confidence
    # Get the maximum num_unmask across the batch for top k
    max_num_unmask = num_unmask.max().item()
    
    # Use top k to get the highest confidence tokens
    tresh_conf, indice_mask = torch.topk(conf, k=max_num_unmask, dim=-1)
    
    # tresh_conf is [B, max_num_unmask]
    # for each sample i, we want to get num_unmask[i] highest confidence tokens

    # handle the case where num_unmask is 0 by setting the threshold to inf
    gather_indices = torch.clamp(num_unmask - 1, min=0)[:, None]
    tresh_conf = tresh_conf.gather(-1, gather_indices)
    tresh_conf = torch.where((num_unmask <= 0)[:, None], torch.inf, tresh_conf)
    
    # replace the chosen tokens
    conf = (conf >= tresh_conf.expand_as(conf))
    if self.config.trainer.ar_shift:
        out = torch.where(conf, pred_code, x[:, 1:])
        out = optional_add_bos(self, out, x0=kwargs.get("x0", None))
    else:
        out = torch.where(conf, pred_code, x)

    if getattr(self.config.eval, "allow_token_updates", False):
        out = torch.where(copy_flag, p_x0.argmax(dim=-1), out)

    del conf, indice_mask, gather_indices, tresh_conf, pred_code, p_x0
    if getattr(self.config.eval, "force_empty_cache", False):
        empty_device_cache()

    return out, nfe_cnt


@torch.no_grad()
def _maskgit_nucleus_update(self, x, t, dt, schedule=None, step=None, **kwargs):
    nfe_cnt = 0
    _sigma = None # sigma useless for non time-conditioned models like us

    copy_flag = (x != self.mask_index)
    if self.config.trainer.ar_shift:
        copy_flag = copy_flag[:, 1:]

    assert getattr(self.config.eval, 'maskgit_r_temp', None) != None
    r_temp = getattr(self.config.eval, "maskgit_r_temp", 10)
    num_unmask = schedule[:, step]
    num_unmask = torch.minimum(num_unmask, (~copy_flag).sum(dim=-1))
    if num_unmask <= 0:
        return x, nfe_cnt
    
    p_x0 = self._ddpm_forward(x, t, _sigma, **kwargs)
    nfe_cnt += 1
    top_p = getattr(self.config.eval, "top_p", 0.95)
    temperature = getattr(self.config.eval, "temperature", 0.9)
    if top_p is not None:
        pred_code = nucleus_sampling_batch(p_x0, top_p=top_p, temperature=temperature)
    else:
        pred_code = torch.multinomial(p_x0.view(-1, p_x0.shape[-1]), 1)[:, 0].view(p_x0.shape[:-1]) # pred tokens?
    conf = torch.gather(p_x0, -1, pred_code.unsqueeze(-1)).squeeze(-1)

    rand = r_temp * torch.from_numpy(np.random.gumbel(size=pred_code.shape)).to(self.device) * t
    conf = torch.log(conf.squeeze()) + rand

    # do not predict on already predicted tokens
    conf = torch.where(copy_flag, -torch.inf, conf)

    # chose the predicted token with the highest confidence
    # get the maximum num_unmask across the batch for top k
    max_num_unmask = num_unmask.max().item()
    
    tresh_conf, indice_mask = torch.topk(conf, k=max_num_unmask, dim=-1)
    
    # for each sample i, we want to get num_unmask[i] highest confidence tokens
    # handle the case where num_unmask is 0 by setting the threshold to inf
    gather_indices = torch.clamp(num_unmask - 1, min=0)[:, None]
    tresh_conf = tresh_conf.gather(-1, gather_indices.long())
    tresh_conf = torch.where((num_unmask <= 0)[:, None], torch.inf, tresh_conf)

    # replace the chosen tokens
    conf = (conf >= tresh_conf)
    if self.config.trainer.ar_shift:
        out = torch.where(conf, pred_code, x[:, 1:])
        out = optional_add_bos(self, out, x0=kwargs.get("x0", None))
    else:
        out = torch.where(conf, pred_code, x)
    return out, nfe_cnt



@torch.no_grad()
def _ddpm_update_finetune_controlled_tweedie(self, x, t, dt, reward_model=None, repeats=10, sampling_step=None, **kwargs):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    _sigma = None if getattr(self.config.trainer, "force_null_sigma", False) else sigma_t
    p_x0 = self._ddpm_forward(x, t, _sigma, **kwargs)
    assert move_chance_t.ndim == p_x0.ndim

    if self.config.trainer.force_bf16_eval: empty_device_cache()
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    copy_flag = (x != self.mask_index).to(x.dtype)

    del p_x0, move_chance_t, move_chance_s
    resample_interval = getattr(self.config.eval, "tweedie_resample_interval", None)
    return_single_sample = False
    _repeats = repeats
    if resample_interval is not None and sampling_step % resample_interval != 0:
        _repeats = 1
        return_single_sample = True

    # Generate 10 samples for each position
    samples = [copy_flag * x + (1 - copy_flag) * optional_add_bos(self, _sample_categorical(q_xs), x0=kwargs.get("x0", None)) for _ in range(_repeats)]

    if return_single_sample:
        return samples[0]

    if not hasattr(self, "reward_model"):
        from unidisc.tokenizers.laion_aesthetic_v2 import get_predictor_func
        self.reward_model = get_predictor_func(self.device)
        rprint("Using reward model. Should delete this after eval.")

    # TODO: Make this more general (e.g., support interleaved text/image)
    # Get scores for each sample
    scores = []
    expected_x0_args = []
    for i in range(repeats):
        # Use Tweedie's formula. Aim to calcuate r(E[x_0|x_t])
        expected_x0 = self._ddpm_forward(samples[i], t, sigma_s, **kwargs) # Calcualte E[x_0|x_t]
        if getattr(self.config.eval, "use_generic_tweedie_rewards", False):
            assert self.config.trainer.interleaved
            expected_x0_arg = torch.argmax(expected_x0, dim=-1)
            expected_x0_args.append(expected_x0_arg)
            assert samples[0].shape[0] == 1
        else:
            expected_x0[..., :self.text_vocab_size] = 0
            expected_x0[..., self.mask_index] = 0
            expected_x0[..., self.text_vocab_size:] = expected_x0[..., self.text_vocab_size:] + 1e-6
            expected_x0_arg = torch.argmax(expected_x0, dim=-1)
            expected_x0_arg = expected_x0_arg - self.text_vocab_size
            expected_x0_img_pred = decode_latents(self.config, self.get_vae(), expected_x0_arg[:, self.static_img_sl])
            scorer = self.reward_model(expected_x0_img_pred) # [B]

            scorer = scorer.squeeze()
            if scorer.ndim == 0:
                scorer = scorer[None]
            scores.append(torch.from_numpy(scorer))

    if getattr(self.config.eval, "use_generic_tweedie_rewards", False):
        orig_modality = kwargs.get("modality", None)
        if orig_modality is not None:
            orig_modality = orig_modality.clone()
            kwargs["modality"] = orig_modality.repeat(len(expected_x0_args), 1)

        orig_sample_ids = kwargs.get("sample_ids", None)
        if orig_sample_ids is not None:
            orig_sample_ids = orig_sample_ids.clone()
            kwargs["sample_ids"] = orig_sample_ids.repeat(len(expected_x0_args), 1)

        decoded_data = self.decode_batch({"input_ids": torch.cat(expected_x0_args, dim=0), **kwargs}, text_only=False)
        kwargs["modality"] = orig_modality
        kwargs["sample_ids"] = orig_sample_ids

        all_imgs = []
        all_txt_ids = []
        for i in range(len(decoded_data)):
            sample_data, sample_modalities = decoded_data[i].to_list()
            assert len(sample_data) == 2
            assert sample_modalities == [0, 1]
            sample_text = wrapped_batch_decode(
                self.tokenizer,
                sample_data[0][None],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
                disable_mask_after_eos=True
            )
            assert len(sample_text) == 1
            all_txt_ids.append(sample_text[0])
            all_imgs.append(self.get_interleaved_image(sample_data, sample_modalities, single_image_only=True, disable_img_save=True))

        all_imgs = torch.cat(all_imgs, dim=0)
        reward_config = getattr(self.config.eval, "tweedie_reward_config")
        scores = self.get_rewards(reward_config, all_imgs, all_txt_ids).float().cpu()
        scores = torch.softmax(scores, dim=0)[None]
    else:
        scores = torch.stack(scores, dim=1)
        scores = torch.softmax(scores, dim=1) # Convert scores to probabilities for each batch

    # Sample from the weighted categorical distribution formed by scores
    # Select the index of the highest score for each batch
    final_sample_indices = torch.argmax(scores, dim=1) # Shape [batch_size]
    final_samples = [samples[final_sample_indices[j]][j,:] for j in range(x.size(0))]  # Select the chosen samples using gathered indices
    final_samples = torch.stack(final_samples, dim=0)
    return final_samples

@try_except(write_error_to_file=True, clear_cuda_cache=True)
def visualize_samples(self, batch, batch_idx, split='val'):
    split = split.removesuffix("/")
    gt_txt = None
    step_metrics = self.get_step_metrics()
    step_metrics["trainer/global_step"] = (batch_idx if self.config.eval.visualize_data_only else self.global_step)
    rprint('[IMPORTANT] Visualizing ground truth samples, verify tokenization')

    if getattr(self.config.eval, "disable_visualization", False):
        return

    if self.config.trainer.interleaved:
        decoded_data = self.decode_batch(batch, text_only=False)
        all_imgs = []
        max_num = 10000 if getattr(self.config.eval, "visualize_data_only", False) else 32
        for i in range(min(len(decoded_data), max_num)):
            sample_data, sample_modalities = decoded_data[i].to_list()
            all_imgs.append(self.get_interleaved_image(sample_data, sample_modalities))

        if not getattr(self.config.eval, "visualize_data_only", False):
            log({f"{split}/rec_img": wandb.Image(Im.concat_horizontal(*all_imgs).pil), **step_metrics})
    else:
        gt_txt, gt_img = self.decode_batch(batch["input_ids"], text_only=False, sample_modality=batch.get("modality", None))
        if gt_img is not None:
            rec_img = decode_latents(self.config, self.get_vae(), gt_img)
            log({f"{split}/rec_img": wandb.Image(rec_img), **step_metrics})
            
        gt_txt = gt_txt[:4]
        if self.config.trainer.multimodal_batches:
            txt_batch = batch["input_ids"][~self.img_txt_pair_batch_mask(batch)]
            if txt_batch.shape[0] > 0:
                rprint(f"Txt Only (GT): {wrapped_batch_decode(self.tokenizer, txt_batch[:4], clean_up_tokenization_spaces=True, skip_special_tokens=True)}")
            else:
                rprint(f"GT Captions: {wrapped_batch_decode(self.tokenizer, gt_txt, clean_up_tokenization_spaces=True, skip_special_tokens=True)}")
        else:
            if gt_txt is not None:
                rprint(f"GT Captions: {wrapped_batch_decode(self.tokenizer, gt_txt, clean_up_tokenization_spaces=True, skip_special_tokens=True)}")

    if getattr(self.config.eval, "visualize_data_only", False):
        exit()

    if split == "train":
        if hasattr(self, "vae"):
            del self.vae
        empty_device_cache()


@try_except(write_error_to_file=True, clear_cuda_cache=True)
def mauve_store_references(self, dataloader):
    total_batches = len(dataloader)
    sample_batch = next(iter(dataloader))
    batch_size = sample_batch["input_ids"].shape[0]
    # only execute on rank 0
    N = self.config.eval.mauve_num_samples
    if not is_main_process():
        return
    if N is None or N <= 0 or batch_size * total_batches < N:
        rprint(f"[WARNING] Skipping Mauve reference storage. N: {N}, batch_size: {batch_size}, total_batches: {total_batches}")
        return
    # need to get N samples from dataloader, which has a batch size of batch_size
    # we need to get ceil(N / batch_size) batches
    num_batches = math.ceil(N / batch_size)
    # store in self.mauve_references
    for i, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Mauve storing references"): #, disable=not is_main_process()):
        if i >= num_batches:
            break
        reference_txt_tokens, _ = self.decode_batch(batch["input_ids"], text_only=False, sample_modality=batch.get("modality", None))
        reference_txt = wrapped_batch_decode(self.tokenizer, reference_txt_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        self.mauve_references.extend(reference_txt)
        
    assert len(self.mauve_references) >= N, f"len(self.mauve_references) ({len(self.mauve_references)}) < N ({N})"
    self.mauve_references = self.mauve_references[:N]   
    save_path = os.path.join(self.config.output_dir, f'mauve_references_{N}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(self.mauve_references, f)
    rprint(f"[MAUVE] Stored {N} references in {save_path}")    
    

@try_except(write_error_to_file=True)
def cleanup_fid_output(self):
    if getattr(self.config.eval, "force_fid_output_dir", None) is not None:
        return
    if hasattr(self, "fid_gen_dir"):
        fid_output_dir_path = Path(self.fid_gen_dir)
        if fid_output_dir_path.exists() and fid_output_dir_path.is_dir():
            rprint(f"Removing fid output dir: {fid_output_dir_path}")
            shutil.rmtree(fid_output_dir_path)

    if hasattr(self, "fid_gt_dir"):
        fid_gt_dir_path = Path(self.fid_gt_dir)
        if fid_gt_dir_path.exists() and fid_gt_dir_path.is_dir():
            rprint(f"Removing fid gt dir: {fid_gt_dir_path}")
            shutil.rmtree(fid_gt_dir_path)

    if hasattr(self, "img_to_txt_mauve_gen_dir"):
        img_to_txt_mauve_gen_dir_path = Path(self.img_to_txt_mauve_gen_dir)
        if img_to_txt_mauve_gen_dir_path.exists() and img_to_txt_mauve_gen_dir_path.is_dir():
            rprint(f"Removing img to txt mauve gen dir: {img_to_txt_mauve_gen_dir_path}")
            shutil.rmtree(img_to_txt_mauve_gen_dir_path)

    if hasattr(self, "img_to_txt_mauve_gt_dir"):
        img_to_txt_mauve_gt_dir_path = Path(self.img_to_txt_mauve_gt_dir)
        if img_to_txt_mauve_gt_dir_path.exists() and img_to_txt_mauve_gt_dir_path.is_dir():
            rprint(f"Removing img to txt mauve gt dir: {img_to_txt_mauve_gt_dir_path}")
            shutil.rmtree(img_to_txt_mauve_gt_dir_path)

def compute_val_metrics_standalone(self):
    rprint("Computing validation metrics standalone")
    self.reset_validation_metrics()
    num_samples = 0
    for i, batch in tqdm(enumerate(self.validation_dataloader), desc="Standalone validation steps", disable=not is_main_process(), leave=False):
        batch = self.update_batch(batch)
        num_samples += batch["input_ids"].shape[0]
        self.compute_loss(batch, prefix="val", batch_idx=i)
        if i >= self.config.eval.num_val_metrics_standalone_batches_per_device:
            break

    log({**self.get_step_metrics(), "num_samples": num_samples * get_world_size()})
    rprint(f"Finished computing validation metrics standalone.")


def compute_val_metrics_constant_per_batch(self):
    rprint("Computing validation metrics standalone")
    self.reset_validation_metrics()
    if self.config.eval.num_val_metrics_standalone_batches_per_device is None or self.config.eval.num_val_metrics_standalone_batches_per_device <= 0:
        return
    num_samples = 0
    for i, batch in tqdm(enumerate(self.validation_dataloader), desc="Standalone validation steps", disable=not is_main_process(), leave=False):
        batch = self.update_batch(batch)
        num_samples += batch["input_ids"].shape[0]
        self.compute_loss(batch, prefix="val", batch_idx=i)
        if i >= self.config.eval.num_val_metrics_standalone_batches_per_device:
            break

    log({**self.get_step_metrics(), "num_samples": num_samples * get_world_size()})
    rprint(f"Finished computing validation metrics standalone.")

def get_interleaved_image(self, sample_data, sample_modalities, single_image_only=False, disable_img_save=False, image_save_postfix=None):
    all_sample_imgs = []
    single_image_only = self.config.eval.auto_enhance or single_image_only or getattr(self.config.eval, "fake_interleaved", False)
    if getattr(self.config.eval, "disable_shm_save", False):
        disable_img_save = True

    if not disable_img_save:
        date_folder = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path("/dev/shm") / os.getenv("USER", 'user') / "imgs" / date_folder
        save_dir.mkdir(exist_ok=True, parents=True)

    for j in range(len(sample_data)):
        if sample_modalities[j] == 0 and not single_image_only:
            sample_text = wrapped_batch_decode(
                self.tokenizer,
                sample_data[j][None],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
                disable_mask_after_eos=True
            )
            txt_image = create_text_image(text=sample_text[0], desired_width=self.config.data.resolution)
            all_sample_imgs.append(txt_image)
        elif sample_modalities[j] == 1:
            sample_img = decode_latents(self.config, self.get_vae(), sample_data[j][None])
            all_sample_imgs.append(sample_img)

    if not disable_img_save:
        image_save_postfix = image_save_postfix or ""
        filename = f"img_{get_rank()}_{str(time.time()).replace('.', '__')}"[:100] + f"{image_save_postfix}.png"
        save_path = save_dir / filename
    if single_image_only:
        if not disable_img_save:
            gprint(Im(all_sample_imgs[0]).save(save_path))
        assert len(all_sample_imgs) == 1, "Expected single image only"
        return all_sample_imgs[0]
    else:
        img = Im.concat_vertical(*all_sample_imgs).pil
        if not disable_img_save:
            gprint(Im(img).save(save_path))
        return img
    

def get_hpsv2_score(
    self,
    images,
    prompts
):
    from unidisc.tokenizers.hpsv2_img_score import score, initialize_model
    if not hasattr(self, "hpsv2_model_dict"):
        self.hpsv2_model_dict = initialize_model(self.device, "v2.1")

    if isinstance(images, Tensor):
        images = [Im(x).pil for x in images]

    with torch.inference_mode(mode=False), torch.no_grad():
        scores = []
        for img, prompt in zip(images, prompts):
            scores.append(score(self.hpsv2_model_dict, img, prompt)[0].item())
    return torch.tensor(scores)

def get_dfn_score(
    self,
    images,
    prompts
):
    if isinstance(images, Tensor):
        images = [Im(x).pil for x in images]

    from open_clip import create_model_from_pretrained, get_tokenizer

    if not hasattr(self, "dfn_model"):
        self.dfn_model, self.dfn_preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
        self.dfn_tokenizer = get_tokenizer('ViT-H-14')
        self.dfn_model.to(str(self.device))

    assert len(images) == len(prompts), "Expected same number of images and prompts"
    images = torch.stack([self.dfn_preprocess(x) for x in images])
    text = self.dfn_tokenizer(prompts, context_length=self.dfn_model.context_length)
    dfn_dtype = next(iter(self.dfn_model.parameters())).dtype

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = self.dfn_model.encode_image(images.to(device=self.device, dtype=dfn_dtype))
        text_features = self.dfn_model.encode_text(text.to(device=self.device))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        sim = (image_features * text_features).sum(dim=-1)

    return sim


def get_clip_score(
    self,
    images,
    prompts
):

    if isinstance(images, Tensor):
        images = [Im(x).pil for x in images]

    from transformers import (
        CLIPTokenizer,
        CLIPTextModelWithProjection, 
        CLIPVisionModelWithProjection,
        CLIPImageProcessor,
    )

    if not hasattr(self, "clip_tokenizer"):
        clip_id = "openai/clip-vit-large-patch14"
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(self.device)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_id)
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(self.device)

    assert len(images) == len(prompts), "Expected same number of images and prompts"

    with torch.no_grad(), torch.cuda.amp.autocast():
        preprocessed_images = self.clip_image_processor(images, return_tensors="pt")["pixel_values"]
        image_features = self.clip_image_encoder(pixel_values=preprocessed_images.to(self.device)).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        tokenized_text = self.clip_tokenizer(
            prompts,
            max_length=self.clip_tokenizer.model_max_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        text_features = self.clip_text_encoder(input_ids=tokenized_text.input_ids.to(self.device)).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        sim = (image_features * text_features).sum(dim=-1)

    return sim

def get_laion_aesthetic_score(
    self,
    images,
    prompts
):
    from unidisc.tokenizers.laion_aesthetic_v2 import get_predictor_func
    if not hasattr(self, "laion_aesthetic_model"):
        self.laion_aesthetic_model = get_predictor_func(self.device)

    return torch.from_numpy(self.laion_aesthetic_model(images)).squeeze(-1)

def get_model_likelihood_score(self, batch, num_timesteps=100, return_unweighed=True):
    class_log_probs = []
    unweighed_class_log_probs = []
    effective_batch_size = batch['modality'].shape[0]
    empty_device_cache()
    times = torch.linspace(0, 1, steps=num_timesteps + 2)[1:-1].to(self.device).to(torch.float32)
    attention_mask = batch['attention_mask']
    
    for i in range(num_timesteps):
        empty_device_cache()
        t = times[i]
        t = t.expand(effective_batch_size)
        sigma, dsigma = self.noise(t)

        unet_conditioning = None # sigma[:, None] -> This causes CUDA OOM
        move_chance = 1 - torch.exp(-sigma[:, None])

        x0 = batch['input_ids']
        xt = self.q_xt(x0, move_chance)

        model_output = self.forward(
            xt, unet_conditioning, return_additional_loss=True, batch=batch, modality=batch['modality']
        )

        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
        log_p_theta = torch.where(attention_mask, log_p_theta, 0)
        std_weighting = (dsigma / torch.expm1(sigma))[:, None]
        unweighed_log_p_theta = -log_p_theta
        loss = -log_p_theta * std_weighting
        log_probs = loss.sum(dim=-1) / attention_mask.sum(dim=-1)
        unweighed_log_probs = unweighed_log_p_theta.sum(dim=-1) / attention_mask.sum(dim=-1)
        # print(f'Weighed loss: {log_probs.mean()} | Log P Theta: {-log_p_theta.mean()} | Std Weighting: {std_weighting.mean()}')
        class_log_probs.append(log_probs)
        unweighed_class_log_probs.append(unweighed_log_probs)

    overall_time_log_probs = torch.stack(class_log_probs) # (num_time, B)
    unweighed_overall_time_log_probs = torch.stack(unweighed_class_log_probs) # (num_time, B)

    if return_unweighed:
        return unweighed_overall_time_log_probs.mean(dim=0) # (B)
    return overall_time_log_probs.mean(dim=0) # (B)

def get_chameleon_score(self, images, prompts):
    return torch.tensor(self.calculate_chameleon_perplexity(None, None, prompts, images))

def get_text_likelihood_score(self, images, prompts):
    return self.compute_generative_perplexity(prompts, return_raw_score=True)

@torch.inference_mode()
def get_text_reward_model_score(
    self,
    images,
    prompts
):
    if not hasattr(self, "text_reward_model"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
        self.text_reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            num_labels=1,
        )
        self.text_reward_tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Please generate a realistic caption for a text-to-image generator. The caption should have proper grammar and describe a realistic scene that a user might ask for. The caption should not be non-sensical. The caption does not need to be elaborate, but should be descriptive and realistic. Penalize improper grammar and spelling."

    batch_size = 4
    formatted_conversations = []
    for resp in prompts:
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
        formatted = self.text_reward_tokenizer.apply_chat_template(conv, tokenize=False)
        formatted_conversations.append(formatted)
    
    all_scores = []
    for i in range(0, len(formatted_conversations), batch_size):
        batch_texts = formatted_conversations[i : i + batch_size]
        batch_inputs = self.text_reward_tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            batch_logits = self.text_reward_model(**batch_inputs).logits.squeeze(-1)

        all_scores.extend(batch_logits.cpu().tolist())

    return torch.tensor(all_scores).to(self.device)


def get_rewards(self, reward_config, images, prompts, batch=None, return_raw_rewards=False):
    assert isinstance(images, Tensor) and isinstance(prompts, list), "Expected images to be a Tensor and prompts to be a list"
    assert images.ndim == 4 and 0 <= images.min() and images.max() <= 1, "Expected images to be in [0, 1]"
    assert len(prompts) == images.shape[0], "Expected same number of images and prompts"
    reward_name_to_fn = dict(
        dfn_score=self.get_dfn_score,
        clip_score=self.get_clip_score,
        hpsv2_score=self.get_hpsv2_score,
        laion_aesthetic_score=self.get_laion_aesthetic_score,
        model_likelihood_score=self.get_model_likelihood_score,
        chameleon_score=self.get_chameleon_score,
        text_likelihood_score=self.get_text_likelihood_score,
        text_reward_model_score=self.get_text_reward_model_score
    )

    rewards = []
    raw_rewards = dict()
    for reward_name, reward_weight in reward_config.items():
        start_time = time.time()
        assert reward_name in reward_name_to_fn, f"Invalid reward name: {reward_name}"
        reward_fn = reward_name_to_fn[reward_name]
        if reward_name == "model_likelihood_score" or reward_name == "chameleon_score" or reward_name == "text_likelihood_score":
            assert batch is not None, "Expected batch to be provided for model likelihood score"
            if reward_name == "chameleon_score" or reward_name == "text_likelihood_score":
                reward = reward_fn(images, prompts).cpu()
            else:
                reward = reward_fn(batch=batch).cpu()
            raw_rewards[reward_name] = reward
            rprint(f"Orig {reward_name}: {reward}")
            reward = -reward
            reward = (reward - reward.min()) / (reward.max() - reward.min())
            rprint(f"Normalized {reward_name}: {reward}")
        else:
            reward = reward_fn(images, prompts).cpu()
            raw_rewards[reward_name] = reward
            # reward = reward.softmax(dim=-1)
            reward = (reward - reward.min()) / (reward.max() - reward.min())

        reward = torch.nan_to_num(reward, nan=0.0)
        rewards.append(reward * reward_weight)
        print(f"Processed {reward_name} in {time.time() - start_time:.2f} seconds")

    rewards = torch.stack(rewards, dim=-1).sum(dim=-1)

    if return_raw_rewards:
        return rewards, raw_rewards

    return rewards

def clear_reward_models(self):
    if hasattr(self, "laion_aesthetic_model"):
        del self.laion_aesthetic_model
    if hasattr(self, "dfn_model"):
        del self.dfn_model
    if hasattr(self, "dfn_tokenizer"):
        del self.dfn_tokenizer
    if hasattr(self, "clip_tokenizer"):
        del self.clip_tokenizer
    if hasattr(self, "clip_text_encoder"):
        del self.clip_text_encoder
    if hasattr(self, "clip_image_processor"):
        del self.clip_image_processor
    if hasattr(self, "clip_image_encoder"):
        del self.clip_image_encoder
    if hasattr(self, "text_reward_model"):
        del self.text_reward_model
    if hasattr(self, "text_reward_tokenizer"):
        del self.text_reward_tokenizer
    if hasattr(self, "hpsv2_model_dict"):
        del self.hpsv2_model_dict

def auto_enhance(self, batch):
    gprint(f"Auto enhancing")
    from dataloader import tokenize_text
    assert isinstance(batch, TensorDict), "Expected batch to be a TensorDict"
    batch = batch.squeeze(1)
    assert batch['input_ids'].ndim == 2, "Expected batch to be 2D"

    # from datasets import load_dataset
    # dataset = load_dataset("nateraw/parti-prompts", split='train')
    # dataset = dataset.filter(lambda x: x["Category"] == "Artifacts")

    x0 = batch["input_ids"].clone()
    add_object = getattr(self.config.eval, "auto_enhance_add_object", False)
    if add_object:
        img_tokens = x0[:, self.static_img_sl] - self.text_vocab_size
        assert 0 <= img_tokens.min() and img_tokens.max() <= self.image_vocab_size, "Expected img tokens to be in [0, img_vocab_size]"
        orig_imgs = decode_latents(self.config, self.get_vae(), img_tokens)
        orig_imgs = [Im(img).pil for img in orig_imgs]
        aug_imgs = [augment_image_with_random_object_coco(img, str(UNIDISC_DIR / "archive" / "objects")) for img in orig_imgs]
        gprint(f"Augmented {len(aug_imgs)} images")
        aug_imgs = torch.stack([Im(img).torch for img in aug_imgs]).to(self.device)
        image_ids = get_image_batch(self.config, self.get_vae(), {"img": aug_imgs}, self.device)
        x0[:, self.static_img_sl] = image_ids + self.text_vocab_size

    gen_batch = batch.clone()
    if 'interleaved_metadata' in gen_batch:
        del gen_batch['interleaved_metadata']
    gen_batch.auto_batch_size_()

    orig_caption = wrapped_batch_decode(self.tokenizer, batch['input_ids'][:, self.static_txt_sl], clean_up_tokenization_spaces=True, skip_special_tokens=True, disable_mask_after_eos=True)

    max_num_augmentations = getattr(self.config.eval, "max_num_auto_enhance_augmentations", 10)

    llm_func = get_llm(llm_model_type="")
    llm_augmented_captions = [llm_func(cap, fake_openai_failure=False)[0] for cap in orig_caption]
    _augmented_captions = []
    for caps in llm_augmented_captions:
        _shuf = deepcopy(caps)
        random.shuffle(_shuf)
        assert len(_shuf) >= max_num_augmentations, "Expected at least max_num_augmentations augmentations"
        _augmented_captions.append(_shuf[:max_num_augmentations])
    gprint(f"Augmented {len(_augmented_captions)} captions")

    _orig_imgs = Im(decode_latents(self.config, self.get_vae(), x0[:, self.static_img_sl] - self.text_vocab_size)).pil
    if not isinstance(_orig_imgs, list):
        _orig_imgs = [_orig_imgs]

    num_iter_per_sample = self.config.eval.num_auto_enhance_iter 
    num_iter = num_iter_per_sample * max_num_augmentations
    bs = 1
    n = num_iter * bs * len(_augmented_captions)
    _gen_batch = []
    for i in range(len(_augmented_captions)):
        for j in range(num_iter):
            _gen_batch.append(gen_batch[[i]])
    gen_batch = torch.cat(_gen_batch, dim=0)

    txt_data = [tokenize_text(self.tokenizer, self.config.data.block_size, caps) for caps in _augmented_captions]
    txt_sl = slice(None, self.config.data.block_size)
    real_captions = []
    augmented_captions = []
    orig_images = []

    gprint(f"Generating {num_iter} samples, gen_batch shape: {gen_batch.shape}")

    for j in range(len(_augmented_captions)):
        for k in range(max_num_augmentations):
            sl = slice(j * max_num_augmentations + k * num_iter_per_sample, j * max_num_augmentations + (k + 1) * num_iter_per_sample)
            gen_batch[sl]['input_ids'][:, txt_sl] = txt_data[j]['input_ids'][k]
            gen_batch[sl]['attention_mask'][:, txt_sl] = txt_data[j]['attention_mask'][k]
            augmented_captions.extend([_augmented_captions[j][k]] * num_iter_per_sample)
            real_captions.extend([orig_caption[j]] * num_iter_per_sample)
            orig_images.extend([_orig_imgs[j]] * num_iter_per_sample)

    # min_val, max_val = 0.94, 0.98
    # _eps_t = torch.rand(n, device=self.device)
    # offset = torch.arange(n, device=self.device) / n
    # _eps_t = (_eps_t / n + offset) % 1
    # t = (max_val - min_val) * _eps_t + min_val

    if getattr(self.config.eval, "auto_enhance_use_low_masking", False):
        mean_txt, std_txt = 0.85, 0.2 / 0.8416  # First half
        mean_img, std_img = 0.75, 0.04 / 1.645  # Second half - higher mean = more masking
    else:
        mean_txt, std_txt = 0.85, 0.2 / 0.8416  # First half
        mean_img, std_img = 0.95, 0.04 / 1.645  # Second half - higher mean = more masking

    def slice_len(_sl, _seq_len):
        # TODO: This is super incorrect
        assert _sl.step is None
        if _sl.start is not None and _sl.start < 0:
            assert _sl.stop is None
            return -_sl.start
        else:
            return (_sl.stop if _sl.stop is not None else _seq_len) - (_sl.start if _sl.start is not None else 0)
    
    seq_len = x0.shape[1]

    t = torch.zeros((n,), device=self.device)
    t = t.to(torch.float32)
    
    t_txt = torch.normal(mean=mean_txt, std=std_txt, size=(n,), device=self.device)
    t_img = torch.normal(mean=mean_img, std=std_img, size=(n,), device=self.device)
    
    t_txt = torch.clamp(t_txt, max=1.0)
    t_img = torch.clamp(t_img, max=1.0)
    move_indices = torch.zeros(n, seq_len, device=self.device, dtype=torch.bool)
    
    move_indices[:, self.static_txt_sl] = torch.rand(move_indices.shape[0], slice_len(self.static_txt_sl, seq_len), device=self.device) < t_txt.unsqueeze(1)
    move_indices[:, self.static_img_sl] = torch.rand(move_indices.shape[0], slice_len(self.static_img_sl, seq_len), device=self.device) < t_img.unsqueeze(1)
    
    x0_unmask = ~move_indices
    rprint(f"Text masking ratio: {move_indices[:, self.static_txt_sl].sum() / move_indices[:, self.static_txt_sl].numel():.3f}")
    rprint(f"Image masking ratio: {move_indices[:, self.static_img_sl].sum() / move_indices[:, self.static_img_sl].numel():.3f}")
    rprint(f"Num unmasked: {x0_unmask.sum(dim=-1).float().mean():.1f}")

    text_samples_list = []
    img_samples_list = []

    x0 = x0.to(self.device)
    x0_unmask = x0_unmask.to(self.device)
    
    idx = 0
    for i in range(len(_augmented_captions)):
        for j in range(num_iter_per_sample):
            _modality = gen_batch[[idx]].get("modality", None)
            _sample_ids = gen_batch[[idx]].get("sample_ids", None)
            if _modality is not None:
                _modality = _modality.to(self.device)
            if _sample_ids is not None:
                _sample_ids = _sample_ids.to(self.device)
            else:
                _sample_ids = torch.zeros_like(_modality)
            text_samples, img_samples, x = self._sample(
                text_only=False,
                num_steps=self.config.sampling.max_sampling_steps,
                batch_size_per_gpu=bs,
                modality=_modality,
                sample_ids=_sample_ids,
                x0=gen_batch["input_ids"][[idx]].to(self.device),
                x0_unmask=x0_unmask[[idx]].to(self.device),
                return_raw_data=True,
                allow_interleaved_conditional=True
            )
            gen_batch[[idx]]['input_ids'] = x
            text_samples_list.extend(text_samples)
            img_samples_list.extend(img_samples)
            rprint(f"Sampled {j + 1} / {num_iter}")
            idx += 1

    # gen_batch = torch.cat([gen_batch, orig_batch], dim=0)
    # for i in range(orig_batch.shape[0]):
    #     _modality = orig_batch[[i]].get("modality", None)
    #     _sample_ids = orig_batch[[i]].get("sample_ids", None)
    #     if _modality is not None:
    #         _modality = _modality.to(self.device)
    #     if _sample_ids is not None:
    #         _sample_ids = _sample_ids.to(self.device)
    #     else:
    #         _sample_ids = torch.zeros_like(_modality)
    #     res = self.decode_sampling(
    #         orig_batch[[i]]["input_ids"].to(self.device),
    #         text_only=False,
    #         modality=_modality,
    #         sample_ids=_sample_ids
    #     )
    #     text_samples_list.extend(res[0])
    #     img_samples_list.extend(res[1])
    #     augmented_captions.append(orig_caption[i])
    #     real_captions.append(orig_caption[i])
    #     orig_images.append(orig_imgs[i])

    text_samples_list = wrapped_batch_decode(
        self.tokenizer,
        torch.stack(text_samples_list, dim=0),
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True,
        disable_mask_after_eos=True
    )

    # for i in range(len(text_samples_list) - orig_batch.shape[0], len(text_samples_list)):
    #     text_samples_list[i] = "Original: " + text_samples_list[i]

    img_samples_list = torch.cat(img_samples_list, dim=0)

    reward_config = self.config.eval.auto_enhance_reward_config
    rewards, raw_rewards = self.get_rewards(reward_config, img_samples_list, text_samples_list, batch=gen_batch, return_raw_rewards=True)

    gprint(f"Avg Rewards: {rewards}")

    sorted_indices = torch.argsort(rewards, descending=True).tolist()
    sorted_text_samples = [text_samples_list[i] for i in sorted_indices]
    sorted_augmented_captions = [augmented_captions[i] for i in sorted_indices]
    sorted_real_captions = [real_captions[i] for i in sorted_indices]
    sorted_img_samples = [img_samples_list[i] for i in sorted_indices]
    sorted_orig_images = [orig_images[i] for i in sorted_indices]
    sorted_avg_rewards = [rewards[i] for i in sorted_indices]
    sorted_raw_rewards = {k: [raw_rewards[k][i] for i in sorted_indices] for k in raw_rewards}

    text_samples_list = sorted_text_samples
    real_captions = sorted_real_captions
    augmented_captions = sorted_augmented_captions
    img_samples_list = sorted_img_samples
    orig_images = sorted_orig_images
    raw_rewards = sorted_raw_rewards

    # clear all reward models
    self.clear_reward_models()

    log_dict = {}
    with try_except(write_error_to_file=True):
        if text_samples_list is not None:
            gprint(f"Gathering {len(text_samples_list)} text samples")
            text_samples_list = gather_object(text_samples_list)

            real_captions = gather_object(real_captions)
            augmented_captions = gather_object(augmented_captions)
            prefix = "auto_enhance"

            if isinstance(img_samples_list, Tensor): img_samples_list = img_samples_list.float().cpu()
            img_samples_list = [Im(img).pil for img in img_samples_list]
            img_samples_list = gather_object(img_samples_list)
            orig_images = gather_object(orig_images)

            dprint(f"Gathered {len(text_samples_list)} text samples")

            new_sorted_avg_rewards = gather_object(sorted_avg_rewards)
            sorted_avg_rewards = new_sorted_avg_rewards

            new_raw_rewards = {k: gather_object(v) for k, v in raw_rewards.items()}
            raw_rewards = new_raw_rewards
            rprint(f"Finished gathering, length: {len(orig_images)}")

            gen_table = wandb.Table(columns=[f"real_caption", f"original_image", f"augmented_caption", f"sampled_caption", f"sampled_image", f"avg_reward", *reward_config.keys()])
            assert len(img_samples_list) == len(text_samples_list) == len(augmented_captions) == len(real_captions) == len(sorted_avg_rewards)
            for real_caption, orig_img, augmented_caption, sampled_caption, sampled_img, avg_reward, *rewards in zip(real_captions, orig_images, augmented_captions, text_samples_list, img_samples_list, sorted_avg_rewards, *raw_rewards.values()):
                gen_table.add_data(real_caption, wandb.Image(Im(orig_img).pil), augmented_caption, sampled_caption, wandb.Image(Im(sampled_img).pil), avg_reward, *rewards)

            log_dict[f"{prefix}_sample_table"] = gen_table

    log({**log_dict, **self.get_step_metrics()})

def save_image_text_pair(self, image_tensor, text_tensor, single_image_only=False, disable_img_save=False, image_save_postfix=None):
    """
    Take separate image and text tensors and save them as paired visualizations.
    
    Args:
        image_tensor: Tensor [B, N] of image tokens
        text_tensor: Tensor [B, M] of text tokens
        single_image_only: If True, only return the image without text visualization
        disable_img_save: If True, don't save to disk
        image_save_postfix: Optional postfix for the saved image filename
    
    Returns:
        PIL Image or tensor of concatenated images and text visualizations
    """
    batch_size = image_tensor.shape[0]
    assert batch_size == text_tensor.shape[0], "Batch sizes must match between image and text tensors"
    
    all_paired_imgs = []
    
    # Check config settings for single_image_only
    if hasattr(self, 'config') and hasattr(self.config, 'eval'):
        single_image_only = self.config.eval.auto_enhance or single_image_only or getattr(self.config.eval, "fake_interleaved", False)
    
    if hasattr(self, 'config') and hasattr(self.config.eval, "disable_shm_save"):
        disable_img_save = disable_img_save or getattr(self.config.eval, "disable_shm_save", False)

    # Create save directory if needed
    if not disable_img_save:
        date_folder = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path("/dev/shm") / os.getenv("USER", 'user') / "paired_imgs" / date_folder
        save_dir.mkdir(exist_ok=True, parents=True)

    for i in range(batch_size):
        pair_imgs = []
        
        # Process text (if not in single_image_only mode)
        if not single_image_only:
            sample_text = wrapped_batch_decode(
                self.tokenizer,
                text_tensor[i:i+1],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=False,
                disable_mask_after_eos=True
            )
            txt_image = create_text_image(text=sample_text[0], desired_width=self.config.data.resolution)
            pair_imgs.append(txt_image)
        
        # Process image
        img_tokens = image_tensor[i:i+1]
        sample_img = decode_latents(self.config, self.get_vae(), img_tokens)
        pair_imgs.append(sample_img)
        
        # Combine text and image for this pair
        if single_image_only:
            all_paired_imgs.append(pair_imgs[0])
        else:
            paired_img = Im.concat_vertical(*pair_imgs).pil
            all_paired_imgs.append(paired_img)

    # Save images if needed
    if not disable_img_save:
        image_save_postfix = image_save_postfix or ""
        for i, img in enumerate(all_paired_imgs):
            filename = f"pair_{get_rank()}_{i}_{str(time.time()).replace('.', '__')}"[:100] + f"{image_save_postfix}.png"
            save_path = save_dir / filename
            gprint(Im(img).save(save_path))
    
    # Return either a single image or all as list
    if batch_size == 1:
        return all_paired_imgs[0]
    else:
        return all_paired_imgs