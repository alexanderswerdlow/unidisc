import math
import random
import types
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import cached_property, partial
from contextlib import ExitStack

from numpy import mask_indices
from unidisc.utils.tensor_utils import get_contiguous_blocks, get_contiguous_blocks_per_sample, get_interleaved_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.utils import gather, gather_object
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor, nn
from tqdm.auto import tqdm

import model_eval
import model_setup
import model_utils
import utils
from decoupled_utils import (Profiler, barrier, dprint, get_rank, get_world_size, gprint,
                             is_local_main_process, is_main_process,
                             is_torch_cuda_available, is_torch_xla_available,
                             print_memory, rprint, save_memory_profile,
                             synchronize_device, try_except, use_dist)
from unidisc.tokenizers.image_tokenizers import (decode_latents, get_image_batch,
                                              get_vae, vae_encode_image)
from unidisc.utils.cuda_utils import sync_times
from unidisc.utils.xla_utils import shard_output
from model_utils import (Loss, ddprint, ema_update, empty_device_cache, get_chameleon_txt_indices, get_interleaved_block_mask, log,
                         replace_nan_dict, update_histogram, update_logs, get_block_mask)
from unidisc.utils.trainer_utils import TrainingState, incremental_dict_update, linear_warmup

is_xla_available = is_torch_xla_available()

if is_xla_available:
    import torch_xla
    from torch_xla.distributed.spmd import XLAShardedTensor




def maybe_unwrap(t: torch.Tensor) -> torch.Tensor:
    return t.global_tensor if isinstance(t, XLAShardedTensor) else t

class Diffusion:
    def __init__(self, config, tokenizer, device, disable_init=False):
        super().__init__()
        setup_methods = [
            'init', 'to', 'get_params', 'get_vae', 'get_cond_vae', 'configure_optimizers',
            '_validate_configuration', 'register_signal_handler', 'on_train_start',
            'optimizer_step', 'init_dataloader', 'set_accelerator', 'set_callbacks', 
            'on_train_step_end', 'init_optimizer_lr_scheduler', 'after_backward', 'checkpoint', 
            'print_hashes', 'shortcut_return', 'reset_validation_metrics', 'unwrap_model'
        ]
        for method_name in setup_methods:
            setattr(self, method_name, types.MethodType(getattr(model_setup, method_name), self))

        utils_methods = [
            'get_coord_plot', '_score_entropy', 'sample_subs_guidance',
            'restore_model_and_semi_ar_sample', '_reconstruction_loss',
            'restore_model_and_sample', 'get_score', '_staggered_score',
            '_analytic_update', '_denoiser_update', '_transp_transition',
            'eval_retokenize', 'compute_generative_perplexity', '_d3pm_loss',
            '_d3pm_parameterization', '_sedd_parameterization',
            'get_base_shapes_for_mup', 'update_histogram', '_maybe_sub_sample',
             'viz_images_from_dataloader', 'compute_cider'
        ]
        for method_name in utils_methods:
            setattr(self, method_name, types.MethodType(getattr(model_utils, method_name), self))

        eval_methods = [
            'get_every_n_evals', 'on_validation_epoch_start', 'sample',
            'predict_step', 'validation_step', 'on_validation_epoch_end',
            'on_validation_epoch_cleanup', '_sample_prior', '_ddpm_forward',
            '_ddpm_update', '_ddpm_caching_update', '_sample', '_ar_sampler',
            'decode_batch', 'sample_transfusion', 'sample_continuous_image',
            'decode_sampling', '_ddpm_update_finetune_controlled_tweedie', 
            'sample_masking', 'log_flops', "visualize_samples", "_maskgit_update", 
            "_first_hitting_update", "update_inline_fid", "compute_inline_fid",
            "update_clean_fid", "compute_clean_fid_eval", "sample_for_fid",
            "compute_clip_score", "mauve_store_references", "zero_shot_eval_step",
            "zero_shot_eval_epoch_end", "get_cfg_weight", "cleanup_fid_output",
            "calculate_chameleon_perplexity", "get_anole_data",
            "update_img_to_txt_mauve_clip", "compute_mauve_entropy",
            "get_top_k", "compute_entropy", "get_mauve_score", "get_valid_seq", "gather_tokens",
            "count_valid_tokens", "compute_val_metrics_standalone", "_maskgit_nucleus_update",
            "get_img_text_saturation_batch", "handle_interleaved_decode", "get_interleaved_image",
            "auto_enhance", "get_clip_score", "get_dfn_score", "get_hpsv2_score", "get_model_likelihood_score",
            "get_laion_aesthetic_score", "get_rewards", "get_chameleon_score", "clear_reward_models",
            "get_text_likelihood_score", "get_text_reward_model_score", "save_image_text_pair"
        ]
        for method_name in eval_methods:
            setattr(self, method_name, types.MethodType(getattr(model_eval, method_name), self))

        if disable_init:
            pass
        else:
            model_setup.init(self, config, tokenizer, device)

    @cached_property
    def xla_mesh(self):
        import torch_xla.distributed.spmd as xs
        return xs.get_global_mesh()

    def on_train_resume(self):
        if not is_torch_xla_available():
            empty_device_cache()

            if self.ema is not None and not self.config.trainer.use_custom_ema:
                self.ema.restore(self.get_params(), raise_error_if_already_restored=False)

            self.backbone.train()
            
    def zero_shot_update_batch(self, batch):
        dataset = self.config.data.train
        if dataset is None:
            return batch

        def get_attr(attr_name):
            return getattr(self.config.model, attr_name, None)

        if dataset == "nlphuji/flickr30k":
            # image captioning dataset
            # above thing but order is [txt, img]
            batch['gt_input_ids'] = batch['input_ids']
            image_input_ids = get_image_batch(self.config, self.get_vae(), batch, self.device)
            image_input_ids += self.text_vocab_size
            batch["input_ids"] = torch.cat([torch.zeros_like(batch['gt_input_ids'], dtype=torch.int64), image_input_ids], dim=-1).to(self.device)
            batch['attention_mask'] = torch.cat([torch.zeros_like(batch['gt_input_ids'], dtype=torch.bool), torch.ones_like(image_input_ids, dtype=torch.bool)], dim=-1).to(self.device)
            batch["modality"] = torch.cat([torch.zeros_like(batch['gt_input_ids'], dtype=torch.int64), torch.ones_like(image_input_ids, dtype=torch.int64)], dim=-1).to(self.device)
        elif dataset == "facebook/winoground":
            # get image and text input ids
            caption_0_input_ids = batch['caption_0_input_ids']
            caption_1_input_ids = batch['caption_1_input_ids']
            image_0 = batch['img_0']
            image_1 = batch['img_1']
            # tokenize and store captions separately
            image_0_input_ids = vae_encode_image(self.config, self.get_vae(), image_0, self.device, get_attr("vae_type")) + self.text_vocab_size
            image_1_input_ids = vae_encode_image(self.config, self.get_vae(), image_1, self.device, get_attr("vae_type")) + self.text_vocab_size
            # make 4 combinat ions of image and text
            batch['input_ids_0_0'] = torch.cat([caption_0_input_ids, image_0_input_ids], dim=-1).to(self.device)
            batch['input_ids_0_1'] = torch.cat([caption_0_input_ids, image_1_input_ids], dim=-1).to(self.device)
            batch['input_ids_1_0'] = torch.cat([caption_1_input_ids, image_0_input_ids], dim=-1).to(self.device)
            batch['input_ids_1_1'] = torch.cat([caption_1_input_ids, image_1_input_ids], dim=-1).to(self.device)
            batch['attention_mask'] = torch.cat([torch.zeros_like(caption_0_input_ids, dtype=torch.bool), torch.ones_like(image_0_input_ids, dtype=torch.bool)], dim=-1).to(self.device)
            batch['modality'] = torch.cat([torch.zeros_like(caption_0_input_ids, dtype=torch.int64), torch.ones_like(image_0_input_ids, dtype=torch.int64)], dim=-1).to(self.device)
        # elif dataset == "facebook/winoground":
        batch["modality_mask"] = F.one_hot(batch["modality"], num_classes=2).to(torch.bool)
        return batch

    def update_batch(self, batch):
        if getattr(self.config.eval, 'big_seq_len_eval', False):
            # new batch of 8192 seq length with txt length 4096 and img length 4096s
            N = self.config.model.length
            new_batch = dict()
            new_batch['input_ids'] = torch.zeros(batch['input_ids'].shape[0], N, device=self.device, dtype=batch['input_ids'].dtype)
            new_batch['attention_mask'] = torch.ones(batch['attention_mask'].shape[0], N, device=self.device, dtype=batch['attention_mask'].dtype)
            new_batch['modality'] = torch.zeros(batch['modality'].shape[0], N, device=self.device, dtype=batch['modality'].dtype)
            new_batch['modality'][:, N//2:] = 1
            new_batch['modality_mask'] = F.one_hot(new_batch['modality'], num_classes=2).to(torch.bool)
            batch = new_batch
            return batch
        
        continuous_mode = self.config.trainer.image_mode == "continuous"
        if batch is None:
            gprint(f"Warning! Batch is None")
            return batch
        
        if isinstance(batch, TensorDict):
            batch.batch_size = (batch.batch_size[0],)
        
        if self.image_model or getattr(self.config.data, "force_image_dataset", False):
            text_input_ids = None
            if isinstance(batch, TensorDict) and (self.is_compiled or getattr(self.config.trainer, "force_convert_to_dict", False)):
                batch = dict(batch.items())

            if "txt_input_ids" in batch or "img_input_ids" in batch:
                index_keys = ["img_input_ids", "txt_input_ids", "sample_ids"]
                for key in index_keys:
                    if key in batch:
                        if isinstance(batch[key], list):
                            batch[key] = torch.stack(batch[key], dim=0)
                        batch[key] = batch[key].to(torch.int64)

                index_keys = ["img_label"]
                for key in index_keys:
                    if key in batch:
                        batch[key] = batch[key].squeeze(-1)

                img_input_ids = batch.pop("img_input_ids")
                batch["input_ids"] = img_input_ids
                batch["attention_mask"] = torch.ones_like(img_input_ids).to(torch.bool)
                if "txt_input_ids" in batch:
                    batch["input_ids"] = torch.cat([batch["txt_input_ids"], batch["input_ids"] + self.text_vocab_size], dim=-1)
                    batch["attention_mask"] = torch.cat([batch["txt_attention_mask"], batch["attention_mask"]], dim=-1)

                batch["input_ids"] = batch["input_ids"].to(torch.int64)

                if "modality" not in batch:
                    if getattr(self.config.trainer, "ignore_text_in_unified", False):
                        modality = torch.ones_like(batch["input_ids"], dtype=torch.int64)
                    else:
                        assert self.config.model.txt_length > 0 and self.config.model.img_length > 0
                        modality = torch.zeros_like(batch["input_ids"], dtype=torch.int64)
                        modality[:, -img_input_ids.shape[-1]:] = 1
                    batch["modality"] = modality

            elif (self.config.trainer.multimodal_batches or continuous_mode) and \
                not getattr(self.config.trainer, "use_legacy_update_batch_fn", False):

                if "img" in batch:
                    is_image_batch = (batch["modality"] == 1).all(dim=-1)
                    image_input_ids = get_image_batch(self.config, self.get_vae(), batch, self.device)
                    assert ((batch["modality"].sum(dim=-1) == 0) | (batch["modality"].sum(dim=-1) >= image_input_ids.shape[1])).all()

                    if getattr(self.config.trainer, "add_label", False):
                        assert (batch["modality"] == 1).all()
                        batch["input_ids"][:, 1:] = torch.where(is_image_batch[:, None], image_input_ids, batch["input_ids"][:, 1:])
                    elif image_input_ids.ndim == 3:
                        batch["img_emb"] = torch.where((batch["modality"] == 1)[:, :, None], image_input_ids, torch.nan)
                    elif (batch["input_ids"][batch["modality"] == 1] == -1).all():
                        batch["input_ids"].masked_scatter_(batch["modality"] == 1, image_input_ids)
                    else:
                        batch["input_ids"] = torch.where(is_image_batch[:, None], image_input_ids, batch["input_ids"])

                    if getattr(self.config.trainer, "force_shift_raw_image_batches", False):
                        assert not getattr(self.config.trainer, "force_shift_image_batches", False)
                        batch["input_ids"] = torch.where(batch["modality"] == 1, batch["input_ids"] + self.text_vocab_size, batch["input_ids"])
                else:
                    if getattr(self.config.trainer, "add_label", False):
                        shift_index = self.vocab_size - self.config.model.add_labels
                        batch["input_ids"] = torch.cat([batch["label"] + shift_index, batch["input_ids"]], dim=-1)
                        batch["attention_mask"] = torch.cat([torch.zeros_like(batch["label"], dtype=torch.bool), batch["attention_mask"]], dim=-1)
                        batch["modality"] = torch.cat([torch.ones_like(batch["label"], dtype=torch.int64), batch["modality"]], dim=-1)
                        assert (batch["modality"] == 1).all()

                batch["input_ids"] = batch["input_ids"].to(torch.int64)
                if "sample_ids" in batch:
                    batch["sample_ids"] = batch["sample_ids"].to(torch.int64)

                if getattr(self.config.trainer, "force_shift_image_batches", False):
                    batch["input_ids"] = torch.where(batch["modality"] == 1, batch["input_ids"] + self.text_vocab_size, batch["input_ids"])
            else:
                if continuous_mode:
                    assert False
                else:
                    if "input_ids" in batch and not self.config.trainer.ignore_text_in_unified:
                        assert self.config.model.unified_model
                        assert "attention_mask" in batch
                        text_input_ids = batch["input_ids"]

                    image_ids = get_image_batch(self.config, self.get_vae(), batch, self.device)
                    image_attention_mask = torch.ones_like(image_ids).to(torch.bool)

                    if "cond_img" in batch:
                        cond_image_ids = get_image_batch(self.config, self.get_cond_vae(), batch, self.device, use_cond=True)
                        batch["cond_input_ids"] = cond_image_ids

                    if text_input_ids is not None:
                        assert batch["input_ids"].shape[1] == self.config.model.txt_length
                        assert image_ids.shape[1] == self.config.model.img_length
                        image_ids = image_ids + self.text_vocab_size

                        batch["input_ids"] = torch.cat([batch["input_ids"].to(self.device), image_ids], dim=-1)
                        batch["attention_mask"] = torch.cat([batch["attention_mask"].to(self.device), image_attention_mask], dim=-1).to(torch.bool)
                        assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1] == self.config.model.length
                        batch["modality"] = torch.zeros_like(batch["input_ids"], dtype=torch.int64)
                        batch["modality"][:, -image_ids.shape[-1]:] = 1
                    else:
                        assert self.unified_model is False
                        batch["input_ids"] = image_ids
                        batch["attention_mask"] = image_attention_mask
                        batch["modality"] = torch.ones_like(batch["input_ids"], dtype=torch.int64)

            if "txt_x0_unmask" in batch and "img_x0_unmask" in batch:
                assert not continuous_mode
                batch["gt_img_input_ids"] = image_ids
                batch["x0_unmask"] = torch.cat([batch["txt_x0_unmask"], batch["img_x0_unmask"]], dim=-1)
                batch["input_ids"][~batch["x0_unmask"]] = self.mask_index

            if (batch["input_ids"].shape[1] != self.config.model.length) and not self.config.trainer.ar_inpainting:
                gprint(f"Warning! Input ids shape: {batch['input_ids'].shape}, model length: {self.config.model.length}")
                batch["input_ids"] = batch["input_ids"][:, : self.config.model.length]
                assert False, f"input ids are not the correct length input ids shape: {batch['input_ids'].shape}, model length: {self.config.model.length}"

        if getattr(self.config.model, "img_cond", False):
            assert "cond_input_ids" in batch
            assert not continuous_mode

        if "modality" in batch:
            batch["modality"] = batch["modality"].to(torch.int64)
            if self.config.trainer.multimodal_batches and batch["modality"].ndim == 2 and batch["modality"].shape[-1] == 1:
                batch["modality"] = batch["modality"].repeat(1, self.config.model.length)
        else:
            if self.image_model and not self.config.trainer.multimodal_batches:
                assert self.config.model.txt_length > 0 and self.config.model.img_length > 0
                modality = torch.zeros_like(batch["input_ids"], dtype=torch.int64)
                modality[:, self.static_img_sl] = 1
                batch["modality"] = modality
            elif self.config.data.txt_only:
                batch["modality"] = torch.zeros_like(batch["input_ids"], dtype=torch.int64)

        if "modality" in batch:
            batch["modality"][batch["modality"] == -1] = 0
            assert batch["modality"].min() == 0 and batch["modality"].max() == 1
            batch["modality_mask"] = F.one_hot(batch["modality"], num_classes=2).to(torch.bool)
            batch["batch_contains_img"] = (batch["modality"] == 1).any(dim=-1)
            batch['txt_sl'] = self.txt_sl(batch)
            batch['img_sl'] = self.img_sl(batch)

        if getattr(self.config.trainer, "force_remove_img_tokens", False):
            assert not continuous_mode
            batch["input_ids"] = batch["input_ids"][batch['txt_sl']]
            batch["attention_mask"] = batch["attention_mask"][batch['txt_sl']]

        if getattr(self.config.trainer, "add_label", False):
            assert getattr(self.config.model, "add_labels", False)
            assert "label" in batch
            batch["label"] = batch["label"].to(torch.int64)
            assert 0 <= batch["label"].min() and batch["label"].max() < self.config.model.add_labels
            shift_index = self.vocab_size - self.config.model.add_labels

            assert batch["input_ids"].shape[-1] == self.config.model.length
            if batch["label"].ndim == 1:
                batch["input_ids"][:, [0]] = (batch["label"] + shift_index).unsqueeze(-1)
            else:
                batch["input_ids"][:, [0]] = batch["label"] + shift_index

            batch["attention_mask"][:, 0] = False

        if isinstance(batch, dict):
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
        elif isinstance(batch, TensorDict):
            assert self.config.backbone != "gemma"
            batch = batch.to(self.device)

        if getattr(self.config.trainer, "force_full_attention_mask", False):
            batch["attention_mask"] = torch.ones_like(batch["attention_mask"], dtype=torch.bool)

        batch["attention_mask"] = batch["attention_mask"].to(torch.bool)

        if self.config.data.require_sample_ids:
            assert "sample_ids" in batch
            batch["sample_ids"][~(batch["attention_mask"].bool())] = -1
            batch["attention_mask"][batch["sample_ids"] == -1] = False

        # Flip [txt, img] -> [img, txt]
        # TODO: Flip by sample not batch. As we train w/~8 batches, it's for now
        if (self.training or getattr(self.config.trainer, "force_flip_ar_val", False)) and self.config.parameterization == "ar" and getattr(self.config.trainer, "rand_flip_ar_prob", None) is not None:
            assert (batch["modality"][:, :self.config.model.txt_length] == 0).all() and (batch["modality"][:, self.config.model.txt_length:] == 1).all(), "Modality does not match img_before_txt configuration"
            batch_flip_mask = torch.rand(batch["modality"].shape[0], device=self.device) < self.config.trainer.rand_flip_ar_prob
            img_slice = slice(-self.config.model.img_length, None)
            txt_slice = slice(None, self.config.model.txt_length)

            for key in ["modality", "attention_mask", "input_ids"]:
                batch[key][batch_flip_mask] = torch.cat([batch[key][batch_flip_mask][:, img_slice], batch[key][batch_flip_mask][:, txt_slice]], dim=1)

            if "modality_mask" in batch:
                batch["modality_mask"] = F.one_hot(batch["modality"], num_classes=2).to(torch.bool)

            batch['txt_sl'] = None
            batch['img_sl'] = None
            batch["batch_flip_mask"] = batch_flip_mask

        if self.config.trainer.interleaved and "sample_ids" not in batch:
            batch["sample_ids"] = torch.zeros_like(batch["modality"], dtype=torch.int64)

        if self.config.trainer.interleaved:
            batch_indices, start_positions, end_positions = get_contiguous_blocks(batch["modality"])
            interleaved_metadata = TensorDict({
                "batch_indices": batch_indices,
                "start_positions": start_positions,
                "end_positions": end_positions
            }, batch_size=[])
            allowed_image_sizes = (64, 256, 1024, 2304, 4096)
            block_sizes = (end_positions - start_positions).to(torch.int32)
            is_txt_block = batch["modality"][batch_indices, start_positions] == 0
            is_valid_img_size = torch.isin(block_sizes, torch.tensor(allowed_image_sizes, dtype=torch.int32, device=self.device))

            if not ((is_txt_block | is_valid_img_size).all()):
                gprint(f"WARNING: Found non-text block of size {block_sizes[~(is_txt_block | is_valid_img_size)]} in interleaved batch")

            if isinstance(batch, TensorDict):
                batch.batch_size = []
            batch["interleaved_metadata"] = interleaved_metadata

        return batch

    def get_cond_dict(self, batch):
        ret_dict = dict()
        if "cond_input_ids" in batch:
            ret_dict["x_cond"] = batch["cond_input_ids"]

        if "img_label" in batch:
            ret_dict["label"] = batch["img_label"]

        if self.config.model.use_attention_mask:
            ret_dict["attention_mask"] = batch["attention_mask"]

        if self.config.trainer.multimodal_batches:
            ret_dict["modality"] = batch["modality"]

        if self.config.trainer.image_mode == "continuous":
            ret_dict["continuous_mode"] = True
            ret_dict["modality"] = batch["modality"]
            
        if self.parameterization == "ar" and "modality" in batch:
            ret_dict["modality"] = batch["modality"]

        return ret_dict

    def training_step(self, batch, batch_idx):
        batch = self.update_batch(batch)
        return self.compute_loss(batch, prefix="train", batch_idx=batch_idx)

    def q_xt(self, x, move_chance, allow_move_mask=None, return_ignore_batch_mask_for_metrics=False, mask_image_square=False, mask_text_region=False, batch=None):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if self.config.backbone == "maskdit" and getattr(self.config.trainer, "force_single_timestep_per_batch", False):
            num_to_mask = int(x.shape[1] * move_chance[0].item())
            batch_size, seq_len = x.shape
            random_indices = torch.rand(batch_size, seq_len, device=x.device).argsort(dim=1)[:, :num_to_mask]
            xt = x.scatter(1, random_indices, self.mask_index)
            return xt

        move_indices = torch.rand(*x.shape, device=x.device) < move_chance

        if mask_image_square:
            latent_dim = int(math.sqrt(self.config.model.img_length))
            img_move_indices = move_indices[:, self.static_img_sl].clone().reshape(move_indices.shape[0], latent_dim, latent_dim)
            max_d = int(math.sqrt(self.config.model.img_length))
            for b in range(move_indices.shape[0]):
                if move_chance[b] == 1:
                    continue
                h, w = img_move_indices[b].shape
                d = random.randint(max_d // 2, max_d - 2)
                i = random.randint(0, h - d)
                j = random.randint(0, w - d)

                mask = torch.zeros_like(img_move_indices[b], dtype=torch.bool)
                mask[i:i+d, j:j+d] = True
                move_indices[b, self.static_img_sl] = mask.reshape(-1)

        if mask_text_region:
            for b in range(x.shape[0]):
                if move_chance[b] == 1:
                    continue
                should_mask = torch.zeros_like(move_indices[b, self.static_txt_sl], dtype=torch.bool)
                max_valid = (x[b] == self.tokenizer.eos_token_id).nonzero()[0, 0] if self.tokenizer.eos_token_id in x[b] else x.shape[1]
                d = random.randint(max_valid//3, max_valid-1)
                start = random.randint(0, max_valid - d)
                should_mask[start:start+d] = True
                move_indices[b, self.static_txt_sl] = should_mask

        ignore_batch_mask_for_metrics = None
        should_mask_txt, should_mask_img = None, None
        if (mask_prob := getattr(self.config.trainer, "mask_entire_modality", None)) is not None \
            and (mask_image_square is False and mask_text_region is False) and self.backbone.training:

            assert batch is not None
            batch_size, seq_len = x.shape
            if getattr(self.config.trainer, "mask_txt_only", False):
                should_mask_txt = torch.rand(batch_size, 1, device=x.device) < mask_prob
                should_mask_img = torch.zeros_like(should_mask_txt, device=x.device)
            else:
                should_mask_txt = torch.rand(batch_size, 1, device=x.device) < mask_prob/2
                should_mask_img = torch.rand(batch_size, 1, device=x.device) < mask_prob/2

            if self.config.trainer.multimodal_batches:
                if self.config.trainer.interleaved:
                    batch_indices, start_positions, end_positions = get_contiguous_blocks_per_sample(batch["modality"], batch["sample_ids"])

                    block_size = end_positions - start_positions
                    size_mask = block_size > 4
                    batch_indices, start_positions, end_positions = batch_indices[size_mask], start_positions[size_mask], end_positions[size_mask]


                    block_counts = torch.zeros_like(batch_indices)
                    max_num_sample_ids = torch.zeros_like(batch_indices)


                    for i in range(len(batch_indices)):
                        curr_sample_id = batch["sample_ids"][batch_indices[i], start_positions[i]]
                        
                        # Find blocks before this one with same batch index and sample_id
                        prev_blocks_mask = (batch_indices[:i] == batch_indices[i]) & \
                            (batch["sample_ids"][batch_indices[:i], start_positions[:i]] == curr_sample_id)
                        
                        total_in_sample = ((batch_indices == batch_indices[i]) & (batch["sample_ids"][batch_indices, start_positions] == curr_sample_id)).sum()
                        
                        block_counts[i] = prev_blocks_mask.sum()
                        max_num_sample_ids[i] = total_in_sample

                    block_prob = (block_counts + 1) / max_num_sample_ids
                    positions = torch.arange(move_indices.shape[-1], device=move_indices.device).unsqueeze(0)  # Shape: [1, N]
                    mask = (positions >= start_positions.unsqueeze(1)) & (positions < end_positions.unsqueeze(1))  # Shape: [M, N]
                    mask = mask & (torch.rand(batch_indices.shape[0], 1, device=x.device) < (mask_prob * block_prob * 2)[..., None])
                    expanded_batch_indices = batch_indices.unsqueeze(1).expand(-1, move_indices.shape[1])  # Shape: [M, N]

                    # True if we should manually mask the part of the sequence
                    accum = torch.zeros_like(move_indices, dtype=torch.int32)  # Shape: [B, N]
                    accum.scatter_add_(0, expanded_batch_indices, mask.int())  # Accumulate counts
                    accum = accum.to(torch.bool)

                    move_indices = move_indices | accum

                    # We ignore the entire sequence if any of the blocks are fully masked
                    ignore_batch_mask_for_metrics = torch.zeros((move_indices.shape[0],), device=x.device, dtype=torch.bool)
                    ignore_batch_mask_for_metrics.scatter_add_(0, batch_indices, mask.any(dim=-1))
                else:
                    # TODO: Be smarter about masking for interleaved
                    # To make sure that we have even masking prob, we prefer to mask less but equally
                    both_mask = should_mask_txt & should_mask_img
                    should_mask_txt = torch.where(both_mask, False, should_mask_txt)
                    should_mask_img = torch.where(both_mask, False, should_mask_img)
                    move_indices = torch.where(should_mask_txt, batch["modality_mask"][..., 0], move_indices)
                    move_indices = torch.where(should_mask_img, batch["modality_mask"][..., 1], move_indices)
                    ignore_batch_mask_for_metrics = should_mask_img | should_mask_txt
            else:
                both_mask = should_mask_txt & should_mask_img
                should_mask_txt[both_mask] = False
                should_mask_img[both_mask] = False
                should_mask_img[batch["txt_sl"].all(dim=-1)] = False
                move_indices[:, self.static_txt_sl] = torch.where(should_mask_txt, True, move_indices[:, self.static_txt_sl])
                move_indices[:, self.static_img_sl] = torch.where(should_mask_img, True, move_indices[:, self.static_img_sl])
                ignore_batch_mask_for_metrics = should_mask_img | should_mask_txt

        joint_ar_nar_mask = None
        if self.config.trainer.joint_ar_nar_prob is not None and self.training:
            batch_size = x.shape[0]
            current_prob = linear_warmup(
                current_step=self.global_step,
                warmup_steps=self.config.trainer.joint_ar_nar_prob_warmup_steps,
                final_value=self.config.trainer.joint_ar_nar_prob,
                initial_value=1.0
            )
            joint_ar_nar_mask = torch.rand(batch_size, device=x.device) < current_prob
            move_indices = torch.where(joint_ar_nar_mask[:, None], False, move_indices)

        if self.config.trainer.add_label:
            move_indices[:, 0] = False

        if self.config.trainer.first_token_dropout is not None and self.training:
            _initial_mask = torch.rand(x.shape[0], device=x.device) < self.config.trainer.first_token_dropout
            move_indices[:, 0] = torch.where(_initial_mask, True, move_indices[:, 0])
            if ignore_batch_mask_for_metrics is None:
                ignore_batch_mask_for_metrics = _initial_mask
            else:
                ignore_batch_mask_for_metrics = ignore_batch_mask_for_metrics | _initial_mask

        if allow_move_mask is not None:
            move_indices = move_indices & allow_move_mask
        
        if getattr(self.config.trainer, "discrete_diffusion_mode", "absorbing") == "uniform":
            if getattr(self.config.model, "force_argmax_valid_indices", False):
                assert self.mask_index == self.text_vocab_size - 1
                text_random_tokens = torch.randint(0, self.text_vocab_size - 1, size=x.shape, device=x.device)
                img_random_tokens = torch.randint(self.text_vocab_size, self.vocab_size, size=x.shape, device=x.device)                
                random_tokens = torch.where(batch["modality_mask"][..., 0], text_random_tokens, img_random_tokens)
                assert not torch.any(random_tokens == self.mask_index)
            else:
                random_tokens = torch.randint(0, vocab_size, size=x.shape, device=x.device)
                random_tokens = torch.where(random_tokens == self.mask_index, random_tokens + 1, random_tokens) # avoid mask index
            xt = torch.where(move_indices, random_tokens, x)
        else:
            xt = torch.where(move_indices, self.mask_index, x)

        if self.parameterization == "ar":
            xt = x.clone()

        if return_ignore_batch_mask_for_metrics:
            return xt, ignore_batch_mask_for_metrics, joint_ar_nar_mask, should_mask_txt, should_mask_img, move_indices
        else:
            return xt

    def _sample_t(self, n, device):
        if self.config.backbone == "maskdit" and getattr(self.config.trainer, "force_single_timestep_per_batch", False):
            _eps_t = torch.rand(1, device=device).repeat(n)
        else:
            _eps_t = torch.rand(n, device=device)
            if self.config.trainer.joint_ar_nar_timestep_warmup_steps is not None:
                max_t = linear_warmup(
                    current_step=self.global_step,
                    warmup_steps=self.config.trainer.joint_ar_nar_timestep_warmup_steps,
                    final_value=1,
                    initial_value=0,
                    start_step=0
                )
                _eps_t = _eps_t * max_t
                if max_t == 1:
                    offset = torch.arange(n, device=device) / n
                    _eps_t = (_eps_t / n + offset) % 1

            elif self.antithetic_sampling:
                offset = torch.arange(n, device=device) / n
                _eps_t = (_eps_t / n + offset) % 1

        if getattr(self.config.trainer, "force_timestep", None) is not None:
            _eps_t[:] = self.config.trainer.force_timestep
        elif getattr(self.config.eval, "ar_inpainting_force_val", None) is not None:
            _eps_t[:] = self.config.eval.ar_inpainting_force_val

        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t.to(torch.float32)

    def _subs_parameterization(self, logits, xt, batch=None, modality=None, **kwargs):
        # log prob at the mask index = - infinity
        if not self.allow_slicing:
            logits = logits.clone()

        logits[..., self.mask_index] += self.neg_infinity
        if getattr(self.config.model, "force_argmax_valid_indices", False):
            if self.config.trainer.multimodal_batches:
                _txt_sl = batch["txt_sl"] if modality is None else modality == 0
                _img_sl = batch["img_sl"] if modality is None else modality == 1
                logits[..., self.text_vocab_size:] = torch.where(_txt_sl[..., None], self.neg_infinity, logits[..., self.text_vocab_size:])
                logits[..., :self.text_vocab_size] = torch.where(_img_sl[..., None], self.neg_infinity, logits[..., :self.text_vocab_size])
            else:
                logits[..., self.static_txt_sl, self.text_vocab_size:] = self.neg_infinity
                logits[..., self.static_img_sl, :self.text_vocab_size] = self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        if self.parameterization != "ar" and xt is not None:
            # Apply updates directly in the logits matrix.
            # For the logits of the unmasked tokens, set all values
            # to -infinity except for the indices corresponding to
            # the unmasked tokens.
            unmasked_indices = xt != self.mask_index
            if not self.allow_slicing:
                logits = torch.where(unmasked_indices.unsqueeze(-1), torch.full_like(logits, self.neg_infinity),  logits)
                logits = torch.where(
                    unmasked_indices.unsqueeze(-1) & (torch.arange(logits.size(-1)).to(logits.device) == xt.unsqueeze(-1)),  
                    torch.zeros_like(logits),
                    logits
                )
            else:
                logits[unmasked_indices] = self.neg_infinity
                logits[unmasked_indices, xt[unmasked_indices]] = 0

        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            assert (self.parameterization == "ar" or self.config.trainer.ar_llm_loss) or self.config.trainer.allow_null_sigma
            return sigma

        if sigma.ndim > 1 and not self.config.trainer.image_mode == "continuous":
            sigma = sigma.squeeze(-1)
            assert sigma.ndim == 1, sigma.shape

        if not self.time_conditioning and getattr(self.config.model, "force_time_conditioning", False):
            sigma = torch.zeros_like(sigma)

        return sigma

    def forward(
        self,
        x,
        sigma,
        batch=None,
        forward_attention_mask=None,
        return_additional_loss=False,
        x_img_emb=None,
        disable_ar_shift=False,
        continuous_mode=False,
        joint_ar_nar_mask=None,
        return_logits=False,
        block_mask=None,
        update_cache_slice=None,
        **kwargs,
    ):
        """Returns log score."""
        sigma = self._process_sigma(sigma)
        if self.config.trainer.image_mode == "continuous": assert "modality" in kwargs
        should_autocast = (((self.config.trainer.disable_forward_autocast_during_eval and self.backbone.training) is False) and (self.dtype != torch.float32))
        with ExitStack() as stack:
            if should_autocast:
                stack.enter_context(torch.autocast(device_type=self.device.type, dtype=self.dtype))

            orig_modality = None
            if self.config.backbone == "elm":
                if getattr(self.config.trainer, "print_llm_ppl", False):
                    _labels = x.clone()
                    _labels[~forward_attention_mask] = -100
                    kwargs['labels'] = _labels

                if "modality" in kwargs:
                    if self.config.mode == "eval": orig_modality = kwargs.pop("modality")
                    else: kwargs.pop("modality")

                if "modality_mask" in kwargs: kwargs.pop("modality_mask")
                if "x0" in kwargs: kwargs.pop("x0")
                if "start_pos" in kwargs: kwargs.pop("start_pos")
                if "sample_ids" in kwargs: kwargs.pop("sample_ids")

                output = self.backbone(input_ids=x, **kwargs)

                if self.config.mode == "eval": kwargs["modality"] = orig_modality

                if isinstance(output, Tensor):
                    logits = output
                else:
                    logits = output.logits

                if getattr(self.config.trainer, "print_llm_ppl", False):
                    rprint(f"AR PPL: {torch.exp(output.loss)}")
            else:
                if self.config.trainer.compile == 'max-autotune' and not is_xla_available:
                    torch.compiler.cudagraph_mark_step_begin()

                logits = self.backbone(x, sigma, continuous_mode=continuous_mode, x_img_emb=x_img_emb, block_mask=block_mask, update_cache_slice=update_cache_slice, **kwargs)
        if self.config.trainer.force_bf16_eval:
            logits = logits.to(torch.bfloat16)

        if continuous_mode:
            assert self.parameterization == "ar"
            logits, logits_img = logits

        if self.config.trainer.ar_shift and not disable_ar_shift:
            # config trainer ar shift is for training
            # disable ar shift is for sampling at inference
            logits = logits[:, :-1]
            xt = x[:, 1:]
            if orig_modality is not None and self.config.mode == 'eval':
                orig_modality = orig_modality[:, 1:]
        else:
            xt = x

        if self.config.trainer.low_precision_loss:
            logits = logits.to(self.dtype)
            if continuous_mode:
                logits_img = logits_img.to(self.dtype)

        if self.parameterization == "planner":
            return logits
        elif self.config.trainer.ar_llm_loss:
            assert not self.parameterization == "ar"
            model_output = self._subs_parameterization(logits, xt=xt, modality=orig_modality), logits
            if is_xla_available: shard_output(model_output[0], self.xla_mesh)
            if is_xla_available: shard_output(model_output[1], self.xla_mesh)
            return model_output if return_additional_loss else model_output[0]
        elif self.parameterization == "ar":
            if not getattr(self.config.trainer, "use_orig_unidisc_dit", False):
                logits = torch.where(
                    torch.arange(logits.shape[-1], device=logits.device)[None, None, :] == self.mask_index, self.neg_infinity, logits
                )
                
                _modality = kwargs.get("modality") if batch is None else batch.get("modality")

                # During eval, we let the sampler handle this part.
                if getattr(self.config.model, "force_argmax_valid_indices", False) and _modality.shape[1] == (logits.shape[1] + 1):
                    if not self.allow_slicing:
                        logits = logits.clone()
                        
                    logits[..., self.text_vocab_size:] = torch.where(
                        (kwargs.get("modality") == 0)[..., 1:, None], torch.finfo(logits.dtype).min, logits[..., self.text_vocab_size:]
                    )
                    logits[..., :self.text_vocab_size] = torch.where(
                        (kwargs.get("modality") == 1)[..., 1:, None], torch.finfo(logits.dtype).min, logits[..., :self.text_vocab_size]
                    )

                logits = logits.log_softmax(-1)

            if continuous_mode:
                return (logits, logits_img)
        elif self.parameterization == "subs":
            if return_logits:
                return logits
            model_output = self._subs_parameterization(logits, xt=xt, batch=batch, **kwargs)
            if is_xla_available: shard_output(model_output, self.xla_mesh)
            return model_output
        elif self.parameterization == "sedd":
            return self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
        elif self.parameterization == "d3pm":
            return self._d3pm_parameterization(logits=logits)

        return logits

    def compute_loss(self, batch, prefix, batch_idx=-1):
        if not is_xla_available and ((self.current_run_fwd_bwd_pass == 0 and self.config.mode == 'train') or batch_idx == 0):
            self.visualize_samples(batch, batch_idx, split=prefix)
        if getattr(self.config.trainer, 'overfit_on_first_batch', False):
            if batch_idx <= 0:
                # store it
                self.overfit_batch = batch.copy()
            else:
                batch = self.overfit_batch
        
        kwargs = self.get_cond_dict(batch)
        modality_mask = batch.get("modality_mask", None)
        (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(batch["input_ids"], batch.get("attention_mask", None))

        continuous_mode = self.config.trainer.image_mode == "continuous"
        joint_ar_nar_mask, modality = None, None
        if continuous_mode:
            assert 'modality' in batch
            x0, img_emb, attention_mask, modality = (
                batch["input_ids"],
                batch["img_emb"],
                batch["attention_mask"],
                batch["modality"],
            )  # img_emb has [0.] * txt_len + img_emb
            xt = x0
            B, N_tot, C = img_emb.shape

            noise_scheduler = self.get_vae().scheduler
            noise = torch.randn_like(img_emb)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=img_emb.device).long()
            img_timesteps = timesteps.unsqueeze(-1).expand(-1, N_tot).to(self.dtype)
            zero_timesteps = torch.zeros_like(img_timesteps)
            unet_conditioning = torch.where(modality == 1, img_timesteps, zero_timesteps)
            # unet_conditioning = timesteps.to(self.dtype)
            # unet_conditioning = torch.where(modality_mask==1, timesteps.to(self.dtype), torch.zeros_like(timesteps.to(self.dtype)))
            x_img_emb = noise_scheduler.add_noise(img_emb, noise, timesteps).to(self.dtype)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(img_emb, noise, timesteps) # todo, might break
            elif noise_scheduler.config.prediction_type:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            target = target.to(self.dtype)
        else:
            unet_conditioning, xt, x0, x_img_emb, modality_mask = None, None, input_tokens, None, batch.get("modality_mask", None)
            if self.parameterization != "ar":
                t = self._sample_t(x0.shape[0], x0.device)
                if self.T > 0:
                    t = (t * self.T).to(torch.int)
                    t = t / self.T
                    t += 1 / self.T # t \in {1/T, 2/T, ..., 1}

                if self.change_of_variables:
                    unet_conditioning = t[:, None]
                    f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
                    f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
                    move_chance = torch.exp(f_0 + t * (f_T - f_0))
                    move_chance = move_chance[:, None]
                else:
                    # total, rate
                    sigma, dsigma = self.noise(t)
                    unet_conditioning = sigma[:, None]
                    move_chance = 1 - torch.exp(-sigma[:, None])

                xt, ignore_batch_mask_for_metrics, joint_ar_nar_mask, should_mask_txt, should_mask_img, move_indices = self.q_xt(x0, move_chance, return_ignore_batch_mask_for_metrics=True, batch=batch)
                if (self.config.model.flex_attention_img_masking_prob is not None or self.config.model.flex_attention_txt_masking_prob is not None) and self.backbone.training:
                    assert xt.shape[1] == (self.config.model.img_length + self.config.model.txt_length)
                    txt_batch_attn_dropout = torch.rand(xt.shape[0], device=xt.device) < self.config.model.flex_attention_txt_masking_prob
                    img_batch_attn_dropout = torch.rand(xt.shape[0], device=xt.device) < self.config.model.flex_attention_img_masking_prob

                    # If we mask out a modality, we cannot let it only see itself
                    txt_batch_attn_dropout = txt_batch_attn_dropout & ~should_mask_txt.squeeze(-1)
                    img_batch_attn_dropout = img_batch_attn_dropout & ~should_mask_img.squeeze(-1)
                    kwargs['block_mask'] = get_block_mask(txt_batch_attn_dropout, img_batch_attn_dropout, self.config.model.txt_length, xt.shape[0], xt.shape[1], xt.device)

                    # TODO: Somehow report these metrics so we know what's going on
                    ignore_batch_mask_for_metrics = ignore_batch_mask_for_metrics | (txt_batch_attn_dropout | img_batch_attn_dropout).unsqueeze(-1)
                    
                if getattr(self.config.trainer, "interleaved_training_flex_attention", False):
                    kwargs['block_mask'] = get_interleaved_block_mask(batch["sample_ids"], batch_size=xt.shape[0], seq_len=xt.shape[1], device=xt.device)
                    kwargs['sample_ids'] = batch["sample_ids"]

            elif self.config.trainer.ar_inpainting:
                x0 = torch.cat([x0, x0], dim=1)
                kwargs['modality'] = torch.cat([kwargs['modality'], kwargs['modality']], dim=1)
                attention_mask = torch.cat([torch.zeros_like(attention_mask, dtype=attention_mask.dtype), torch.ones_like(attention_mask, dtype=attention_mask.dtype)], dim=1)
                modality_mask = torch.cat([modality_mask, modality_mask], dim=1)
                min_val, max_val = 0.0, 1.0
                n = x0.shape[0]
                _eps_t = torch.rand(n, device=self.device)
                offset = torch.arange(n, device=self.device) / n
                _eps_t = (_eps_t / n + offset) % 1
                t = (max_val - min_val) * _eps_t + min_val
                if getattr(self.config.eval, "ar_inpainting_force_val", None) is not None:
                    t = torch.full_like(t, getattr(self.config.eval, "ar_inpainting_force_val"), dtype=t.dtype, device=t.device)
                move_indices = torch.rand(*x0.shape, device=x0.device) < t[:, None]
                move_indices[:, x0.shape[1] // 2:] = False
                x0 = torch.where(move_indices, self.mask_index, x0)
                xt = x0
            else:
                xt = x0
                if (self.training or getattr(self.config.trainer, "force_flip_ar_val", False)) and self.config.trainer.rand_ar_modality_dropout is not None:
                    assert not is_xla_available
                    xt = xt.clone()
                    batch_modality_dropout = torch.rand(xt.shape[0], device=xt.device) < self.config.trainer.rand_ar_modality_dropout
                    first_modality = batch["modality"][:, 0]
                    first_modality_mask = batch["modality"] == first_modality[:, None]
                    xt = torch.where(first_modality_mask & batch_modality_dropout[:, None], self.mask_index, xt)
                    attention_mask = torch.where(first_modality_mask & batch_modality_dropout[:, None], False, attention_mask)
        true_logits = None
        model_output = self.forward(
            xt, unet_conditioning, return_additional_loss=True, batch=batch, x_img_emb=x_img_emb, joint_ar_nar_mask=joint_ar_nar_mask, **kwargs
        )
        if isinstance(model_output, tuple):
            if continuous_mode:
                model_output, img_output = model_output # model_output is for text, img_output is for image although both will have N_total length (zeroed out according to modality mask)
                B, _, C = img_output.shape
                # use modality mask to get the correct logits
                x0 = x0[modality==0].reshape(B, -1)
                xt = xt[modality==0].reshape(B, -1)
                attention_mask = torch.ones_like(x0, dtype=torch.bool) # since we separate text, we don't need to mask it out
                img_output = img_output[modality==1].reshape(B, -1, C)
                target = target[modality==1].reshape(B, -1, C)
            else:
                model_output, true_logits = model_output

        to_dtype = self.dtype if self.config.trainer.low_precision_loss else torch.float32
        model_output = model_output.to(to_dtype)
        if true_logits is not None:
            true_logits = true_logits.to(self.dtype)

        if continuous_mode:
            img_output = img_output.to(to_dtype)
            target = target.to(to_dtype)

        # if prefix != 'train':
        #     breakpoint()

        if self.config.trainer.ar_shift:
            x0 = x0[:, 1:]
            xt = xt[:, 1:]
            attention_mask = attention_mask[:, 1:]
            if modality_mask is not None: modality_mask = modality_mask[:, 1:]
            if modality is not None: modality = modality[:, 1:]

        if not self.is_compiled:
            utils.print_nans(model_output, "model_output")

        if self.parameterization == "sedd":
            return dsigma[:, None] * self._score_entropy(model_output, sigma[:, None], xt, x0)
        elif self.parameterization == "planner":
            return F.binary_cross_entropy_with_logits(model_output.squeeze(-1), move_indices.float()).mean()

        diffusion_loss = None
        if self.T > 0:
            diffusion_loss = self._d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)
            if self.parameterization == "d3pm":
                reconstruction_loss = self._reconstruction_loss(x0)
            elif self.parameterization == "subs" or self.parameterization == "ar":
                reconstruction_loss = 0
            # return reconstruction_loss + diffusion_loss

        if self.parameterization == "ar":
            if getattr(self.config.trainer, "use_orig_unidisc_dit", False):
                return self.shortcut_return(model_output, x0, attention_mask, prefix)
            else:
                log_p_theta = model_output.gather(-1, x0[:, :, None])[:, :, 0]
        else:
            # SUBS parameterization, continuous time
            log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        if self.parameterization == "ar" or getattr(self.config.trainer, "no_ce_weighting", False):
            std_weighting = 1
        else:
            std_weighting = (dsigma / torch.expm1(sigma))[:, None]
        
        # ddprint(f"self.current_run_fwd_bwd_pass: {self.current_run_fwd_bwd_pass}, log_p_theta: {torch.isnan(log_p_theta).any()}")
        # if torch.isnan(log_p_theta).any() or self.current_run_fwd_bwd_pass > 15473:
        #     import pickle
        #     import time
        #     rank = get_rank()
        #     timestamp = int(time.time() * 1e9)  # nanosecond timestep
        #     filename = f'batch_datastep_{self.current_run_fwd_bwd_pass}_rank{rank}_{timestamp}.pkl'
        #     with open(filename, 'wb') as f:
        #         pickle.dump(log_p_theta, f)
        #     ddprint(f"Saved batch to {filename}")

        loss = -log_p_theta * std_weighting
        if not (self.parameterization == "ar" or (self.config.trainer.ar_llm_loss and joint_ar_nar_mask is None) or getattr(self.config.trainer, "no_ce_weighting", False)):
            gamma = getattr(self.config.trainer, "softmin_snr", None)
            if gamma is not None:
                softmin_weighting = (dsigma / (torch.expm1(sigma) + (1 / gamma)))[:, None]
                loss = -log_p_theta * softmin_weighting

        if diffusion_loss is not None:
            assert self.T > 0
            loss = diffusion_loss

        std_loss = -log_p_theta * std_weighting
        loss_dict = dict(std_loss=std_loss.detach(), extra_losses=dict())

        if self.config.trainer.log_seperate_modal_losses:
            assert not continuous_mode
            loss_dict.update(
                dict(
                    std_txt_loss=(std_loss.detach() * modality_mask[..., 0] * attention_mask), 
                    std_img_loss=(std_loss.detach() * modality_mask[..., 1] * attention_mask)
                )
            )

        if getattr(self.config.trainer, "mask_entire_modality", None) is not None and self.backbone.training and not self.config.parameterization == "ar":
            loss_dict['batch_ignore_loss'] = ignore_batch_mask_for_metrics.squeeze(-1)

        if joint_ar_nar_mask is not None:
            if "batch_ignore_loss" in loss_dict:
                loss_dict["batch_ignore_loss"] = loss_dict["batch_ignore_loss"] | joint_ar_nar_mask
            else:
                loss_dict["batch_ignore_loss"] = joint_ar_nar_mask

        if (self.config.trainer.multimodal_batches or (self.config.trainer.text_loss_weight is not None and self.config.trainer.img_loss_weight is not None)) and not continuous_mode:
            txt_mask = modality_mask[..., 0] & attention_mask
            img_mask = modality_mask[..., 1] & attention_mask
            txt_count = txt_mask.sum()
            img_count = img_mask.sum()
            total_count = txt_count + img_count
            txt_frac = txt_count / total_count
            img_frac = img_count / total_count
            loss_dict["extra_losses"]["trainer/img_frac"] = img_frac
            loss_dict["extra_losses"]["trainer/txt_frac"] = txt_frac
            loss_dict["extra_losses"]["trainer/attention_mask_valid_frac"] = attention_mask.sum() / attention_mask.numel()
            if "batch_ignore_loss" in loss_dict:
                loss_dict["extra_losses"]["trainer/ignore_batch_metrics_frac"] = loss_dict["batch_ignore_loss"].sum() / loss_dict["batch_ignore_loss"].numel()

        if joint_ar_nar_mask is not None:
            pass # Defer loss mean until after ar_loss is calculated
        elif self.config.trainer.text_loss_weight is not None and self.config.trainer.img_loss_weight is not None:
            assert not continuous_mode
            loss = loss * attention_mask
            txt_loss = (
                loss[txt_mask].sum() / txt_count
            ) * txt_frac * self.config.trainer.text_loss_weight
            img_loss = (
                loss[img_mask].sum() / img_count
            ) * img_frac * self.config.trainer.img_loss_weight

            if getattr(self.config.trainer, "set_max_txt_loss_ratio", None) is not None and not (torch.isnan(img_loss).any() or torch.isnan(txt_loss).any()):
                max_txt_loss = getattr(self.config.trainer, "set_max_txt_loss_ratio", 1.5) * img_loss.detach()
                scale = torch.minimum(torch.tensor(1.0, device=txt_loss.device), max_txt_loss / (txt_loss.detach() + 1e-8))
                txt_loss = txt_loss * scale

            txt_loss = torch.nan_to_num(txt_loss, nan=0.0)
            img_loss = torch.nan_to_num(img_loss, nan=0.0)

            if getattr(self.config.trainer, "force_remove_img_tokens", False):
                img_loss = torch.tensor(0, device=loss.device, dtype=loss.dtype)

            loss = txt_loss + img_loss
            loss_dict.update(dict(txt_loss=txt_loss.clone().detach(), img_loss=img_loss.clone().detach()))

        elif continuous_mode:
            img_loss = F.mse_loss(img_output, target)
            
            if attention_mask[:, self.static_txt_sl].numel() == 0:
                # Let grads pass even though this is zeros...
                txt_loss = (loss[:, self.static_txt_sl] * attention_mask[:, self.static_txt_sl]).sum()
            else:
                txt_loss = (loss[:, self.static_txt_sl] * attention_mask[:, self.static_txt_sl]).sum() / attention_mask[:, self.static_txt_sl].sum()
            loss = txt_loss + img_loss * self.config.trainer.image_loss_weight
            loss_dict.update(dict(img_loss=img_loss.clone().detach(), txt_loss=txt_loss.clone().detach()))
        else:
            _attention_mask = torch.ones_like(attention_mask) if getattr(self.config.trainer, "force_full_attention_mask_loss_only", False) else attention_mask
            loss = (loss * _attention_mask).sum() / _attention_mask.sum()
            loss = torch.nan_to_num(loss, nan=0.0)

        ar_loss = None
        if self.config.trainer.ar_llm_loss:
            assert not continuous_mode
            valid_loss = xt == self.mask_index
            _labels = x0.clone()
            _labels = torch.where(valid_loss, _labels, -1)
            _labels = torch.where(~attention_mask.to(torch.bool), -1, _labels)

            _logits = true_logits
            _logits[:, :, self.mask_index] += self.neg_infinity

            if getattr(self.config.model, "force_argmax_valid_indices", False):
                assert not self.config.trainer.multimodal_batches
                _logits[:, self.static_txt_sl, self.text_vocab_size:] = torch.finfo(_logits.dtype).min
                _logits[:, self.static_img_sl, : self.text_vocab_size] = torch.finfo(_logits.dtype).min

            _logits = _logits.contiguous().view(-1, _logits.shape[-1])
            _labels = _labels.contiguous().view(-1)

            if self.config.trainer.ar_print_loss:
                _labels = _labels.to(_logits.device)
                ce_loss = loss_fct(_logits, _labels)
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                ce_loss = ce_loss.mean(dim=-1)
                if hasattr(self, 'histogram') is False:
                    self.histogram = {}

                update_histogram(self.histogram, t, ce_loss)
                rprint(f"ELM loss: move: {move_chance}, t:{t}, {ce_loss}")

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none' if joint_ar_nar_mask is not None else 'mean')
            ce_loss = loss_fct(_logits, _labels)
            loss_dict["extra_losses"]["trainer/ce_loss"] = ce_loss
            ar_loss = ce_loss

        if joint_ar_nar_mask is not None:
            __true_logits = true_logits.clone()
            __true_logits = torch.where(torch.arange(true_logits.shape[-1], device=true_logits.device)[None, None, :] == self.mask_index, self.neg_infinity, __true_logits)
            log_softmax = __true_logits.log_softmax(-1)
            ar_loss = -log_softmax.gather(-1, x0[:, :, None])[:, :, 0]

            assert ar_loss is not None
            assert ar_loss.ndim == 2
            assert loss.ndim == 2
            ar_loss_weight = joint_ar_nar_mask.sum(dim=0) / joint_ar_nar_mask.shape[0]
            nar_loss_weight = 1 - ar_loss_weight
            loss_dict["extra_losses"]["trainer/ar_loss_weight"] = ar_loss_weight.detach().float()
            loss_dict["extra_losses"]["trainer/nar_loss_weight"] = nar_loss_weight.detach().float()
            loss_dict["extra_losses"]["trainer/ce_loss"] = ar_loss.mean().detach().float()
            ar_loss = (ar_loss * ar_loss_weight) * attention_mask
            nar_loss = (loss * nar_loss_weight) * attention_mask
            valid_count = attention_mask.sum()
            if not is_xla_available:
                ar_valid_count = attention_mask[joint_ar_nar_mask].sum()
                nar_valid_count = attention_mask[~joint_ar_nar_mask].sum()
                loss_dict["extra_losses"]["trainer/ar_loss"] = (ar_loss[joint_ar_nar_mask].sum() / ar_valid_count).detach().float()
                loss_dict["extra_losses"]["trainer/nar_loss"] = (loss[~joint_ar_nar_mask].sum() / nar_valid_count).detach().float()
                loss_dict["extra_losses"]["trainer/ar_ppl"] = torch.exp(loss_dict["extra_losses"]["trainer/ar_loss"]).detach().float()
                loss_dict["extra_losses"]["trainer/nar_ppl"] = torch.exp(loss_dict["extra_losses"]["trainer/nar_loss"]).detach().float()
            loss = (torch.where(joint_ar_nar_mask[:, None], ar_loss, nar_loss).sum() / valid_count) + weighted_z_loss
        elif ar_loss is not None:
            loss = ar_loss

        loss_dict = dict(loss=loss, **loss_dict)
        std_loss = loss_dict.get("std_loss", 0)
        std_nlls = std_loss * attention_mask

        if "batch_ignore_loss" in loss_dict:
            attention_mask = torch.where(loss_dict['batch_ignore_loss'][:, None].repeat(1, attention_mask.shape[-1]), torch.full_like(attention_mask, False), attention_mask)
            
        losses = Loss(
            loss=loss_dict["loss"],
            img_loss=loss_dict.get("img_loss", 0),
            txt_loss=loss_dict.get("txt_loss", 0),
            nlls=std_nlls,
            txt_nlls=loss_dict.get("std_txt_loss", 0),
            img_nlls=loss_dict.get("std_img_loss", 0),
            token_mask=attention_mask,
            modality_mask=modality_mask,
            extra_losses=loss_dict.get("extra_losses", None),
        )

        if getattr(self.config.trainer, "disable_torchmetrics", False):
            raise NotImplementedError("Torchmetrics disabled")
        
        elif prefix == "train":
            return losses
        elif prefix == "val":
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            if hasattr(self, "valid_txt_metrics"):
                self.valid_txt_metrics.update(losses.txt_nlls, losses.modality_mask[..., 0] & losses.token_mask)
                self.valid_img_metrics.update(losses.img_nlls, losses.modality_mask[..., 1] & losses.token_mask)

        elif prefix == "test":
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        else:
            raise ValueError(f"Invalid prefix: {prefix}")
        
    @torch.no_grad()
    def zero_shot_eval(self):
        dataloader = self.validation_dataloader
        total_batches = len(dataloader)
        rprint(f"Zero shot eval with {total_batches} batches with limit_val_batches: {self.config.trainer.limit_val_batches}")
        for idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Zero shot eval validation steps", disable=not is_main_process()):
            if self.config.trainer.limit_val_batches is not None and idx >= self.config.trainer.limit_val_batches:
                break
            self.zero_shot_eval_step(batch, idx)
        
        self.zero_shot_eval_epoch_end()

    def validate(self, state: TrainingState):
        self.on_validation_epoch_start()

        if getattr(self.config.eval, "compute_val_metrics_standalone", False) and getattr(self.config.eval, "bypass_normal_validation", False):
            batch = next(iter(self.validation_dataloader))
            self.on_validation_epoch_end(example_batch=batch)
            self.on_validation_epoch_cleanup()
            return

        total_len = 10 if self.config.data.iterable or self.config.data.webdataset_indexed else len(self.validation_dataloader)
        dprint(f"Validation batches: {total_len}")

        total_batches = (
            self.config.trainer.limit_val_batches
            if (self.config.trainer.limit_val_batches is not None and self.fid_eval is False)
            else total_len
        )
        if getattr(self.config.eval, 'pplx_full_dataset', False):
            rprint("[INFO] PPLX full dataset eval, setting total_batches to total_len")
            total_batches = total_len
        elif self.config.eval.max_num_fid_batches_per_device is not None and self.fid_eval:
            total_batches = min(total_len, self.config.eval.max_num_fid_batches_per_device)

        _dataloader = self.train_dataloader if self.config.eval.val_with_train_data else self.validation_dataloader
        rprint(f"Validating with {total_batches} batches on {self.world_size} GPUs with batch size {self.config.loader.eval_batch_size}")
        for idx, batch in tqdm(enumerate(_dataloader), total=total_batches, desc="Validation steps", disable=not is_main_process()):
            if self.config.trainer.limit_val_batches is not None and idx >= total_batches:
                break
            self.validation_step(batch, idx)

        if getattr(self.config.eval, "eval_large_batch", None) is not None:
            assert isinstance(batch, TensorDict)
            dataloader_iter = iter(_dataloader)
            large_batch = [next(dataloader_iter, None) for _ in range(getattr(self.config.eval, "eval_large_batch", None))]
            large_batch = [b for b in large_batch if b is not None]
            large_batch = torch.stack(large_batch, dim=0)
            batch = large_batch
            gprint(f"Large batch shape: {batch.shape}")
        else:
            batch = next(iter(_dataloader))

        if self.config.eval.visualize_data_only:
            return

        if self.config.eval.compute_standalone_mauve and not getattr(self.config.eval, "global_disable_mauve", False):
            self.mauve_store_references(_dataloader)

        if self.config.mode == "eval":
            gprint(f"Batch shape: {batch['input_ids'].shape}")
        
        self.on_validation_epoch_end(example_batch=batch)
        self.on_validation_epoch_cleanup()

    @cached_property
    def global_batch_size(self):
        """Batch size for a single step over all GPUs"""
        # SPMD treats all ranks [regardless of node] as a single device
        return self.step_batch_size * (1 if (self.config.trainer.xla_spmd and is_xla_available) else self.world_size)

    @cached_property
    def step_batch_size(self):
        """Batch size for a single step for a single GPU"""
        return self.config.loader.batch_size * self.config.trainer.accumulate_grad_batches

    @cached_property
    def world_size(self):
        """Number of GPUs over all nodes"""
        return get_world_size()

    @cached_property
    def num_tokens_per_sample(self):
        """Number of tokens per sample"""
        return self.config.model.length

    @cached_property
    def gradient_accumulation_steps(self):
        """Number of gradient accumulation steps"""
        return self.config.trainer.accumulate_grad_batches

    @cached_property
    def static_txt_sl(self):
        return slice(None, self.config.model.txt_length)

    @cached_property
    def static_img_sl(self):
        return slice(-self.config.model.img_length, None)

    def img_txt_pair_batch_mask(self, batch=None):
        return batch["modality_mask"][..., 1].sum(dim=-1) > 0

    def txt_sl(self, batch=None):
        return batch["modality_mask"][..., 0]

    def img_sl(self, batch=None):
        return batch["modality_mask"][..., 1]

    @cached_property
    def is_compiled(self):
        return is_xla_available or self.config.trainer.compile
    
    @property
    def allow_slicing(self):
        return not is_xla_available and not self.backbone.training

    @property
    def training(self):
        return self.backbone.training

    def get_step_metrics(self):
        return {
            "trainer/global_step": self.global_step,
            "global_samples": self.global_step * self.global_batch_size,
            "train_metrics/global_tokens": self.global_step * self.global_batch_size * self.config.model.length,
            "effective_global_tokens": self.global_step * self.global_batch_size * self.config.model.length * (0.5 if self.config.parameterization == "subs" else 1.0),
            "effective_global_step": int(self.global_step * (0.5 if self.config.parameterization == "subs" else 1.0)),
        }

    def train(self):
        tr = self.config.trainer
        total_batch_size = self.global_batch_size
        initial_global_step = self.global_step
        true_step = 0
        first_epoch = 0
        self.current_run_global_step = 0
        self.current_run_fwd_bwd_pass = 0
        rprint(f"Started at step {self.accelerator.step}")
        if self.non_embedding_params < 1e9:
            with try_except(write_error_to_file=True, clear_cuda_cache=True):
                self.print_hashes()

        # There is an unknown bug with accelerator where non-master ranks don't load the step count from a checkpoint.
        # We workaround by broadcasting the step count if necessary
        if is_torch_cuda_available():
            dprint(f"Gathering step from {self.world_size} ranks")
            starting_steps = gather_object([self.accelerator.step])
            rprint(f"Starting steps: {starting_steps}")
            if not all([x > 0 for x in starting_steps]):
                rprint(f"Not all ranks have >0 step, setting to: {starting_steps[0]}")
                self.accelerator.step = starting_steps[0]

        if is_xla_available:
            import torch_xla.core.xla_model as xm
            import torch_xla.debug.profiler as xp
            assert (self.config.trainer.accumulate_grad_batches == 1) or getattr(self.config.trainer, "allow_accum_grad_batches_xla", False), "Accumulate grad batches must be 1 for XLA"

        rprint(f"***** Starting training at global step: {self.global_step} *****")
        rprint(f"  Instantaneous batch size per device = {self.config.loader.batch_size}")
        rprint(f"  Gradient Accumulation steps = {tr.accumulate_grad_batches}")
        rprint(f"  Num GPUs = {tr.devices}")
        rprint(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        rprint(f"  Total optimization steps = {tr.max_steps}")
        rprint(f"  Reported Global Batch Size: {self.global_batch_size}, Reported Step Batch Size: {self.step_batch_size}, Reported World Size: {self.world_size}")

        if not self.config.data.iterable and not self.config.data.webdataset_indexed and is_torch_cuda_available():
            num_epoch_steps = len(self.train_dataloader)
            rprint(f"  Num examples = {len(self.train_dataloader.dataset)}")
            rprint(f"  Num batches each epoch = {len(self.train_dataloader)}")
            rprint(f"Train Dataloader Size on single GPU: {num_epoch_steps}")
            if len(self.train_dataloader.dataset) < total_batch_size:
                rprint("The training dataloader is smaller than the total batch size. This may lead to unexpected behaviour.")
        else:
            num_epoch_steps = 10000

        if self.config.trainer.pytorch_profile:
            profiler = Profiler(
                output_dir=self.config.output_dir, warmup_steps=tr.profiler_warmup_steps, active_steps=tr.profiler_active_steps, record_memory=True
            )

        if self.config.trainer.viz_images_only:
            return self.viz_images_from_dataloader()

        progress_bar = tqdm(range(0, tr.max_steps), initial=initial_global_step, desc="Steps", disable=not is_local_main_process(), leave=False, smoothing=0.15)

        global_step_metrics = defaultdict(float)
        global_extra_wandb_metrics = dict()
        accumulate_steps = 0
        first_start_time = time.time()
        self.on_train_start()

        rprint(f"Training for {tr.num_epochs} epochs...")
        last_end_step_time = start_timing(f"Dataloading accum:{accumulate_steps}, #{true_step}, global_step:{self.global_step}")
        for epoch in range(first_epoch, tr.num_epochs):
            rprint(f"Starting epoch {epoch}...")
            for step, batch in enumerate(self.train_dataloader):
                ddprint(f"Data Step: {step}")
                if self.config.trainer.iterate_dataloader_only:
                    rprint(f"Iterating dataloader only: {step}")
                    # rprint((batch["modality"] == 0).sum(), (batch["modality"] == 1).sum())
                    if (batch["attention_mask"] == 0).all(dim=-1).any():
                        breakpoint()
                    batch = self.update_batch(batch)
                    if (batch["sample_ids"] == -1).all(dim=-1).any():
                        breakpoint()
                    continue

                elif getattr(self.config.trainer, "iterate_dataloader_n_dataloader_batches", None) is not None and step <= self.config.trainer.iterate_dataloader_n_dataloader_batches:
                    self.current_run_fwd_bwd_pass += 1
                    if self.current_run_fwd_bwd_pass % self.config.trainer.accumulate_grad_batches == 0:
                        self.global_step += 1
                        self.current_run_global_step += 1
                    ddprint(f"Iterating dataloader only for {self.config.trainer.iterate_dataloader_n_dataloader_batches} dataloader batches. At step {self.global_step=}, {self.current_run_global_step=}, {self.current_run_fwd_bwd_pass=}")
                    continue

                if self.config.trainer.tpu_force_mark_step: xm.mark_step()
                if self.config.trainer.sync_dataloader_timing: synchronize_device()
                global_step_metrics[f"dataloading_time"] += end_timing(last_end_step_time)

                if self.config.trainer.nvtx_profile and self.is_compiled and step == 4:
                    torch.cuda.cudart().cudaProfilerStart()

                if self.current_run_global_step == 1 and is_xla_available:
                    gprint(f"First start time: {time.time() - first_start_time}")

                if getattr(self.config.data, "force_dummy_tensordict", False):
                    gprint(self.global_step, self.current_run_global_step, true_step, batch["idx"].tolist(), batch["dataset_idx"].tolist())

                if getattr(self.config.trainer, "assert_at_n_steps", None) is not None and self.global_step == self.config.trainer.assert_at_n_steps:
                    gprint(batch["img_input_ids"].min(), batch["img_input_ids"].max(), batch["txt_input_ids"].min(), batch["txt_input_ids"].max())

                if batch is None:
                    rprint(f"Batch is None at step {step}")
                    continue

                if self.config.trainer.tpu_force_mark_step: xm.mark_step()
                ddprint(f"After Data Step 2: {step}")
                with nullcontext() if is_xla_available else self.accelerator.accumulate(self.backbone):
                    ddprint(f"Before forward pass for global_step: {self.global_step}")
                    start_forward_time = start_timing(f"Forward Pass accum:{accumulate_steps}, #{true_step}, global_step:{self.global_step}")
                    global_step_metrics["examples_seen_per_gpu"] += len(next(iter(batch.values())))
                    state: TrainingState = TrainingState(
                        epoch_step=step,
                        num_epoch_steps=num_epoch_steps,
                        global_step=self.global_step,
                        epoch=epoch,
                        true_step=true_step,
                        current_run_global_step=self.current_run_global_step,
                    )

                    if self.accelerator.sync_gradients and is_xla_available is False:
                        self.cb_handler.on_train_step_start(state=state, unit=None)

                    if self.config.trainer.tpu_force_mark_step: xm.mark_step()

                    ddprint(f"Before Fwd: {step}")
                    with xp.StepTrace('Forward', step_num=step) if self.config.trainer.tpu_profile else nullcontext():
                        losses = self.training_step(batch, step)

                    ddprint(f"After Fwd: {step}")
                    global_step_metrics["forward_pass_time"] += end_timing(start_forward_time)
                    true_step += 1
                    evaluate_extra_log_data = lambda: dict()

                    if self.config.trainer.tpu_force_mark_step: xm.mark_step()

                    if isinstance(losses, dict):
                        for k, v in losses.items():
                            if isinstance(v, torch.Tensor):
                                global_step_metrics[k.removeprefix("metric_")] += v.detach().cpu().item()
                            else:
                                global_extra_wandb_metrics[k.removeprefix("metric_")] = v
                        losses = dict(
                            filter(lambda item: not item[0].startswith("metric_"), losses.items())
                        )  # Allow for custom metrics that are not losses
                        loss = sum(losses.values())
                    elif isinstance(losses, Loss):
                        loss = losses.loss
                        metrics = self.train_metrics(losses.nlls, losses.token_mask)
                        if hasattr(self, "txt_metrics") and losses.modality_mask is not None:
                            txt_metrics = self.txt_metrics(losses.txt_nlls, losses.modality_mask[..., 0] & losses.token_mask)
                        if hasattr(self, "img_metrics") and losses.modality_mask is not None:
                            img_metrics = self.img_metrics(losses.img_nlls, losses.modality_mask[..., 1] & losses.token_mask)

                        extra_losses_dict = losses.extra_losses
                        extra_losses_dict = extra_losses_dict if extra_losses_dict is not None else dict()
                        if self.config.trainer.tpu_force_mark_step: xm.mark_step()

                        def evaluate_extra_log_data():
                            if hasattr(self, "txt_metrics"):
                                return {
                                    **{f"train/txt_{k.split('/')[-1]}": v for k, v in replace_nan_dict(txt_metrics).items()},
                                    **{f"train/img_{k.split('/')[-1]}": v for k, v in replace_nan_dict(img_metrics).items()},
                                }
                            else:
                                return {}

                        ddprint(f"Before loss: {step}")
                        incremental_dict_update(global_extra_wandb_metrics, {
                            "trainer/loss": loss,
                            "trainer/img_loss": losses.img_loss,
                            "trainer/txt_loss": losses.txt_loss,
                            **{
                                "global_samples": self.global_step * self.global_batch_size,
                                "train_metrics/global_tokens": self.global_step * self.global_batch_size * self.config.model.length,
                                "effective_global_tokens": self.global_step * self.global_batch_size * self.config.model.length * (0.5 if self.config.parameterization == "subs" else 1.0),
                                "effective_global_step": int(self.global_step * (0.5 if self.config.parameterization == "subs" else 1.0)),
                            },
                            **metrics,
                            **extra_losses_dict,
                        })
                        if self.config.trainer.tpu_force_mark_step: xm.mark_step()
                    else:
                        loss = losses

                    if is_torch_cuda_available():
                        global_step_metrics["loss"] = loss.detach().cpu().item()  # Only on the main process to avoid syncing

                    ddprint(f"Before backward pass for global_step: {self.global_step}")

                    # Short-circuit to avoid XLA eval
                    if tr.backward_pass and (is_xla_available or torch.isfinite(loss).all()):
                        start_backward_time = start_timing(f"Backward Pass accum:{accumulate_steps}, #{true_step}, global_step:{self.global_step}")
                        if self.accelerator.sync_gradients:
                            start_sync_time = start_timing(f"Gradient Sync global_step:{self.global_step}")
                            if getattr(self.config.trainer, "sync_timing", False):
                                sync_times(self.device)

                        if self.config.trainer.tpu_force_mark_step: xm.mark_step()

                        # After each fwd, we perform a bwd. However, if we are accumulating there is an internal no_sync so the gradients remain on the GPU until
                        # the final bwd before a step. This can be controlled by sync_each_batch. Note that for the last bwd, the sync happens inside the bwd call below, so any timing for stragglers needs to happen before this call.
                        with xp.StepTrace('Backward', step_num=step) if self.config.trainer.tpu_profile else nullcontext():
                            ddprint(f"Before accelerator.backward for global_step: {self.global_step}")
                            self.accelerator.backward(loss)
                            ddprint(f"After accelerator.backward for global_step: {self.global_step}")

                        with xp.StepTrace('After Backward + Clip', step_num=step) if self.config.trainer.tpu_profile else nullcontext():
                            if self.accelerator.sync_gradients:
                                ddprint(f"Before after.backward for global_step: {self.global_step}")
                                self.after_backward(state)
                                if tr.gradient_clip_val is not None:
                                    ddprint(f"Before self.accelerator.clip_grad_norm_ for global_step: {self.global_step}")
                                    total_grad_norm = self.accelerator.clip_grad_norm_(self.backbone.parameters(), tr.gradient_clip_val)
                                    ddprint(f"After self.accelerator.clip_grad_norm_ for global_step: {self.global_step}")

                        with xp.StepTrace('Optimizer + Scheduler Step', step_num=step) if self.config.trainer.tpu_profile else nullcontext():
                            ddprint(f"Before optimizer step for global_step: {self.global_step}, {step}")
                            if is_xla_available and False:
                                # TODO: xm.optimizer_step(self.optimizer) does not appear to be needed for XLA
                                xm.optimizer_step(self.optimizer)
                            else:
                                self.optimizer.step()
                            ddprint(f"After optimizer step for global_step: {self.global_step}, {step}")
                            self.lr_scheduler.step()
                            ddprint(f"After lr_scheduler step for global_step: {self.global_step}, {step}")

                        zero_grad_kwargs = dict()
                        if "apex" not in self.config.trainer.optimizer_cls:
                            zero_grad_kwargs["set_to_none"] = tr.set_grads_to_none

                        ddprint(f"Before zero_grad for global_step: {self.global_step}, {step}")
                        self.optimizer.zero_grad(**zero_grad_kwargs)
                        ddprint(f"Zeroed gradients for global_step: {self.global_step}, {step}")

                        if self.accelerator.sync_gradients:
                            if self.ema is not None: 
                                if self.config.trainer.use_custom_ema:
                                    ema_update(self.unwrap_model(self.ema), self.unwrap_model(self.backbone), self.config.trainer.ema)
                                else:
                                    self.ema.step(self.get_params())
                            global_step_metrics["gradient_sync_time"] += end_timing(start_sync_time)

                        global_step_metrics["backward_pass_time"] += end_timing(start_backward_time)
                    else:
                        if not torch.isfinite(loss).all(): gprint(f"Loss is not finite: {loss}")
                        gprint("Skipping backward pass!")

                    accumulate_steps += 1
                    self.current_run_fwd_bwd_pass += 1

                # Important: A single "global_step" is a single optimizer step. The accumulate decorator silently skips backward + optimizer to allow for gradient accumulation.
                # A "true_step" counts the number of forward passes (on a per-GPU basis). The condition below should only happen immediately after a backward + optimizer step.
                ddprint(f"Syncing gradients for global_step: {self.global_step}. Should sync: {self.accelerator.sync_gradients}, {step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")
                if self.accelerator.sync_gradients:
                    start_gradient_sync_time = start_timing(f"On Sync Gradients global_step:{self.global_step}, {step}")

                    ddprint(f"Before on_train_step_end for global_step: {self.global_step}, {step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")
                    state.batch = batch
                    del loss, losses, batch
                    gradient_sync_time_after_train_step_end_time = start_timing(f"On Sync Gradients global_step:{self.global_step}, {step}")
                    self.on_train_step_end(state)
                    ddprint(f"After on_train_step_end for global_step: {self.global_step}, {step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")
                    global_step_metrics["gradient_sync_time_after_train_step_end"] += end_timing(gradient_sync_time_after_train_step_end_time)

                    if is_xla_available and self.config.trainer.tpu_force_mark_step: xm.mark_step()

                    if self.config.trainer.profile_memory and self.global_step + 1 >= tr.max_steps:
                        rprint("Finished profiling memory...")
                        break

                    if self.config.trainer.pytorch_profile and profiler.step(self.global_step):
                        rprint(f"Profiling finished at step: {self.global_step}")
                        break

                    if getattr(self.config.trainer, "throw_failure_for_testing", False) and self.current_run_global_step == 5:
                        raise RuntimeError("Test failure")

                    if is_xla_available and self.config.trainer.tpu_force_mark_step: xm.mark_step()

                    progress_bar.update(1)
                    self.global_step += 1
                    self.current_run_global_step += 1
                    global_step_metrics["gradient_sync_time"] += end_timing(start_gradient_sync_time)

                    logs = {
                        "examples_seen": self.global_step * total_batch_size,
                        "trainer/global_step": self.global_step,
                        **{k:v for k, v in global_step_metrics.items()},
                        **{f"lr_{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())},
                        **global_extra_wandb_metrics,
                    }

                    if is_torch_cuda_available():
                        logs["gpu_max_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
                        logs["gpu_cur_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                        logs["gpu_max_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
                        logs["gpu_cur_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)

                    if is_xla_available:
                        if self.global_step % getattr(self.config.trainer, "log_every_n_steps", 1) == 0:
                            xm.add_step_closure(update_logs, args=(logs, evaluate_extra_log_data), run_async=False)
                            del logs
                            global_extra_wandb_metrics = dict()
                            if self.config.trainer.tpu_force_mark_step: xm.mark_step()
                    else:
                        logs.update(evaluate_extra_log_data())
                        progress_bar.set_postfix(**logs)
                        log(logs)
                        global_extra_wandb_metrics = dict()


                    if getattr(self.config.trainer, "sync_timing", False):
                        global_step_metrics = {f"rank_{get_rank()}/{k}": v for k, v in global_step_metrics.items()}
                        all_step_metrics = self.accelerator.gather_for_metrics([global_step_metrics], use_gather_object=True)
                        merged_metrics = {k: v for d in all_step_metrics for k, v in d.items()}
                        log(merged_metrics)

                    if is_xla_available and self.config.trainer.tpu_force_mark_step: xm.mark_step()

                    global_step_metrics = defaultdict(float)
                    accumulate_steps = 0

                    if self.global_step >= tr.max_steps:
                        break

                    ddprint(f"After logging for step v3: {self.global_step}, {step}")

                    if getattr(self.config.trainer, "assert_at_n_steps", None) is not None and self.current_run_global_step >= getattr(self.config.trainer, "assert_at_n_steps", None):
                        raise RuntimeError(f"Assertion failed at step {self.current_run_global_step}")

                    ddprint(f"After logging for step v4: {self.global_step}, {step}")

                    if is_xla_available and self.config.trainer.tpu_profile and (self.global_step == 0 or self.global_step % 50 == 0) and is_main_process():
                        import torch_xla.debug.metrics as met
                        rprint(met.metrics_report())
                        met.clear_all()

                    if is_xla_available and self.config.trainer.tpu_force_mark_step: xm.mark_step()
                    ddprint(f"Finished sync_gradients: {self.global_step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")

                ddprint(f"Finished step: {self.global_step},{step},{self.accelerator.step},{self.accelerator.gradient_accumulation_steps},{self.accelerator.gradient_state.__repr__()}")
                if self.config.trainer.sync_dataloader_timing: synchronize_device()
                last_end_step_time = start_timing(f"Dataloading #{true_step + 1}")

            if self.global_step >= tr.max_steps:
                break

            dprint(f"Finished epoch: {epoch}")

        # Create the pipeline using using the trained modules and save it.
        rprint("Training finished.")
        barrier()

        if tr.profile_memory:
            print_memory(verbose=True)
            save_memory_profile(self.config.output_dir / "profile")

        if tr.pytorch_profile:
            profiler.finish()
        elif tr.nvtx_profile:
            torch.cuda.cudart().cudaProfilerStop()
        elif self.global_step > 100 or tr.skip_early_checkpointing is False:
            self.checkpoint(state)

        barrier()
