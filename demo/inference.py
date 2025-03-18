from __future__ import annotations

import re
import copy
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from decoupled_utils import clear_cache

from model_utils import wrapped_batch_decode
from unidisc.tokenizers.image_tokenizers import decode_latents, get_image_batch

os.environ["UNIDISC_FORCE_CUDNN_SPDA_CONTEXT"] = "1"
os.environ["UNIDISC_DISABLE_APEX_RMSNORM"] = "1"

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize
from image_utils import Im
from omegaconf import OmegaConf, open_dict
from PIL import Image
from accelerate import PartialState

import dataloader
from decoupled_utils import (breakpoint_on_error, gprint,
                             set_global_breakpoint, set_global_exists)
from demo.inference_utils import (convert_to_model_input, messages_to_batch, save_grid_image)
from demo.server import ChatRequest, ChatMessage, ContentPart
from utils import set_omega_conf_resolvers, set_torch_defaults

os.environ["HYDRA_FULL_ERROR"] = "1"

set_global_breakpoint()  # Overrides breakpoint() to use ipdb.set_trace() instead and handle distributed training
set_global_exists()
set_omega_conf_resolvers()

def get_cfg(overrides: Union[str, list[str]]):
    with initialize(version_base=None, config_path='configs'):
        cfg = compose(config_name='config.yaml', return_hydra_config=False, overrides=overrides)
        return cfg
    
def set_accelerator(config):
    from accelerate import Accelerator
    mixed_precision = "bf16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    compute_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        compute_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        compute_dtype = torch.bfloat16
    gprint(f"Compute dtype is: {compute_dtype}")
    with open_dict(config):
        config.trainer.devices = accelerator.num_processes
        config.trainer.dtype = str(compute_dtype)

    return config, accelerator
    
def setup(config=None, save_config=False, demo_type="jan", device=None, profile_memory=True):
    if profile_memory:
        torch.cuda.memory._record_memory_history()
        from torchtnt.utils.oom import attach_oom_observer
        attach_oom_observer(output_dir=str(os.getcwd()), trace_max_entries=500000)

    set_torch_defaults()
    if config is not None:
        demo_type = "jan"
        config_path = Path(__file__).parent / f'outputs/config_{demo_type}.pkl'
        
    if save_config:
        OmegaConf.resolve(config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        yaml_config_path = config_path.with_suffix('.yaml')
        with open(yaml_config_path, 'w') as yaml_file:
            OmegaConf.save(config, yaml_file)
        print(f"Saved config to {config_path}")
        exit()
    elif config is None:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        print(f"Loaded config from {config_path}")

    if config is not None:
        demo_type = "jan"
    
    config, accelerator = set_accelerator(config)
    from model import Diffusion
    device = PartialState().device if device is None else device
    model = Diffusion(config=config, tokenizer=dataloader.get_tokenizer(config), device=device)
    model.set_accelerator(accelerator)
    return partial(inference, config=config, model=model)

mask_token = "<m>"

def expand_mask_tokens(messages: List[ChatMessage]) -> List[ChatMessage]:
    # Expand <mN> -> <m><m><m>...<m> (N times)
    import re
    def replace_match(match: re.Match) -> str:
        # Extract the number after <m
        count = int(match.group(1))
        return mask_token * count
    
    def expand_mask_in_text(text: str) -> str:
        # Match <m followed by one or more digits and >
        pattern = r'<m(\d+)>'
        return re.sub(pattern, replace_match, text)
    
    messages = copy.deepcopy(messages)
    for message in messages:
        for content in message.content:
            if content.type == "text":
                print(f"Input text: {content.text}")
                content.text = expand_mask_in_text(content.text)
                print(f"Expanded text: {content.text}")
    
    return messages

def get_fixed_batch(config, tokenizer, model, input_data, resolution):
    assert len(input_data) == 2, "Input data must contain 2 messages"
    images = [content["image_url"] for content in input_data if content['type'] == "image_url"]
    texts = [content["text"] for content in input_data if content['type'] == "text"]
    assert len(images) == 1
    assert len(texts) == 1
    _img = Im(images[0])
    if not _img.height == _img.width:
        _img = _img.square(resolution, resolution)
    elif _img.height != resolution or _img.width != resolution:
        _img = _img.resize(resolution, resolution)
    img_image_ids = get_image_batch(config, model.get_vae(), {"img": _img.torch[None]}, model.device)
    txt_input_ids = dataloader.tokenize_text(tokenizer, config.data.block_size, texts)

    data = {}
    seq_len = config.data.block_size + img_image_ids.shape[1] # Allow variable token length
    data["input_ids"] = txt_input_ids["input_ids"]
    data["attention_mask"] = txt_input_ids["attention_mask"]
    data["modality"] = torch.full((1, seq_len,), dtype=torch.int64, fill_value=1)  # assuming images
    data["modality"][..., :data["input_ids"].shape[1]] = 0
    data["input_ids"] = torch.cat([data["input_ids"].to(model.device), img_image_ids], dim=-1)
    data["attention_mask"] = torch.cat([data["attention_mask"], torch.full((1, seq_len - data["attention_mask"].shape[1],), dtype=torch.bool, fill_value=True)], dim=-1).bool()
    data["img"] = Im(images[0]).torch[None]
    data["sample_ids"] = torch.full((1, seq_len), dtype=torch.int64, fill_value=0)
    for k in list(data.keys()):
        data[k] = data[k].to(model.device)

    data["input_ids"] = torch.where(
        (data["modality"] == 1) & (data["input_ids"] != -1),
        data["input_ids"] + config.data.img_token_shift,
        data["input_ids"]
    )

    return data


def inference(
    request: ChatRequest,
    config = None,
    model = None,
):
    messages = request.messages
    messages = expand_mask_tokens(messages)
    input_request = copy.deepcopy(request)

    with open_dict(config):
        config.eval.top_p = request.top_p
        config.eval.temperature = request.temperature
        config.eval.maskgit_r_temp = request.maskgit_r_temp
        config.eval.cfg = request.cfg
        config.sampling.predictor = request.sampler
        model.sampler = config.sampling.predictor

    gen_img = False
    gen_txt = False
    resolution = request.resolution
    print(f"messages: {messages}")
    img_contains_mask = any(content.is_mask for msg in messages for content in msg.content)
    input_contains_img = any(content.type == "image_url" for msg in messages for content in msg.content)
    if img_contains_mask:
        print(f"img_contains_mask: {img_contains_mask}")
        recent_mask = [content.image_url for content in messages[-1].content if content.is_mask]
        recent_img = [content.image_url for content in messages[-1].content if (content.type == "image_url" and not content.is_mask)]
        assert len(recent_mask) == len(recent_img) == 1, "Number of masks must match number of images"
        recent_mask = recent_mask[0]
        recent_img = recent_img[0]
        for msg in messages:
            msg.content = [content for content in msg.content if not content.is_mask]
    
    if any("<image>" in content.text for content in messages[-1].content if content.type == "text") or (not input_contains_img):
        print(f"Generating image: {messages[-1].content[-1].text}")
        gen_img = True
        messages[-1].content[-1].text = messages[-1].content[-1].text.replace("<image>", "")
        messages.append(ChatMessage(
            role="assistant",
            content=[ContentPart(
                type="image_url",
                image_url=Image.new("RGB", (resolution, resolution), color=(0, 0, 0))
            )]
        ))
    elif not any(mask_token in content.text for content in messages[-1].content if content.type == "text") and config.trainer.interleaved and not getattr(config.eval, "static_img_txt_demo", False):
        print(f"Generating {request.max_tokens} tokens of text")
        gen_txt = True
        messages.append(ChatMessage(
            role="assistant",
            content=[ContentPart(
                # "authentication" is a single token in the tokenizer so this gives us exact control over the number of tokens
                type="text", text="authentication" * request.max_tokens
            )]
        ))
    elif any(mask_token in content.text for content in messages[-2].content if content.type == "text") and getattr(config.eval, "static_img_txt_demo", False):
        print(f"Got user text input with mask tokens, generating text")
        gen_txt = True

    force_reorder = True
    mask_eos = True
    if force_reorder and input_contains_img:
        image_messages = [msg for msg in messages if any(content.type == "image_url" for content in msg.content)]
        messages = [msg for msg in messages if not any(content.type == "image_url" for content in msg.content)]
        messages.extend(image_messages)
        print(f"Reordered messages, images are now last")
    
    messages = convert_to_model_input(messages)
    print(f"input messages: {messages}")

    all_special_tokens = {x.content for x in model.tokenizer.added_tokens_decoder.values()}
    if mask_token not in all_special_tokens:
        new_tokens = [mask_token]
        new_tokens = list(set(new_tokens) - set(model.tokenizer.get_vocab().keys()))
        model.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens}, replace_additional_special_tokens=False)
        assert model.tokenizer.added_tokens_decoder[len(model.tokenizer) - 1].content == mask_token
        print(model.tokenizer(new_tokens, add_special_tokens=False))

    mask_token_id = len(model.tokenizer) - 1
    if config.trainer.interleaved:
        batch = messages_to_batch(config, model.tokenizer, model, messages, resolution=resolution)
    else:
        batch = get_fixed_batch(config, model.tokenizer, model, messages, resolution=resolution)

    sampling_steps = request.sampling_steps
    sample_modality = batch["modality"]
    x0 = batch["input_ids"]
    x0_unmask = torch.ones_like(x0, dtype=torch.bool)
    txt_contains_mask = False
    for i in range(x0.shape[0]):
        if gen_img or img_contains_mask:
            modality_seq = sample_modality[i]
            changes = torch.diff(modality_seq)
            change_points = torch.where(changes != 0)[0] + 1
            change_points = torch.cat([torch.tensor([0], device=change_points.device), change_points, torch.tensor([len(modality_seq)], device=change_points.device)])
            
            sequences = []
            for start, end in zip(change_points[:-1], change_points[1:]):
                if modality_seq[start] == 1:
                    sequences.append((start.item(), end.item()))
            
            if sequences:
                last_start, last_end = sequences[-1]
                x0_unmask[i, last_start:last_end] = False
                print(f"Masked slice: {last_start}:{last_end}")
            else:
                print(f"WARNING: No sequences found")

            if img_contains_mask:
                def downscale_bool(arr: np.ndarray, D: int) -> np.ndarray:
                    if len(arr.shape) == 3:
                        print(f"Converting (H, W, C) to (H, W)")
                        arr = arr.sum(axis=-1)
                    H, W = arr.shape
                    assert H % D == 0 and W % D == 0, "H and W must be divisible by D"
                    return arr.reshape(H // D, D, W // D, D).any(axis=(1, 3))
                
                import math
                _res = int(math.sqrt(last_end - last_start) * config.model.downscale_ratio)
                if recent_mask.size != (_res, _res):
                    print(f"WARNING!! recent_mask.size: {recent_mask.size}, last_end - last_start: {last_end - last_start}")
                mask_arr = downscale_bool(np.array(recent_mask.convert("RGB").resize((resolution, resolution), resample=Image.Resampling.NEAREST)).astype(np.bool_), config.model.downscale_ratio)
                print(Im(mask_arr).save())
                mask_arr = torch.from_numpy(mask_arr).to(x0_unmask.device).reshape(-1).nonzero().squeeze()
                mask_arr = mask_arr + last_start
                x0_unmask[i, last_start:last_end] = True
                x0_unmask[i, mask_arr] = False
        
        if gen_img and not gen_txt and getattr(config.eval, "static_img_txt_demo", False):
            print(f"Unmasking all text positions for static_img_txt_demo: {x0_unmask[i, modality_seq == 0].sum().item()}")
            x0_unmask[i, modality_seq == 0] = True
        elif gen_txt and not getattr(config.eval, "static_img_txt_demo", False):
            bos_positions = torch.where(x0[i] == model.tokenizer.bos_token_id)[0]
            if len(bos_positions) == 0:
                continue
                
            last_bos = bos_positions[-2] if force_reorder else bos_positions[-1]
            eos_positions = torch.where((x0[i] == model.tokenizer.eos_token_id) & (torch.arange(len(x0[i]), device=x0.device) > last_bos))[0]
            
            print(f"BOS positions: {bos_positions}, EOS positions: {eos_positions}")
            unmask_to_eos = True
            if unmask_to_eos and len(eos_positions) > 0: 
                last_eos = eos_positions[0]
            else:
                last_eos = None # Mask everything after last BOS

            x0_unmask[i, last_bos+1:last_eos] = False
            if mask_eos and force_reorder:
                x0_unmask[i, last_bos+1:last_eos+3] = False
            print(f"Masked slice: {last_bos}:{last_eos}")

        to_mask = x0[i] == mask_token_id
        if to_mask.sum().item() > 0:
            x0_unmask[i, to_mask] = False
            print(f"Found {to_mask.sum().item()} text mask tokens")
            txt_contains_mask = True


    # Add metrics for x0_unmask[0]
    true_indices = torch.where(x0_unmask[0])[0]
    first_true = true_indices[0].item()
    last_true = true_indices[-1].item()
    total_true = x0_unmask[0].sum().item()
    
    masked_modalities = batch["modality"][0][x0_unmask[0]]
    zeros = (masked_modalities == 0).sum().item()
    ones = (masked_modalities == 1).sum().item()
    
    print(f"x0_unmask num unmasked: {total_true}, x0_unmask, first position: {first_true}, x0_unmask last position: {last_true}")
    print(f"x0_unmask num txt (0) count: {zeros}, x0_unmask num img (1) count: {ones}")
    print(f"Masking {((~x0_unmask) & (batch['sample_ids'] >= 0)).sum().item()} positions, modality shape: {batch['modality'].shape}")

    # Find first invalid sample ID, defaulting to full length if none found
    invalid_positions = (batch["sample_ids"][0].long() == -1).nonzero(as_tuple=True)[0]
    first_invalid_sample_id = invalid_positions[0].item() if len(invalid_positions) > 0 else len(batch["sample_ids"][0])
    print(f"First invalid sample ID position: {first_invalid_sample_id}")
    row_len = save_grid_image(x0_unmask[0][:first_invalid_sample_id], "x0_unmask_viz.png")
    save_grid_image(batch["modality"][0][:first_invalid_sample_id], "modality_viz.png", row_len=row_len)
    _sc = batch["sample_ids"][0].clone()
    _sc[_sc == 0] = 1
    _sc[_sc == -1] = 0
    save_grid_image(_sc, "sample_ids_viz.png", row_len=row_len)

    if request.use_reward_models:
        idx = 0
        bs = 1
        num_iter = 4
        from tensordict import TensorDict
        gen_batch = TensorDict.from_dict(batch, batch_size=[batch['input_ids'].shape[0]])
        text_samples_list = []
        img_samples_list = []
        _gen_batch = []
        for i in range(num_iter):
            _gen_batch.append(gen_batch[[idx]])
        gen_batch = torch.cat(_gen_batch, dim=0)

        for j in range(num_iter):
            _modality = gen_batch[[idx]].get("modality", None)
            _sample_ids = gen_batch[[idx]].get("sample_ids", None)
            if _modality is not None:
                _modality = _modality.to(model.device)
            if _sample_ids is not None:
                _sample_ids = _sample_ids.to(model.device)
            else:
                _sample_ids = torch.zeros_like(_modality)
            text_samples, img_samples, x = model._sample(
                text_only=False,
                num_steps=sampling_steps,
                batch_size_per_gpu=bs,
                modality=_modality,
                sample_ids=_sample_ids,
                x0=gen_batch["input_ids"][[idx]].to(model.device),
                x0_unmask=x0_unmask[[idx]].to(model.device),
                return_raw_data=True,
                allow_interleaved_conditional=True
            )
            gen_batch[[idx]]['input_ids'] = x
            text_samples_list.extend(text_samples)
            img_samples_list.extend(img_samples)
            print(f"Sampled {j + 1} / {num_iter}")

        text_samples_list = wrapped_batch_decode(
            model.tokenizer,
            torch.stack(text_samples_list, dim=0),
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
            disable_mask_after_eos=True
        )

        img_samples_list = torch.cat(img_samples_list, dim=0)
        reward_config = config.eval.auto_enhance_reward_config
        rewards, raw_rewards = model.get_rewards(reward_config, img_samples_list, text_samples_list, batch=gen_batch, return_raw_rewards=True)

        gprint(f"Avg Rewards: {rewards}")

        sorted_indices = torch.argsort(rewards, descending=True).tolist()
        sorted_text_samples = [text_samples_list[i] for i in sorted_indices]
        sorted_img_samples = [img_samples_list[i] for i in sorted_indices]
        sorted_avg_rewards = [rewards[i] for i in sorted_indices]
        sorted_raw_rewards = {k: [raw_rewards[k][i] for i in sorted_indices] for k in raw_rewards}

        txt_samples = [sorted_text_samples[0]]
        img_samples = [Im(sorted_img_samples[0]).pil]
    else:
        txt_samples, img_samples = model._sample(
            text_only=False,
            num_steps=sampling_steps,
            batch_size_per_gpu=1,
            example_batch=batch,
            sample_batch_idx=0,
            modality=batch["modality"],
            sample_ids=batch["sample_ids"],
            allow_interleaved_conditional=True,
            x0_unmask=x0_unmask,
            x0=x0,
        )

        if not config.trainer.interleaved:
            txt_samples = model.tokenizer.batch_decode(txt_samples[..., model.static_txt_sl], remove_special_tokens=True)
            txt_samples[0] = txt_samples[0].replace("<unk>", "").strip()
            img_len = (resolution // config.model.downscale_ratio)**2
            img_samples = decode_latents(config, model.get_vae(), img_samples[..., -img_len:])
            assert img_samples.shape[0] == 1
            img_samples = [Im(img_samples[0]).pil]
    
    returned_message = ChatMessage(
        role="assistant",
        content=[]
    )
    if img_contains_mask or gen_img:
        print(f"Inference returned img_samples: {img_samples}")
        returned_message.content.append(ContentPart(
            type="image_url",
            image_url=img_samples[-1]
        ))

    if txt_contains_mask or not gen_img:
        print(f"Inference returned txt_samples: {txt_samples}")
        last_new_txt = ""
        for i, _txt in enumerate(txt_samples[-1].rsplit("<s>")):
            if len(_txt) > 0:
                _txt = _txt.replace("</s>", "").replace("<image>", "").replace("<s>", "").strip().replace('  ', ' ').replace('  ', ' ').replace(' .', '.').replace('\\n', ' ')
                _txt = re.sub(r'[^a-zA-Z. ]', '', _txt)
                if len(_txt) > 0:
                    last_new_txt = _txt

        returned_message.content.append(ContentPart(
            type="text",
            text=last_new_txt
        ))
    
    input_request.messages.append(returned_message)

    clear_cache()

    
    return input_request

@hydra.main(version_base=None, config_path="../configs", config_name="config")
@torch.no_grad()
def main(config=None):
    inference = setup(config, save_config=True)
    exit()
    inference([{"type": "text", "text": "Hello, how are you?"}])
    
if __name__ == "__main__":
    with breakpoint_on_error():
        main()