from __future__ import annotations

import base64
import copy
import io
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math
from PIL import Image
from image_utils import Im

from decoupled_utils import gprint

if TYPE_CHECKING:
    from demo.server import ChatRequest

def tensor_center_crop(tensor_image, crop_size):
    _, _, h, w = tensor_image.shape

    while h >= 2 * crop_size[0] and w >= 2 * crop_size[1]:
        tensor_image = F.interpolate(tensor_image, size=(h // 2, w // 2), mode='area')
        _, _, h, w = tensor_image.shape

    scale = max(crop_size[0] / h, crop_size[1] / w)
    new_h, new_w = round(h * scale), round(w * scale)
    tensor_image = F.interpolate(tensor_image, size=(new_h, new_w), mode='bilinear')

    crop_top = random.randint(0, new_h - crop_size[0])
    crop_left = random.randint(0, new_w - crop_size[1])
    crop_bottom = crop_top + crop_size[0]
    crop_right = crop_left + crop_size[1]
    return tensor_image[:, :, crop_top:crop_bottom, crop_left:crop_right]

def parse_messages(messages: List[dict]) -> Tuple[List[Image.Image], List[List[dict]]]:
    """
    Given a list of message dicts with format:
    [
        {"type": "text", "text": msg},
        {"type": "image_url", "image_url": <PIL Image>}
    ]
    
    Returns:
      - all_images: a list containing the PIL images, in the order of their appearance
      - all_content: a nested list (single conversation) with dicts indicating message type
    """
    all_images: List[Image.Image] = []
    conversation: List[dict] = []
    
    for msg in messages:
        if msg["type"] == "text":
            conversation.append(msg)
        elif msg["type"] == "image_url":
            idx = len(all_images)
            all_images.append(msg["image_url"])
            _msg = copy.deepcopy(msg)
            _msg["image_url"] = {"url": idx}
            conversation.append(_msg)
        else:
            raise ValueError(f"Unsupported message type: {msg['type']}. Expected 'text' or 'image_url'.")
    
    all_content = [conversation]
    return all_images, all_content

def messages_to_batch(config, tokenizer, model, input_data, resolution):
    import copy

    from model import get_image_batch
    from unidisc.tokenizers.tokenize_interleaved import _has_image, preprocess

    # Build conversations and extract images.
    all_images = []
    conversations = []
    for item in input_data:
        role = item["role"]
        assert role in ["user", "assistant"]
        role = "human" if role == "user" else "gpt"
        if item["type"] == "image_url":
            token = "<image>"
            all_images.append(item["image_url"])
        elif item["type"] == "text":
            token = item["text"]
        else:
            continue
        if conversations and conversations[-1]["from"] == role:
            conversations[-1]["value"] += " " + token
        else:
            conversations.append({"from": role, "value": token})

    output_list = []
    entry = {"id": "1", "conversations": conversations}
    if all_images:
        entry["image"] = {}
    output_list.append(entry)
    all_content = output_list

    vae = model.get_vae()
    device = model.device
    if not all_images:
        image_ids = None
    else:
        _img = torch.cat([
            tensor_center_crop(
                torch.from_numpy(np.array(img))[None, :].permute(0, 3, 1, 2) / 255,
                (resolution, resolution)
            ) for img in all_images
        ])
        try:
            batch_size = 32
            image_ids_list = []
            for i in range(0, len(_img), batch_size):
                batch = _img[i:i+batch_size]
                batch_ids = get_image_batch(config, vae, {"img": batch}, device)
                image_ids_list.append(batch_ids)
            image_ids = torch.cat(image_ids_list)
        except Exception as e:
            gprint(f"{_img.shape}, {e}")
            import traceback
            traceback.print_exc()

    all_input_ids = []
    all_attention_masks = []
    all_modality = []
    assert len(all_content) == 1
    for sources in all_content:
        has_image = _has_image(sources)
        sources = copy.deepcopy([sources["conversations"]])
        _image_ids = image_ids if has_image else None
        try:
            print(f"Sources: {sources}")
            data_dict = preprocess(sources, tokenizer, has_image=has_image, image_ids=_image_ids)
        except Exception as e:
            import traceback
            traceback.print_exc()
            gprint(f"Error in preprocess: {e}")
            return None, None, None
        input_ids = data_dict["input_ids"][0]
        attention_mask = data_dict["attention_mask"][0]
        modality = data_dict["modality"][0]
        if (input_ids[-2:] == tokenizer.eos_token_id).all():
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]
            modality = modality[:-1]

        assert config.model.length >= input_ids.shape[0], f"Input ids length {input_ids.shape[0]} is greater than model length {config.model.length}"

        attention_mask = attention_mask.bool()
        print(f"Attention mask: {attention_mask.shape}, input ids: {input_ids.shape}, modality: {modality.shape}")
        
        if modality[-1] == 1:
            is_image = modality == 1
            change_points = torch.where(is_image[:-1] != is_image[1:])[0] + 1
            if change_points.numel() > 0:
                start_pos = change_points[-1].item()
                modality[start_pos:] = 0
                attention_mask[start_pos:] = False
                input_ids[start_pos:] = tokenizer.pad_token_id
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_modality.append(modality)

    all_input_ids = torch.stack(all_input_ids)
    all_attention_masks = torch.stack(all_attention_masks)
    all_modality = torch.stack(all_modality)
    all_sample_ids = torch.zeros_like(all_modality, dtype=torch.long)
    all_sample_ids[~all_attention_masks] = -1
    batch = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "modality": all_modality.long(),
        "sample_ids": all_sample_ids.long(),
    }

    for k in batch:
        batch[k] = batch[k].to(device)

    batch["input_ids"] = torch.where(
        (batch["modality"] == 1) & (batch["input_ids"] != -1),
        batch["input_ids"] + config.data.img_token_shift,
        batch["input_ids"]
    )

    return batch

def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def convert_to_model_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model_input = []
    for msg in messages:
        for part in msg.content:
            if part.type == "text" and part.text:
                model_input.append({
                    "type": "text",
                    "text": part.text,
                    "role": msg.role
                })
            elif part.type == "image_url" and part.image_url:
                model_input.append({
                    "type": "image_url",
                    "image_url": part.image_url,
                    "role": msg.role
                })
    return model_input

def convert_request_pil_to_base64(request: ChatRequest) -> ChatRequest:
    for msg in request.messages:
        for part in msg.content:
            if part.type == "image_url" and isinstance(part.image_url, Image.Image):
                buffered = io.BytesIO()
                part.image_url.convert("RGB").save(buffered, format="JPEG")
                base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                part.image_url = {"url": f"data:image/jpeg;base64,{base64_str}"}
    
    return request

def convert_request_base64_to_pil(request: ChatRequest) -> ChatRequest:
    for message in request.messages:
        for part in message.content:
            if part.type == "image_url" and "url" in part.image_url:
                image_data = part.image_url["url"]
                # Remove any data URL header, e.g. "data:image/jpeg;base64,"
                if image_data.startswith("data:"):
                    try:
                        header, image_data = image_data.split(",", 1)
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid image URL format: {image_data}"
                        ) from e
                try:
                    decoded_bytes = base64.b64decode(image_data)
                    part.image_url = Image.open(io.BytesIO(decoded_bytes))
                except Exception as e:
                    raise ValueError(
                        f"Error decoding or loading image. Ensure the base64 string is valid. Details: {e}"
                    ) from e
    return request

def trim_merge_messages(request: ChatRequest) -> ChatRequest:
    # Remove empty text parts from each message
    for msg in request.messages:
        msg.content = [
            part for part in msg.content 
            if not (part.type == "text" and part.text.strip() == "")
        ]
    
    # Remove messages with no content
    request.messages = [
        msg for msg in request.messages
        if msg.content
    ]
    
    # Merge consecutive messages with the same role
    merged_messages = []
    for msg in request.messages:
        if merged_messages and merged_messages[-1].role == msg.role:
            merged_messages[-1].content.extend(msg.content)
        else:
            merged_messages.append(msg)
    
    request.messages = merged_messages
    return request

def save_grid_image(input_arr: torch.Tensor, output_name, row_len=None):
    # Convert to boolean then to int (0/1)
    x0_bool = input_arr.bool().long()
    n = x0_bool.numel()
    if row_len is None:
        row_len = math.ceil(math.sqrt(n))
    rows = math.ceil(n / row_len)
    total = rows * row_len
    # Pad with -1 to mark padded positions
    padded = torch.full((total,), -1, dtype=torch.long)
    padded[:n] = x0_bool
    grid = padded.reshape(rows, row_len)
    # Create an RGB image: false=black, true=white, padded=red
    image = torch.zeros((rows, row_len, 3), dtype=torch.uint8)
    mask_true = (grid == 1)
    mask_padding = (grid == -1)
    image[mask_true] = torch.tensor([255, 255, 255], dtype=torch.uint8)
    image[mask_padding] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    img = Image.fromarray(image.numpy(), mode='RGB')
    
    from datetime import datetime
    output = Im(img).save(datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + "_" + output_name)
    print(f"Saved visualization to {output}")
    return row_len