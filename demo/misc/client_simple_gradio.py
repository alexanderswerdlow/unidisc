import time
import gradio as gr
import requests
import asyncio
from pathlib import Path
import base64
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import io
import uuid
from demo.server import ChatRequest, ChatMessage, ContentPart

API_URL = "http://localhost:8000/v1/chat/completions"

# Encode a file on disk as a base64 data URL.
def encode_image(file_path: Path) -> Dict[str, str]:
    with file_path.open("rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}

# Convert a numpy array (or a PIL image) to a base64-encoded JPEG data URL.
def encode_array_image(array: np.ndarray) -> Dict[str, str]:
    im = Image.fromarray(array) if isinstance(array, np.ndarray) else array
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}

def decode_image(img_data: str) -> Image:
    base64_data = img_data.split("base64,")[1]
    image_bytes = base64.b64decode(base64_data)
    return Image.open(io.BytesIO(image_bytes))

# Helper: compute a boolean mask from the image editor data.
def get_boolean_mask(image_data):
    if image_data is None:
        return None
    layers = image_data.get("layers", [])
    if not layers:
        bg = image_data.get("background")
        if bg is not None:
            height, width = bg.shape[:2]
            return np.zeros((height, width), dtype=np.uint8)
        return None
    mask_layer = layers[0]
    if mask_layer.shape[-1] == 4:
        colored = mask_layer[..., 3] > 0
        return (colored.astype(np.uint8) * 255), image_data["composite"]
    else:
        colored = mask_layer > 0
        return (colored.astype(np.uint8) * 255), image_data["composite"]

# Convert the stored content into a list of ContentPart objects.
def convert_to_content_parts(raw: Any) -> List[ContentPart]:
    if isinstance(raw, str):
        return [ContentPart(type="text", text=raw)]
    elif isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(ContentPart(**item))
            else:
                raise ValueError(f"Unexpected list element type: {type(item)}")
        return parts
    elif isinstance(raw, tuple):
        return [ContentPart(type="image_url", image_url=encode_image(Path(raw[0])))]
    elif isinstance(raw, dict):
        _content = raw.value if isinstance(raw, gr.Image) else raw
        if "path" in _content:
            return [ContentPart(type="image_url", image_url=encode_image(Path(_content["path"])))]
        else:
            raise ValueError(f"Expected 'path' in content dict, got: {_content}")
    else:
        raise ValueError(f"Unexpected content type: {type(raw)}")

def add_user_msg_to_history(history: List[Dict[str, Any]], message: Dict[str, Any]) -> List[Dict[str, Any]]:
    for file_path in message.get("files", []):
        history.append({"role": "user", "content": {"path": file_path}})
    if text := message.get("text"):
        history.append({"role": "user", "content": text})
    return history

def add_assistant_msg_to_history(history: List[Dict[str, Any]], content: List[Any]) -> List[Dict[str, Any]]:
    for item in content:
        if isinstance(item, str):
            history.append({"role": "assistant", "content": item})
        elif isinstance(item, tuple):
            img_data, _ = item
            if isinstance(img_data, str) and img_data.startswith("data:image"):
                image = decode_image(img_data)
            else:
                image = img_data
            history.append({"role": "assistant", "content": gr.Image(value=image)})
    return history

def build_chat_request(
    history: List[Dict[str, Any]],
    message: Dict[str, Any],
    model: str = "unidisc",
    max_tokens: int = 1024,
    temperature: float = 0.9,
    top_p: float = 0.95,
    resolution: int = 256,
    sampling_steps: int = 35,
    maskgit_r_temp: float = 4.5,
    cfg: float = 3.5,
    sampler: str = "maskgit"
) -> ChatRequest:
    messages = [ChatMessage(role=entry["role"], content=convert_to_content_parts(entry["content"])) for entry in history]
    if "mask" in message and message.get("files"):
        messages[-1].content.append(ContentPart(type="image_url", image_url=encode_array_image(message["mask"]), is_mask=True))
    
    return ChatRequest(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        resolution=resolution,
        sampling_steps=sampling_steps,
        maskgit_r_temp=maskgit_r_temp,
        cfg=cfg,
        sampler=sampler
    )

async def send_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = await asyncio.to_thread(lambda: requests.post(API_URL, json=payload))
    response.raise_for_status()
    return response.json()

def process_response(response: Dict[str, Any]) -> str | List[Any]:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    result = []
    for part in content:
        if part.get("type") == "text":
            result.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            img_data = part.get("image_url", {}).get("url", "")
            if img_data.startswith("data:image"):
                result.append((img_data, "image"))
    return ["\n".join(result)] if all(isinstance(item, str) for item in result) else result

def save_composite_image(composite: np.ndarray, file_path: str) -> str:
    image = Image.fromarray(composite.astype('uint8'), 'RGBA')
    image.save(file_path)
    return file_path

def overwrite_input_img(history: List[Dict[str, Any]], message: Dict[str, Any]) -> List[Dict[str, Any]]:
    if 'composite' in message:
        composite_image_path = save_composite_image(message['composite'], f'/tmp/gradio/{uuid.uuid4()}.png')
        for entry in reversed(history):
            if not isinstance(entry['content'], str):
                entry['content'] = gr.Image(value=composite_image_path)
                return history
    return history

async def bot(
        history: List[Dict[str, Any]],
        message: Dict[str, Any],
        max_tokens: int,
        resolution: int,
        sampling_steps: int,
        top_p: float,
        temperature: float,
        maskgit_r_temp: float,
        cfg: float,
        sampler: str
    ):
    history = add_user_msg_to_history(history, message)
    chat_request = build_chat_request(
        history,
        message,
        max_tokens=int(max_tokens),
        resolution=int(resolution),
        sampling_steps=int(sampling_steps),
        top_p=float(top_p),
        temperature=float(temperature),
        maskgit_r_temp=float(maskgit_r_temp),
        cfg=float(cfg),
        sampler=str(sampler)
    )
    do_overwrite_input_img = True
    payload = chat_request.model_dump()
    if do_overwrite_input_img:
        history = overwrite_input_img(history, message)
    try:
        response = await send_request(payload)
        content = process_response(response)
        history = add_assistant_msg_to_history(history, content)
    except requests.HTTPError as e:
        history.append({"role": "assistant", "content": f"Error: {e}"})
    return history, gr.update(value=None, interactive=True)

async def handle_submit(history, message, mask_editor, max_tokens, resolution, sampling_steps, top_p, temperature, maskgit_r_temp, cfg, sampler):
    if mask_editor is not None:
        mask, composite = get_boolean_mask(mask_editor)
        if mask is not None and mask.sum() > 0:
            message["mask"] = mask
            message["composite"] = composite
    history_out, chat_input_update = await bot(history, message, max_tokens, resolution, sampling_steps, top_p, temperature, maskgit_r_temp, cfg, sampler)
    return history_out, chat_input_update, gr.update(value=None), 0

def square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))

def update_image_editor(chat_input_value, image_editor_value, num_editor_updates, desired_resolution: int = 256):
    print(f"num_editor_updates: {num_editor_updates}, chat_input_value: {chat_input_value}")
    files = chat_input_value.get("files", [])
    if len(files) == 0:
        print(f"len files 0 returning image_editor_value, new num_editor_updates: {0}")
        return image_editor_value, 0

    # For some reason when you upload a file, this is called twice. We want to prevent further updates to avoid resetting masking while e.g., typing.
    if num_editor_updates >= 2:
        print(f"returning image_editor_value, new num_editor_updates: {num_editor_updates}")
        return image_editor_value, num_editor_updates
        
    file_path = files[0]
    image = Image.open(file_path)
    cropped_image = square_crop(image)
    if desired_resolution > 0:
        cropped_image = cropped_image.resize(
            (int(desired_resolution), int(desired_resolution)), Image.LANCZOS
        )

    if (len(chat_input_value['text']) > 0 and num_editor_updates >= 0):
        print(f"setting background,new num_editor_updates: {num_editor_updates + 1}")
        image_editor_value["background"] = cropped_image
        return image_editor_value, num_editor_updates + 1
    else:
        print(f"returning cropped_image, new num_editor_updates: {num_editor_updates + 1}")
        return cropped_image, num_editor_updates + 1

demo_examples = [
    {"text": "This is a<mask><mask><mask><mask><mask>", "files": [str(Path("demo/assets/dog.jpg").resolve())]},
]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        type="messages",
        render_markdown=False,
    )
    with gr.Row():
        with gr.Column(scale=2):
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Enter message or upload file...",
                show_label=False,
                sources=["upload"],
            )
        with gr.Column(scale=1):
            image_editor = gr.ImageMask(
                label="Mask the image",
                brush=gr.Brush(default_size=64, colors=["#000000"], color_mode='fixed')
            )
    
    gr.Examples(
        examples=demo_examples,
        inputs=chat_input,
        label="Try these examples"
    )

    with gr.Row():
        max_tokens_input = gr.Number(value=32, label="Tokens to Generate", precision=0)
        resolution_input = gr.Number(value=256, label="Resolution", precision=0)
        sampling_steps_input = gr.Number(value=32, label="Sampling Steps", precision=0)
    with gr.Row():
        top_p_input = gr.Number(value=0.95, label="Top P [maskgit_nucleus only]", precision=2)
        temperature_input = gr.Number(value=0.9, label="Temperature [maskgit_nucleus only]", precision=2)
    with gr.Row():
        maskgit_r_temp_input = gr.Number(value=4.5, label="MaskGit R Temp", precision=2)
        cfg_input = gr.Number(value=2.5, label="CFG", precision=2)
        sampler_input = gr.Dropdown(
            choices=["maskgit", "maskgit_nucleus", "ddpm_cache"],
            value="maskgit_nucleus",
            label="Sampler"
        )
    
    # State to track the last set of files we processed for the editor.
    num_editor_updates = gr.State(0)

    # We only invoke `update_image_editor` on change, but it will no-op
    # if no new file is present or if the file hasn't changed.
    chat_input.change(
        fn=update_image_editor,
        inputs=[chat_input, image_editor, num_editor_updates, resolution_input],
        outputs=[image_editor, num_editor_updates]
    )
    
    chat_input.submit(
        handle_submit,
        [
            chatbot, chat_input, image_editor,
            max_tokens_input, resolution_input, sampling_steps_input,
            top_p_input, temperature_input, maskgit_r_temp_input,
            cfg_input, sampler_input
        ],
        [chatbot, chat_input, image_editor, num_editor_updates]
    )
    
if __name__ == "__main__":
    demo.launch(share=True)
