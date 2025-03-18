import streamlit as st
import requests
from pathlib import Path
import base64
from PIL import Image
import numpy as np
import io
import uuid
from streamlit_drawable_canvas import st_canvas
from demo.api_data_defs import ChatRequest, ChatMessage, ContentPart
from typing import Dict
import time
import json

API_URL = "http://localhost:8000/v1/chat/completions"
DEMO_DIR = Path("demo")

def square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))

def process(image: Image.Image, desired_resolution: int = 256) -> Image.Image:
    cropped_image = square_crop(image.convert("RGB"))
    return cropped_image.resize(
        (int(desired_resolution), int(desired_resolution)), Image.LANCZOS
    )

DEMOS = [
    {
        "name": "Dog",
        "image": DEMO_DIR / "assets" / "dog.jpg",
        "mask": DEMO_DIR / "assets" / "dog.json",
        "text": "A corgi playing in the snow",
    },
    {
        "name": "Landscape",
        "image": DEMO_DIR / "assets" / "mountain.jpg",
        "mask": DEMO_DIR / "assets" / "mountain.json",
        "text": "Snowy mountain peak.",
    },
    {
        "name": "Architecture",
        "image": DEMO_DIR / "assets" / "building.jpg",
        "mask": DEMO_DIR / "assets" / "building.json",
        "text": "Modern glass skyscraper",
    }
]

# Custom CSS for animations and layout
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.response-card {
    animation: fadeIn 0.5s ease-in;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.demo-card {
    position: relative;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.2s ease;
    cursor: pointer;
    margin: 0.5rem;
    padding: 0.5rem;
}
.demo-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.demo-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.3);
    opacity: 0;
    transition: opacity 0.2s ease;
}
.demo-card:hover .demo-overlay {
    opacity: 1;
}
.demo-content {
    position: relative;
    padding: 0.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.demo-title {
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #1a1a1a;
}
.demo-text {
    font-size: 0.9rem;
    color: #666;
    line-height: 1.4;
}
.demo-image-container {
    position: relative;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.stButton > button {
    width: 95% !important;
    margin: 0 auto !important;
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

def load_demo_assets(demo, config):
    """Load demo assets with error handling"""
    try:
        st.session_state.demo_image = process(Image.open(demo["image"]), config["resolution"])
        st.session_state.original_image = np.array(st.session_state.demo_image)
        st.session_state.demo_text = demo["text"]
        if demo["mask"].exists():
            with demo["mask"].open("r") as f:
                print(f"Loaded mask from {demo['mask']}")
                st.session_state.initial_drawing = json.load(f)
                breakpoint()
        else:
            st.warning(f"Mask not found for {demo['name']}")
            st.session_state.initial_drawing = None
    except Exception as e:
        st.error(f"Failed to load {demo['name']} demo: {str(e)}")

def encode_image(file: Path | io.BytesIO | Image.Image) -> Dict[str, str]:
    if isinstance(file, Image.Image):
        buffered = io.BytesIO()
        file.save(buffered, format="JPEG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(file, Path):
        with file.open("rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    else:
        base64_str = base64.b64encode(file.getvalue()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}

def encode_array_image(array: np.ndarray) -> Dict[str, str]:
    im = Image.fromarray(array) if isinstance(array, np.ndarray) else array
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}

def get_boolean_mask(canvas_data):
    if canvas_data is None or canvas_data.image_data is None:
        return None, None
    mask_data = canvas_data.json_data.get("objects", [])
    if not mask_data:
        return np.zeros_like(st.session_state.original_image, dtype=np.uint8), None
    mask = np.zeros(st.session_state.original_image.shape[:2], dtype=np.uint8)
    for obj in mask_data:
        if obj.get("type") == "path":
            path = obj.get("path")
            # Custom processing of the path could be added here
    return mask * 255, None

# Initialize session state variables
if "demo_image" not in st.session_state:
    st.session_state.demo_image = None
if "demo_text" not in st.session_state:
    st.session_state.demo_text = ""
if "initial_drawing" not in st.session_state:
    st.session_state.initial_drawing = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "stroke_image" not in st.session_state:
    st.session_state.stroke_image = None
if "response" not in st.session_state:
    st.session_state.response = None

# Main UI title and demo selection
st.title("Image + Text Input Demo")

# Add configuration options in sidebar before any processing
st.sidebar.header("Configuration")
config = {
    "max_tokens": st.sidebar.number_input("Max Tokens", value=32, min_value=1, key="max_tokens"),
    "resolution": st.sidebar.number_input("Resolution", value=256, min_value=64, key="resolution"),
    "sampling_steps": st.sidebar.number_input("Sampling Steps", value=32, min_value=1, key="sampling_steps"),
    "top_p": st.sidebar.number_input("Top P", value=0.95, min_value=0.0, max_value=1.0, key="top_p"),
    "temperature": st.sidebar.number_input("Temperature", value=0.9, min_value=0.0, max_value=2.0, key="temperature"),
    "maskgit_r_temp": st.sidebar.number_input("MaskGit R Temp", value=4.5, min_value=0.0, key="maskgit_r_temp"),
    "cfg": st.sidebar.number_input("CFG", value=2.5, min_value=0.0, key="cfg"),
    "sampler": st.sidebar.selectbox(
        "Sampler",
        options=["maskgit", "maskgit_nucleus", "ddpm_cache"],
        index=1,
        key="sampler"
    ),
    "save_mask_enabled": True
}

st.subheader("Example Inputs")
with st.container():
    cols = st.columns(len(DEMOS))
    for col, demo in zip(cols, DEMOS):
        with col:
            try:
                demo_html = f"""
                <div class="demo-card" onclick="this.querySelector('button').click()">
                    <div class="demo-image-container">
                        <img src="{encode_image(process(Image.open(demo['image'])))['url']}" style="width:100%; height:auto; border-radius:4px;">
                        <div class="demo-overlay"></div>
                    </div>
                    <div class="demo-content">
                        <div class="demo-title">{demo['name']} Example</div>
                        <div class="demo-text">{demo['text']}</div>
                    </div>
                </div>
                """
                st.markdown(demo_html, unsafe_allow_html=True)
                
                if st.button(f"Load {demo['name']}", key=f"demo_{demo['name']}"):
                    load_demo_assets(demo, config)
                
                if not demo["image"].exists():
                    st.warning(f"Missing assets for {demo['name']}")
                    
            except Exception as e:
                st.error(f"Error loading {demo['name']}: {str(e)}")

# Layout: two columns - left for input, right for output
col_input, col_output = st.columns(2)

with col_input:
    st.subheader("Input")
    # st.markdown('<div style="height: 0px;"></div>', unsafe_allow_html=True)
    canvas_placeholder = st.empty()
    user_input = st.text_input(
        "Input — \"<m>\" denotes a mask token. \"<mN>\" denotes N.",
        value=st.session_state.get("demo_text", "")
    )
    uploader_placeholder = st.empty()

    # Always show uploader below canvas to allow image changes
    with uploader_placeholder.container():
        # Use a unique key for the uploader so it stays consistent
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="uploader")
        if uploaded_file:
            image = process(Image.open(uploaded_file), config["resolution"])
            st.session_state.original_image = np.array(image)

    # Render canvas only when an image is available
    if st.session_state.original_image is not None:
        print(f"Loading canvas...")
        with canvas_placeholder.container():
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=6,
                stroke_color="#000000",
                background_image=Image.fromarray(st.session_state.original_image),
                initial_drawing=st.session_state.initial_drawing,
                height=256,
                width=256,
                drawing_mode="freedraw",
                key="canvas"
            )
    else:
        canvas_result = None
        canvas_placeholder.empty()

    # Add save mask button conditional on flag
    if config["save_mask_enabled"] and canvas_result is not None and canvas_result.image_data is not None:
        if st.button("💾 Save Current Mask", help="Save drawn mask as SVG"):
            # Generate unique filename
            save_dir = DEMO_DIR / "assets" / "saved_masks"
            save_dir.mkdir(exist_ok=True)
            filename = f"mask_{uuid.uuid4().hex[:8]}.json"
            
            json_data = json.dumps(canvas_result.json_data)
            (save_dir / filename).write_text(json_data)
            
            st.session_state.last_saved_mask = {
                "path": str(save_dir / filename),
                "timestamp": time.time()
            }
            st.success(f"Mask saved as {filename}")

    # Show save confirmation temporarily
    if "last_saved_mask" in st.session_state and (time.time() - st.session_state.last_saved_mask["timestamp"]) < 5:
        st.info(f"Last saved: {Path(st.session_state.last_saved_mask['path']).name}")

    # Submission button
    if st.button("Submit"):
        if uploaded_file or user_input or st.session_state.demo_image:
            with st.spinner("Generating response..."):
                start_time = time.time()
                mask, composite = get_boolean_mask(canvas_result)
                messages = []
                if user_input:
                    messages.append(ContentPart(type="text", text=user_input))
                current_image = uploaded_file if uploaded_file else st.session_state.demo_image
                if current_image:
                    if uploaded_file:
                        img_data = encode_image(io.BytesIO(uploaded_file.getvalue()))["url"]
                    else:
                        img_data = encode_image(current_image)["url"]
                    img_part = ContentPart(
                        type="image_url",
                        image_url={"url": img_data},
                        is_mask=False
                    )
                    messages.append(img_part)
                    print(f"mask is none: {mask is None}")
                    if mask is not None:
                        mask_data = encode_array_image(mask)["url"]
                        mask_part = ContentPart(
                            type="image_url",
                            image_url={"url": mask_data},
                            is_mask=True
                        )
                        messages.append(mask_part)

                # print(f"messages: {messages}")
                payload = ChatRequest(
                    messages=[ChatMessage(role="user", content=messages)],
                    model="unidisc",
                    **config  # Use the config dictionary instead of inline sidebar inputs
                ).model_dump()

                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    st.session_state.response = response.json()
                else:
                    st.error(f"API Error: {response.text}")

with col_output:
    st.subheader("Output")
    if st.session_state.response:
        if "choices" in st.session_state.response:
            content = st.session_state.response["choices"][0]["message"]["content"]
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        st.text_input(value=part["text"], label="Unmasked Text", disabled=True)
                    elif part["type"] == "image_url":
                        st.image(part["image_url"]["url"], use_container_width=False, width=256)  # Set a fixed width
                        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            else:
                st.text_input(value=content, label="Unmasked Text", disabled=True)
