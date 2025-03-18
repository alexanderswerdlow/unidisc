from fasthtml.common import *
from fasthtml.svg import *
from monsterui.all import *
from pathlib import Path
import requests
import base64
from PIL import Image
import numpy as np
import io
import json

SHOW_DEV_BUTTONS = True
DEMO_DIR = Path("demo")
ADD_DEV_FORM = True

DEMOS = [
    {
        "name": "Dog",
        "image": DEMO_DIR / "assets" / "dog.jpg",
        "mask": DEMO_DIR / "assets" / "dog.json",
        "text": "A brown bulldog<m><m><m><m>.",
    },
    {
        "name": "Pickup Truck",
        "image": DEMO_DIR / "assets" / "pickup.jpg",
        "mask": DEMO_DIR / "assets" / "pickup.json",
        "text": "A <m> pickup truck.",
    },
    {
        "name": "Taj Mahal",
        "image": DEMO_DIR / "assets" / "tajmahal.jpg",
        "mask": DEMO_DIR / "assets" / "tajmahal.json",
        "text": "The <m><m><m> in <m>.",
    },
    {
        "name": "Venice",
        "image": DEMO_DIR / "assets" / "venice.jpg",
        "mask": DEMO_DIR / "assets" / "venice.json",
        "text": "A<m> in<m><m><m><m>.",
    },
    {
        "name": "T2I",
        "text": "A <m> sits at the counter of an art-deco loungebar, drinking whisky from a tumbler glass.",
    }
]

# Use MonsterUI's theme headers.
app, rt = fast_app(hdrs=(Theme.blue.headers(),))

def square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))

def process(image: Image.Image, desired_resolution: int = 512) -> Image.Image:
    cropped_image = square_crop(image.convert("RGB"))
    return cropped_image.resize((desired_resolution, desired_resolution), Image.LANCZOS)

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
    # Handle boolean masks more efficiently
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    im = Image.fromarray(array)
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG", quality=95)
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"url": f"data:image/jpeg;base64,{base64_str}"}


def get_boolean_mask(mask_data: str) -> np.ndarray:
    """Decode compressed mask data from client"""
    mask_info = json.loads(mask_data)
    data = base64.b64decode(mask_info['data'])
    width, height = mask_info['width'], mask_info['height']
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='big')[:width * height]
    return bits.reshape((height, width)).astype(bool)

def get_input_card_params():
    return dict(
        header=Div(H4("Input"), Subtitle("You can mask the image, text, or both.")),
        id="input-card",
        title="Input",
    )

def create_input_card_content(text_content=""):
    """Create the shared input card content structure."""
    content = [
        Div(id="preview-container", cls="relative flex justify-center items-center mb-4 p-4 empty:p-0 empty:mb-0"),
        TextArea(text_content, name="user_input", id="user-input-text", cls="resize-none h-12 w-full mb-4"),
        Input(type="file", name="uploaded_file", id="upload-image-input", cls="mb-4"),
        Input(type="hidden", name="mask_data", id="mask-data")
    ]
    return content

@rt("/")
def get(session):
    demo_cards = []
    for demo in DEMOS:
        if 'image' in demo:
            demo_image_url = encode_image(process(Image.open(demo['image'])))['url']
        
        inner_content = Div(
            Div(
                Loading(cls="hidden", htmx_indicator=True),
                id=f"demo-spinner-{DEMOS.index(demo)}",
                cls="absolute inset-0 flex items-center justify-center"
            ),
            Div(
                Img(src=demo_image_url,
                    cls="w-32 h-32 object-cover rounded-md transition-opacity hover:opacity-60 cursor-pointer mb-3"),
                cls="demo-image-container relative flex justify-center"
            ),
            P(demo['text'],
              cls="mt-2 text-sm text-muted-foreground group-hover:text-foreground transition-colors text-center"),
            cls="flex flex-col items-center p-1"
        )

        demo_card = Card(
            inner_content,
            cls="demo-card hover:shadow-md transition-shadow cursor-pointer w-fit mx-auto",
            title=f"{demo['name']}",
            hx_post=f"/load_demo/{DEMOS.index(demo)}",
            hx_target="#input-card",
            hx_swap="innerHTML",
            hx_indicator=f"#demo-spinner-{DEMOS.index(demo)}"
        )
        demo_cards.append(demo_card)

        js_script = fr"""
        document.body.addEventListener('htmx:beforeRequest', function(ev) {{
            const target = ev.detail.elt.querySelector('[hx-indicator]');
            if(target) target.querySelector('.loading').classList.remove('hidden');
        }});
        document.body.addEventListener('htmx:afterRequest', function(ev) {{
            const target = ev.detail.elt.querySelector('[hx-indicator]');
            if(target) target.querySelector('.loading').classList.add('hidden');
        }});

        const demoMaskData = {json.dumps(session.get('demo_mask'))} || undefined;
        if (typeof demoMaskData !== 'undefined' && demoMaskData !== null) {{
            const maskInfo = JSON.parse(demoMaskData);
            const data = atob(maskInfo.data);
            const arr = new Uint8Array(data.length);
            for (let i = 0; i < data.length; i++) {{
                arr[i] = data.charCodeAt(i);
            }}
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < arr.length * 8; i++) {{
                const byteIndex = Math.floor(i / 8);
                const bitIndex = 7 - (i % 8);
                if (arr[byteIndex] & (1 << bitIndex)) {{
                    const x = i % canvas.width;
                    const y = Math.floor(i / canvas.width);
                    imageData.data[(y * canvas.width + x) * 4 + 3] = 255;
                }}
            }}
            ctx.putImageData(imageData, 0, 0);
            updateMaskData(canvas);
        }}
        function downloadMaskData() {{
            const maskData = document.getElementById('mask-data').value;
            if (!maskData) return;
            const blob = new Blob([maskData], {{type: 'application/json'}});
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `mask_${{Date.now()}}.json`;
            link.click();
        }}
        function initializeCanvas(img, wrapper) {{
            const DISPLAY_SIZE = 256; // fixed display size in pixels

            // Set the preview image to the fixed size
            img.style.width = DISPLAY_SIZE + "px";
            img.style.height = DISPLAY_SIZE + "px";

            const canvas = document.createElement('canvas');
            // Use our fixed display size for the canvas dimensions
            canvas.width = DISPLAY_SIZE;
            canvas.height = DISPLAY_SIZE;
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.cursor = 'crosshair';

            const ctx = canvas.getContext('2d');
            // Compute a scale factor relative to the image's natural dimensions
            const scaleFactor = DISPLAY_SIZE / Math.max(img.naturalWidth, img.naturalHeight);
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 35 * scaleFactor; // adjust line width proportionally

            let drawing = false;
            canvas.addEventListener('mousedown', e => {{
                drawing = true;
                ctx.beginPath();
                ctx.moveTo(e.offsetX, e.offsetY);
            }});
            canvas.addEventListener('mousemove', e => {{
            if (drawing) {{
                ctx.lineTo(e.offsetX, e.offsetY);
                ctx.stroke();
            }}
            }});
            canvas.addEventListener('mouseup', e => {{
                drawing = false;
                updateMaskData(canvas);
            }});
            canvas.addEventListener('mouseleave', e => {{
                if (drawing) {{
                    drawing = false;
                    updateMaskData(canvas);
                }}
            }});
            wrapper.appendChild(canvas);
            return canvas;
        }}
        function updateMaskData(canvas) {{
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            const buffer = new Uint8Array(Math.ceil((canvas.width * canvas.height) / 8));
            for (let i = 0; i < data.length; i += 4) {{
                const pixelIndex = i / 4;
                const byteIndex = Math.floor(pixelIndex / 8);
                const bitIndex = 7 - (pixelIndex % 8);
                if (data[i + 3] > 0) {{
                    buffer[byteIndex] |= (1 << bitIndex);
                }}
            }}
            const base64 = btoa(String.fromCharCode(...buffer));
            document.getElementById('mask-data').value = JSON.stringify({{
                data: base64,
                width: canvas.width,
                height: canvas.height
            }});
        }}
        function clearMask() {{
            const canvas = document.querySelector('#preview-container canvas');
            if (canvas) {{
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                updateMaskData(canvas);
            }}
            document.getElementById('mask-data').value = '';
        }}
        function clearImage() {{
            // Clear file input and preview
            const fileInput = document.getElementById('upload-image-input');
            fileInput.value = ''; // Reset file input
            const previewContainer = document.getElementById('preview-container');
            previewContainer.innerHTML = ''; // Clear canvas and image
            document.getElementById('mask-data').value = ''; // Clear mask data
            
            // If there's a demo image, re-initialize it
            const demoImg = {json.dumps(session.get('demo_image', ''))};
            if (demoImg) {{
                const img = new Image();
                img.onload = function() {{
                    const wrapper = document.createElement('div');
                    wrapper.style.position = 'relative';
                    wrapper.style.display = 'inline-block';
                    initializeCanvas(img, wrapper);
                    wrapper.appendChild(img);
                    previewContainer.appendChild(wrapper);
                }};
                img.src = demoImg;
            }}
        }}
        // Helper function to square crop an image (crop centered)
        function squareCropImage(img) {{
            const side = Math.min(img.naturalWidth, img.naturalHeight);
            const left = (img.naturalWidth - side) / 2;
            const top = (img.naturalHeight - side) / 2;
            const offCanvas = document.createElement("canvas");
            offCanvas.width = side;
            offCanvas.height = side;
            const offCtx = offCanvas.getContext("2d");
            offCtx.drawImage(img, left, top, side, side, 0, 0, side, side);
            return offCanvas.toDataURL("image/jpeg");
        }}

        // Listen for file uploads and square crop the image before previewing
        document.getElementById('upload-image-input').addEventListener('change', function(event) {{
            const file = event.target.files[0];
            if (file) {{
                const img = new Image();
                img.onload = function() {{
                    // Square crop the loaded image
                    const croppedDataUrl = squareCropImage(img);
                    const croppedImg = new Image();
                    croppedImg.onload = function() {{
                        const previewContainer = document.getElementById('preview-container');
                        previewContainer.innerHTML = '';
                        const wrapper = document.createElement('div');
                        wrapper.style.position = 'relative';
                        wrapper.style.display = 'inline-block';
                        croppedImg.style.display = 'block';
                        croppedImg.style.maxWidth = '100%';
                        initializeCanvas(croppedImg, wrapper);
                        wrapper.appendChild(croppedImg);
                        previewContainer.appendChild(wrapper);
                    }};
                    croppedImg.src = croppedDataUrl;
                }};
                img.src = URL.createObjectURL(file);
            }}
        }});
        """

    main_content = Container(
        Div(
            DivFullySpaced(
                Style("""
                      .top-left {
                        position: absolute;
                        top: 3%;
                        left: 2%;
                        /* Additional styling as needed */
                    }
                      
                    .custom_middle {
                        position: relative;
                        top: 0%;
                        left: 50%;
                        transform: translate(-50%, 0%);
                        /* Additional styling as needed */
                    }
                      """),
                H1("UniDisc Demo", cls="text-4xl font-light tracking-tight top-left"),
                Div(*demo_cards, 
                    cls="grid grid-cols-3 gap-4 max-w-5xl custom_middle"),
                cls="flex items-center justify-between mb-8 px-4"
            ),
            Form(
                Grid(
                    Card(
                        Div(*create_input_card_content()),
                        **get_input_card_params()
                    ),
                    Card(
                        Div(id="output-content", cls="space-y-4"),
                        header=Div(H4("Output")),
                        id="output-card",
                        title="Output"
                    ),
                    cls="grid grid-cols-2 gap-6 mb-0"
                ),
                CardFooter(
                    Grid(
                        Button(
                            Div(
                                Span("Submit", cls="submit-text"),
                                Loading(cls="hidden h-4 w-4 animate-spin", id='loading', htmx_indicator=True),
                                cls="flex gap-2 items-center justify-center"
                            ),
                            cls=(ButtonT.primary,'w-full'),
                            hx_indicator="this .loading"
                        ),
                        Button("Clear Mask", type="button", cls=(ButtonT.primary,'w-full'), onclick="clearMask()"),
                        Button("Clear Image", type="button", cls=(ButtonT.primary,'w-full'), onclick="clearImage()"),
                        *([Button("Download Mask", type="button", 
                            cls=(ButtonT.primary, 'w-full', 'dev-only'),
                            onclick="downloadMaskData()")] if SHOW_DEV_BUTTONS else []),
                        Button(
                            # DivFullySpaced(UkIcon('move-down', 20, 20, 3),"Sampling Configs"), 
                            "Sampling Configs",
                            uk_toggle="target: #config-modal", id="config-modal-button", cls=(ButtonT.primary, 'w-full')
                        ),
                        cls="grid grid-cols-4 gap-2"
                    ),
                    
                ),
                Card(
                    Grid(
                        Div(
                            LabelInput("Max Tokens", name="max_tokens", type="number", value=32, cls="w-full"),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelSelect(
                                *Options(256, 512, 1024, selected_idx=1),
                                name="resolution",
                                label="Resolution",
                                cls="w-full",
                            ),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelInput("Sampling Steps", name="sampling_steps", type="number", value=32, cls="w-full"),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelInput("Top P", name="top_p", type="number", value=0.95, step="0.01", cls="w-full"),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelInput("Temperature", name="temperature", type="number", value=0.9, step="0.1", min_value="0.0", max_value="2.0", cls="w-full"),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelInput("MaskGit R Temp", name="maskgit_r_temp", type="number", value=4.5, step="0.1", cls="w-full"),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelInput("CFG", name="cfg", type="number", value=2.5, step="0.1", cls="w-full"),
                            cls="space-y-1.5"
                        ),
                        Div(
                            LabelSelect(
                                *Options("maskgit", "maskgit_nucleus", "ddpm_cache", selected_idx=1),
                                name="sampler",
                                label="Sampler",
                                cls="w-full",
                            ),
                            cls="space-y-1.5"
                        ),
                        *([
                            Div(
                            LabelInput("Port", name="port", type="number", value=8001, step="0.01", cls="w-full"),
                            cls="space-y-1.5"
                            ),
                            Div(
                            LabelSelect(*Options("False", "True", selected_idx=0), name="reward_models",label="Reward Models",  cls="w-full",),
                            cls="space-y-1.5"
                            )
                        ] if ADD_DEV_FORM else []),
                        Hidden(name="save_mask_enabled", value="True"),
                        cls="grid grid-cols-4 gap-4",
                    ),
                    cls="mb-6",
                    title="Configuration",
                    id="config-modal",
                    hidden=True
                ),
                hx_swap="innerHTML",
                hx_target="#output-content",
                hx_post="/submit",
                enctype="multipart/form-data",
                cls="mb-6"
            ),
        ),
        Script(js_script),
    )

    return main_content


@rt("/load_demo/{demo_index}")
def post(demo_index: int, session):
    demo = DEMOS[demo_index]
    if 'image' in demo:
        session['demo_image'] = encode_image(process(Image.open(demo['image'])))['url']
    if 'text' in demo:
        session['demo_text'] = demo['text']

    if 'mask' in demo and demo['mask'] and Path(demo['mask']).exists():
        session['demo_mask'] = json.loads(Path(demo['mask']).read_text())
    else:
        session['demo_mask'] = None

    content = create_input_card_content(
        text_content=session['demo_text'],
    )

    mask_json = 'undefined' if not session['demo_mask'] else json.dumps(session['demo_mask'])
    content.append(Script(fr"""
        img = new Image();
        img.onload = async function() {{
            const previewContainer = document.getElementById('preview-container');
            previewContainer.innerHTML = '';
            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            wrapper.style.display = 'inline-block';
            
            const canvas = initializeCanvas(img, wrapper);
            const ctx = canvas.getContext('2d');
            
            wrapper.appendChild(img);
            previewContainer.appendChild(wrapper);

            const dataUrl = {json.dumps(session.get('demo_image', ''))};
            const base64Data = dataUrl.split(',')[1];
            const byteCharacters = atob(base64Data);
            const byteArrays = [];
            
            for (let offset = 0; offset < byteCharacters.length; offset += 1024) {{
                const slice = byteCharacters.slice(offset, offset + 1024);
                const byteNumbers = new Array(slice.length);
                for (let i = 0; i < slice.length; i++) {{
                    byteNumbers[i] = slice.charCodeAt(i);
                }}
                const byteArray = new Uint8Array(byteNumbers);
                byteArrays.push(byteArray);
            }}
            
            const blob = new Blob(byteArrays, {{ type: 'image/jpeg' }});
            const file = new File([blob], "demo_image.jpg", {{
                type: 'image/jpeg',
                lastModified: Date.now()
            }});

            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            
            const fileInput = document.getElementById('upload-image-input');
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));

            const demoMaskData = {mask_json};
            if (typeof demoMaskData !== 'undefined' && demoMaskData !== null) {{
                const maskInfo = demoMaskData;
                const data = atob(maskInfo.data);
                const arr = new Uint8Array(data.length);
                for (let i = 0; i < data.length; i++) {{
                    arr[i] = data.charCodeAt(i);
                }}
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                for (let i = 0; i < arr.length * 8; i++) {{
                    const byteIndex = Math.floor(i / 8);
                    const bitIndex = 7 - (i % 8);
                    if (arr[byteIndex] & (1 << bitIndex)) {{
                        const x = i % canvas.width;
                        const y = Math.floor(i / canvas.width);
                        imageData.data[(y * canvas.width + x) * 4 + 3] = 255;
                    }}
                }}
                ctx.putImageData(imageData, 0, 0);
                updateMaskData(canvas);
            }}
        }};
        img.src = {json.dumps(session.get('demo_image', ''))};
    """))

    return Card(
        Div(*content),
        **get_input_card_params()
    )


@rt("/submit")
def post(
    req,
    temperature: float,
    top_p: float, 
    maskgit_r_temp: float,
    cfg: float,
    max_tokens: int,
    resolution: int,
    sampling_steps: int,
    sampler: str,
    user_input: str | None = None,
    mask_data: str | None = None,
    uploaded_file: UploadFile | None = None,
    port: int | None = 8001,
    reward_models: str | None = "False"
):
    messages = []
    if user_input:
        messages.append({"type": "text", "text": user_input})

    current_image = None
    if uploaded_file is not None and uploaded_file.filename != "No image":
        current_image = process(Image.open(io.BytesIO(uploaded_file.file.read())), int(resolution))
        img_data = encode_image(current_image)["url"]

        messages.append({
            "type": "image_url",
            "image_url": {"url": img_data},
            "is_mask": False
        })

        if mask_data is not None and len(mask_data) > 0:
            mask_array = get_boolean_mask(mask_data)
            mask_data_url = encode_array_image(mask_array)["url"]
            messages.append({
                "type": "image_url",
                "image_url": {"url": mask_data_url},
                "is_mask": True
            })

    config_payload = {
        "max_tokens": int(max_tokens),
        "resolution": int(resolution),
        "sampling_steps": int(sampling_steps),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "maskgit_r_temp": float(maskgit_r_temp),
        "cfg": float(cfg),
        "sampler": sampler,
        "use_reward_models": reward_models == "True"
    }

    payload = {
        "messages": [{"role": "user", "content": messages}],
        "model": "unidisc",
        **config_payload
    }

    
    API_URL = f"http://localhost:{port}/v1/chat/completions"
    response = requests.post(API_URL, json=payload)
    components = []

    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json:
            content = response_json["choices"][0]["message"]["content"]
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        components.append(Card(
                            P(part["text"], cls="p-4"), 
                            cls="response-card mb-4", 
                            title="Response"
                        ))
                    elif part["type"] == "image_url":
                        components.append(
                            Card(
                                Div(
                                    Img(
                                        src=part["image_url"]["url"],
                                        cls="w-64 h-64 object-cover rounded-md"
                                    ),
                                    cls="flex justify-center items-center p-4"
                                ),
                                cls="response-card mb-4"
                            )
                        )
            else:
                components.append(Card(P(content, cls="p-4"), cls="response-card mb-4", title="Response"))
    else:
        components.append(Card(P(f"API Error: {response.text}"), cls="response-card destructive mb-4", title="Error"))
    
    output_content = Div(*components, id="output-content", cls="space-y-4 flex flex-col")

    return output_content


serve(port=5003)