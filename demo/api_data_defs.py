from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

class ContentPart(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    type: str  # "text" or "image_url"
    text: Union[str, None] = None
    image_url: Union[Dict[str, str], Image.Image, None] = None
    is_mask: bool = False

class ChatMessage(BaseModel):
    role: str
    content: List[ContentPart]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "unidisc"
    max_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    unmask_to_eos: bool = False  # Controls masking behavior between BOS and EOS tokens
    resolution: int = 256         # New: resolution for image (default: 256)
    sampling_steps: int = 35      # New: number of sampling steps (default: 35)
    maskgit_r_temp: float = 4.5   # new parameter default
    cfg: float = 3.5            # new parameter default
    sampler: str = "maskgit"    # new parameter default
    use_reward_models: bool = False
    request_hash: Optional[str] = None