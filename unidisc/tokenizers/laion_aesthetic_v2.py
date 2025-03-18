import torch
import torch.nn as nn
import numpy as np
import clip
import os
import math
from constants import UNIDISC_DIR
from functools import partial

aesthetic_path = str(UNIDISC_DIR / "ckpts" / "ava+logos-l14-linearMSE.pth")

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def get_image_features(image, device, model, preprocess):
    image = preprocess(image)
    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True) # l2 normalize

    image_features = image_features.cpu().detach().numpy()
    return image_features

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def orig_score(image=None, predictor=None, clip_model=None, clip_preprocess=None, device=None, prompt="", reverse=False):
    image_features = get_image_features(image, device, clip_model, clip_preprocess)
    score_origin = predictor(torch.from_numpy(image_features).to(device).float()).item() - 5.6
    if reverse:
        score_origin = score_origin*-1
    _score = sigmoid(score_origin)
    return _score

def score(image=None, predictor=None, clip_model=None, clip_preprocess=None, device=None, prompt="", reverse=False):
    image_features = get_image_features(image, device, clip_model, clip_preprocess)
    score_origin = predictor(torch.from_numpy(image_features).to(device).float()) - 5.6
    score_origin = score_origin.detach().cpu().numpy()
    if reverse:
        score_origin = score_origin*-1
    _score = sigmoid(score_origin)
    return _score

@torch.no_grad()
def get_predictor_func(device, accept_pillow=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_state = torch.load(aesthetic_path, map_location=torch.device('cpu'))

    # CLIP embedding dim is 768 for CLIP ViT L 14
    predictor = AestheticPredictor(768)
    predictor.load_state_dict(pt_state)
    predictor.to(device)
    predictor.eval()
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    if not accept_pillow:
        clip_preprocess.transforms = [clip_preprocess.transforms[0], clip_preprocess.transforms[1], clip_preprocess.transforms[4]]
        get_reward = partial(score, predictor=predictor, clip_model=clip_model, clip_preprocess=clip_preprocess, device=device)
    else:
        get_reward = partial(orig_score, predictor=predictor, clip_model=clip_model, clip_preprocess=clip_preprocess, device=device)
    return get_reward

    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    get_reward = get_predictor_func(device, accept_pillow=False)
    from image_utils import Im
    rand_img = torch.rand(5, 3, 224, 224)
    rand_img[1] = Im.random().resize(224, 224).torch
    rand_img[2] = Im.random().resize(224, 224).torch
    rand_img[3] = Im.random().resize(224, 224).torch
    rand_img[4] = Im("https://img.freepik.com/premium-photo/majestic-3d-lion-illustration-retro-aesthetic-artwork_971394-242.jpg").resize(224, 224).torch
    torch_rewards = get_reward(image=rand_img)
    print(torch_rewards)
    from PIL import Image
    pil_images = [Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for img in rand_img]
    get_reward_pil = get_predictor_func(device, accept_pillow=True)
    pil_rewards = [get_reward_pil(image=pil_img) for pil_img in pil_images]
    print(pil_rewards)
