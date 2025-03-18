import sys
from math import sqrt
from pathlib import Path
from types import FrameType

import einops
import hydra
import hydra.utils
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from constants import LIB_DIR, UNIDISC_DIR
from decoupled_utils import gprint, rprint
from torchvision import transforms

def get_vae(config, device, use_cond: bool = False):
    def get_attr(attr_name):
        if use_cond:
            return getattr(config.model, f"cond_{attr_name}", None)
        return getattr(config.model, attr_name, None)

    vae_type = get_attr("vae_type")
    if vae_type == "maskgit":
        rprint(f"Using MaskGit VQGAN")
        from vqgan.modeling_maskgit_vqgan import MaskGitVQGAN

        vae = MaskGitVQGAN.from_pretrained(Path(__file__).parent.parent.parent / "vqgan" / "vqgan_pretrained")
        assert get_attr("image_vocab_size") == vae.config.num_embeddings
    elif vae_type == "taming":
        rprint(f"Using Taming VQGAN")
        from vqgan.modeling_taming_vqgan import VQGANModel

        vae = VQGANModel.from_pretrained(Path(__file__).parent / "vqgan" / "vqgan_taming_ckpt")
    elif vae_type == "diffusers":
        from diffusers import VQModel

        vae = VQModel.from_pretrained(get_attr("use_custom_vae_ckpt"), subfolder="vqvae")
        vae.config.lookup_from_codebook = True
    elif vae_type == "raw":
        return None
    elif vae_type == "video_vqvae":
        sys.path.append(str(LIB_DIR / "Open-Sora-Plan"))
        from opensora.models.ae import VQVAEModel

        vae = VQVAEModel.download_and_load_model("kinetics_stride4x4x4")
    elif vae_type == "VQ-16" or vae_type == "VQ-8":
        sys.path.append(str(LIB_DIR / "LlamaGen"))
        from tokenizer.tokenizer_image.vq_model_hf import (VQ_models_HF, VQModelHF)
        if get_attr("use_custom_vae_ckpt") is not None:
            from tokenizer.tokenizer_image.vq_model import VQ_models
            vae = VQ_models["VQ-8" if vae_type == "VQ-8" else "VQ-16"](codebook_size=get_attr("image_vocab_size"), codebook_embed_dim=getattr(config.model, "codebook_embed_dim", 256))
            vae.load_state_dict(torch.load(get_attr("use_custom_vae_ckpt"), map_location=device)["model"])
            assert get_attr("downscale_ratio") == (8 if vae_type == "VQ-8" else 16)
        elif vae_type == "VQ-8":
            vae = VQ_models_HF["VQ-8"]()
            vae.load_state_dict(torch.load(UNIDISC_DIR / "ckpts/vq_ds8_c2i.pt")["model"])
            assert get_attr("downscale_ratio") == 8
        elif vae_type == "VQ-16":
            vae = VQModelHF.from_pretrained("FoundationVision/vq-ds16-c2i")
            assert get_attr("downscale_ratio") == 16
        assert (
            get_attr("image_vocab_size") == vae.config.codebook_size
        ), f"Image vocab size {get_attr('image_vocab_size')} does not match VAE codebook size {vae.config.codebook_size}"
    elif vae_type == "lfq_128" or vae_type == "lfq_256":
        sys.path.append(str(LIB_DIR / "Open-MAGVIT2"))
        from magvit_inference import load_vqgan_new
        from omegaconf import OmegaConf

        if get_attr("use_custom_vae_ckpt") is not None and get_attr("use_custom_vae_config") is not None:
            config_file = get_attr("use_custom_vae_config")
            ckpt_path = get_attr("use_custom_vae_ckpt")
            rprint(f"Using custom VAE config: {config_file} and ckpt: {ckpt_path}")
        else:
            config_file = LIB_DIR / "Open-MAGVIT2" / "configs" / f"imagenet_lfqgan_{128 if '128' in vae_type else '256'}_B.yaml"
            ckpt_path = LIB_DIR / "Open-MAGVIT2" / "ckpts" / f"imagenet_{128 if '128' in vae_type else '256'}_B.ckpt"
        configs = OmegaConf.load(config_file)
        vae = load_vqgan_new(configs, ckpt_path).to(device)
    elif vae_type == "bsq_18":
        sys.path.append(str(LIB_DIR / "bsq-vit"))
        from scripts.main_image_tokenizer import get_model
        vae = get_model("bsq_18").to(device)
    elif vae_type == 'cosmos':
        # To use Cosmos, you first need to download the pretrained models from HuggingFace.
        # from huggingface_hub import login, snapshot_download
        # import os
        # HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
        # login(token=HUGGINGFACE_TOKEN, add_to_git_credential=True)
        # model_names = [
        #         # "Cosmos-0.1-Tokenizer-CI8x8",
        #         # "Cosmos-0.1-Tokenizer-CI16x16",
        #         # "Cosmos-0.1-Tokenizer-DI8x8",
        #         "Cosmos-0.1-Tokenizer-DI16x16",
        # ]
        # for model_name in model_names:
        #     hf_repo = "nvidia/" + model_name
        #     local_dir = "pretrained_ckpts/" + model_name
        #     os.makedirs(local_dir, exist_ok=True)
        #     print(f"downloading {model_name}...")
        #     snapshot_download(repo_id=hf_repo, local_dir=local_dir)
        import importlib
        import cosmos_tokenizer.image_lib
        importlib.reload(cosmos_tokenizer.image_lib)
        from cosmos_tokenizer.image_lib import ImageTokenizer
        model_name = 'Cosmos-0.1-Tokenizer-DI16x16' # @param ["Cosmos-0.1-Tokenizer-CI16x16", "Cosmos-0.1-Tokenizer-CI8x8", "Cosmos-0.1-Tokenizer-DI8x8", "Cosmos-0.1-Tokenizer-DI16x16"]
        cosmos_dir = Path(LIB_DIR / "Cosmos-Tokenizer")
        encoder_ckpt = str(cosmos_dir / f"pretrained_ckpts/{model_name}/encoder.jit")
        decoder_ckpt = str(cosmos_dir / f"pretrained_ckpts/{model_name}/decoder.jit")
        vae = ImageTokenizer(
            checkpoint_enc=encoder_ckpt,
            checkpoint_dec=decoder_ckpt,
            device="cuda",
            dtype="bfloat16",
        )

        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
    elif "titok" in vae_type:
        from huggingface_hub import hf_hub_download
        sys.path.append(str(LIB_DIR / "1d-tokenizer"))
        from modeling.titok import TiTok
        if vae_type == "titok256":
            vae = TiTok.from_pretrained("yucornetto/tokenizer_titok_sl256_vq8k_imagenet")
        elif vae_type == "titok128":
            vae = TiTok.from_pretrained("yucornetto/tokenizer_titok_bl128_vq8k_imagenet")
        elif vae_type == "titok64":
            vae = TiTok.from_pretrained("yucornetto/tokenizer_titok_b64_imagenet")
        else:
            raise ValueError(f"Unknown TiTok type: {vae_type}")
        vae.eval()
        vae.requires_grad_(False)
    elif vae_type == "chameleon":
        from transformers import ChameleonForConditionalGeneration
        model = ChameleonForConditionalGeneration.from_pretrained(
            "leloy/Anole-7b-v0.1-hf",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        vae = model.model
        vae.vqmodel.to(torch.float32)
        if config.data.resolution == 256:
            vae.vqmodel.quantize.quant_state_dims = [16, 16]
        elif config.data.resolution == 512:
            vae.vqmodel.quantize.quant_state_dims = [32, 32]
    elif vae_type == "lumina":
        from unidisc.tokenizers.chameleon_tokenizers import ItemProcessor
        vae = ItemProcessor(target_size=config.data.resolution)
    elif vae_type == "stable_diffusion":
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "benjamin-paine/stable-diffusion-v1-5", torch_dtype=torch.float16
        )  # since runwayml/stable-diffusion-v1-5 dont work now
        vae = pipe.vae

        # add pipe.scheduler to vae as a new attribute
        vae.scheduler = pipe.scheduler
    elif vae_type == "magvit":
        # sys.path.append(str(LIB_DIR / "Show-o"))
        import importlib.util
        def load_package(alias, pkg_path):
            pkg_path = Path(pkg_path)
            init_file = pkg_path / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                alias, 
                str(init_file), 
                submodule_search_locations=[str(pkg_path)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            spec.loader.exec_module(module)
            return module
        magvit2 = load_package("MAGVITv2", str(LIB_DIR / "Show-o" / "models"))
        vae = magvit2.MAGVITv2.from_pretrained("showlab/magvitv2").to(device)
        vae.requires_grad_(False)
        vae.eval()
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    if vae_type != "lumina":
        vae.requires_grad_(False)
        vae = vae.to(device)
    return vae


@torch.no_grad()
def vae_encode_image(config, vae, image, device, vae_type: str, use_cond: bool = False):
    def get_attr(attr_name):
        if use_cond:
            return getattr(config.model, f"cond_{attr_name}", None)
        return getattr(config.model, attr_name, None)
    with torch.autocast(device_type="cuda", enabled=False):
        image = image.to(device=device, dtype=torch.float32)
        assert image.min() >= 0 - 1e-2 and image.max() <= 1 + 1e-2, f"Image values out of bounds: {image.min()}, {image.max()}"
        downscale_ratio = get_attr("downscale_ratio")
        batch_size = image.shape[0]
        latent_dim = image.shape[-1] // downscale_ratio

        if vae_type == "stable_diffusion":
            # continuous latents

            train_transforms = transforms.Compose(
                [
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            latents = vae.encode(train_transforms(image).to(vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor  # shape = (B, C, H, W)
            latents = torch.permute(latents, (0, 2, 3, 1)) # shape = (B, H, W, C)
            latents = einops.rearrange(latents, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=config.model.patching_downscale, p2=config.model.patching_downscale) # shape = (B, H*W, C*config.model.patching_downscale**2)
            return latents
        image.clamp_(0, 1)  # todo verify if needed for continuous vae
        if vae_type == "diffusers":
            if "CompVis/ldm-celebahq-256" in get_attr("use_custom_vae_ckpt"):
                image = (image * 2) - 1
            latents = vae.encode(image).latents
            discrete = vae.quantize(latents)[-1][-1]
            discrete = rearrange(discrete, "(b n) -> b n", b=batch_size)
        elif vae_type == "raw":
            discrete = rearrange((image * 255).to(torch.int64), "b c h w -> b (c h w)")
        elif vae_type == "maskgit":
            discrete = vae.get_code(image)
        elif vae_type == "taming":
            _, discrete = vae.encode(image)
        elif vae_type == "video_vqvae":
            vae.temporal_dim_length = image.shape[2]
            vae.spatial_dim_length = image.shape[3]
            discrete = vae.encode(image.to(device))  # [B, C, T, H, W]
            discrete = rearrange(discrete, "b t h w -> b (t h w)")
        elif vae_type == "VQ-8" or vae_type == "VQ-16":
            image = (image * 2) - 1
            latent, _, [_, _, discrete] = vae.encode(image)
            discrete = rearrange(discrete, "(b h w) -> b (h w)", h=latent_dim, w=latent_dim)
        elif vae_type == "lfq_128" or vae_type == "lfq_256":
            _, _, _, indices = vae(image, return_indices=True)
            discrete = rearrange(indices, "(b n) -> b n", b=batch_size)
        elif vae_type == "bsq_18":
            image = (image * 2) - 1
            quant, loss, info = vae.encode(image, skip_quantize=False)
            discrete = info["indices"]
            breakpoint()
        elif vae_type == "cosmos":
            discrete, _ = vae.encode((image.to(device) * 2) - 1)
            discrete = rearrange(discrete, "b h w -> b (h w)")
        elif "titok" in vae_type:
            discrete = vae.encode(image.to(device))[1]["min_encoding_indices"].squeeze(1)
        elif vae_type == "chameleon":
            image = (image * 2) - 1
            discrete = vae.get_image_tokens(image)
        elif vae_type == "lumina":
            breakpoint()
        elif vae_type == "magvit":
            discrete = vae.get_code(image)
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")
    return discrete


@torch.no_grad()
def vae_decode_image(config, vae, discrete, use_cond: bool = False):
    if discrete is None or (not isinstance(discrete, list) and discrete.shape[1] == 0):
        return torch.zeros(1, 3, config.data.resolution, config.data.resolution)
    
    def get_attr(attr_name):
        if use_cond:
            return getattr(config.model, f"cond_{attr_name}", None)
        return getattr(config.model, attr_name, None)

    with torch.autocast(device_type="cuda", enabled=False):
        vae_type = get_attr("vae_type")
        latent_dim = config.data.resolution // get_attr("downscale_ratio")
        if vae_type == "stable_diffusion":
            # input - (B, N, C)
            original_height = config.data.resolution // config.model.downscale_ratio
            discrete = rearrange(discrete, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=original_height, p1=config.model.patching_downscale, p2=config.model.patching_downscale)
            discrete = 1 / vae.config.scaling_factor * discrete
            image = vae.decode(discrete.to(torch.float16), return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            # image = image.detach().cpu().permute(0, 2, 3, 1).float()#.numpy()
            return image
        
        if not isinstance(discrete, list):
            discrete = discrete.to(dtype=torch.int64)
            if config.trainer.add_label and discrete.shape[-1] % 2 != 0:
                discrete = discrete[:, 1:]

        if vae_type != "chameleon" and vae_type != "lumina" and "titok" not in vae_type and discrete.shape[-1] != latent_dim ** 2:
            for test_res in (128, 256, 512, 1024):
                if (test_res // get_attr("downscale_ratio")) ** 2 == discrete.shape[-1]:
                    latent_dim = test_res // get_attr("downscale_ratio")
                    break
            else:
                raise ValueError(f"Unknown latent dimension: {latent_dim}")
        
        if vae_type == "diffusers":
            image = vae.decode(
                discrete, force_not_quantize=True, shape=(discrete.shape[0], latent_dim, latent_dim, vae.config.latent_channels)
            ).sample
            if "CompVis/ldm-celebahq-256" in get_attr("use_custom_vae_ckpt"):
                image = (image + 1) / 2
        elif vae_type == "raw":
            image = discrete / 255
            latent_dim = int(sqrt(discrete.shape[1] // 3))
            image = rearrange(image, "b (c h w) -> b c h w", c=3, h=latent_dim, w=latent_dim)
        elif vae_type == "maskgit" or vae_type == "taming":
            if not 0 <= discrete.min() and discrete.max() < get_attr("image_vocab_size"):
                raise ValueError(f"Discrete values out of bounds: {discrete.min()}, {discrete.max()}")
            assert 0 <= discrete.min() and discrete.max() < get_attr("image_vocab_size")
            image = vae.decode_code(discrete)
        elif vae_type == "video_vqvae":
            image = vae.decode(
                rearrange(discrete, "b (t h w) -> b t h w", t=vae.temporal_dim_length, h=vae.spatial_dim_length, w=vae.spatial_dim_length)
            )  # [B, T // 4, H, W]
        elif vae_type == "VQ-8" or vae_type == "VQ-16":
            image = vae.decode_code(discrete, shape=(discrete.shape[0], vae.config.codebook_embed_dim, latent_dim, latent_dim))
            image = (image + 1) / 2
        elif vae_type == "lfq_128" or vae_type == "lfq_256":
            x = discrete
            # From taming/modules/vqvae/lookup_free_quantize.py. Index -> -1/1 float
            mask = 2 ** torch.arange(vae.quantize.codebook_dim - 1, -1, -1, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0
            x = (x * 2.0) - 1.0
            x = rearrange(x, "b (h w) c -> b c h w", h=latent_dim, w=latent_dim)
            image = vae.decode(x)
            image = torch.clamp(image, 0.0, 1.0)
        elif vae_type == "bsq_18":
            quant = vae.quantize.get_codebook_entry(discrete)
            image = vae.decode(quant)
            image = torch.clamp(image, 0.0, 1.0)
        elif vae_type == "cosmos":
            image = vae.decode(discrete.reshape(discrete.shape[0], 16, 16))
            image = (image / 2) + 0.5
            image = torch.clamp(image, 0.0, 1.0)
        elif "titok" in vae_type:
            image = vae.decode_tokens(discrete.unsqueeze(1))
            image = torch.clamp(image, 0.0, 1.0)
        elif vae_type == "chameleon":
            image = vae.decode_image_tokens(discrete)
            image = (image + 1) / 2
            image = torch.clamp(image, 0.0, 1.0)
        elif vae_type == "lumina":
            # We always expect either [B, N] or [[B, N], ...]
            if not isinstance(discrete, list):
                discrete = [discrete]

            images = []
            for i in range(len(discrete)):
                for j in range(discrete[i].shape[0]):
                    images.append(torch.from_numpy(np.array(vae.decode_image(discrete[i][j].cpu().tolist()))))
                    
            image = torch.stack(images, dim=0).permute(0, 3, 1, 2) / 255
        elif vae_type == "magvit":
            image = vae.decode_code(discrete)
            image = torch.clamp(image, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

        image.clamp_(0, 1)
    return image


def auto_batch(config, fn, data):
    split_size = 32 if getattr(config.eval, "force_empty_cache", False) else 128
    if getattr(config.eval, "force_empty_cache", False):
        from model_utils import empty_device_cache
        empty_device_cache()

    if data.shape[0] > split_size:
        return torch.cat([fn(chunk) for chunk in torch.split(data, split_size, dim=0)], dim=0)
    else:
        return fn(data)

def get_image_batch(config, vae, batch, device, use_cond: bool = False):
    def get_attr(attr_name):
        if use_cond:
            return getattr(config.model, f"cond_{attr_name}", None)
        return getattr(config.model, attr_name, None)

    vae_type = get_attr("vae_type")
    if "img" in batch:
        return auto_batch(config, lambda img: vae_encode_image(config, vae, img, device, vae_type, use_cond), batch["img"])
    elif "video" in batch:
        if vae_type == "video_vqvae":
            return vae_encode_image(config, vae, batch["video"], device, vae_type, use_cond)
        else:
            return torch.cat(
                [
                    vae_encode_image(config, vae, batch["video"][:, :, frame_idx], device, vae_type, use_cond)
                    for frame_idx in range(batch["video"].shape[2])
                ],
                dim=-1,
            )
    else:
        raise ValueError(f"Unknown batch type: {batch}")

def decode_latents(config, vae, sample, use_cond: bool = False, batched: bool = True):
    if getattr(config.model, "video_model", False):
        num_frames = config.data.num_frames
        frames = torch.split(sample, sample.shape[-1] // num_frames, dim=-1)
        return torch.cat([vae_decode_image(config, vae, frame, use_cond) for frame in frames], dim=-2)
    else:
        if batched:
            return auto_batch(config, lambda s: vae_decode_image(config, vae, s, use_cond), sample)
        else:
            return np.stack([vae_decode_image(config, vae, s.unsqueeze(0), use_cond).squeeze(0) for s in sample])