<div align="center">
<br>
<img src="docs/images/banner.webp" width="1000">
<h3>Unified Multimodal Discrete Diffusion</h3>

[Alexander Swerdlow](https://aswerdlow.com/)<sup>1&#42;</sup>&nbsp;
[Mihir Prabhudesai](https://mihirp1998.github.io/)<sup>1&#42;</sup>&nbsp;
[Siddharth Gandhi](hhttps://www.ssgandhi.com/)<sup>1</sup>&nbsp;
[Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)<sup>1</sup>&nbsp;
[Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)<sup>1</sup>&nbsp;
<br>

<sup>1</sup> Carnegie Mellon University&nbsp;
 
[![ArXiv](https://img.shields.io/badge/ArXiv-<2503.20853>-<COLOR>.svg)](https://arxiv.org/abs/2503.20853) [![Webpage](https://img.shields.io/badge/Webpage-UniDisc-<COLOR>.svg)](https://unidisc.github.io/) 

<!-- [![Demo](https://img.shields.io/badge/Demo-Custom-<COLOR>.svg)](https://huggingface.co/spaces/todo) -->
  
</div>

## Hugging Face models

The UniDisc checkpoints are available on Hugging Face:
* [Interleaved Image-Text](https://huggingface.co/aswerdlow/unidisc_interleaved)
* [Non-Interleaved (Static)](https://huggingface.co/aswerdlow/unidisc_non_interleaved)

## Getting Started

To install the dependencies, run:
```bash
git submodule update --init --recursive
uv sync --no-group dev
uv sync
```

For a more detailed installation guide, please refer to [INSTALL.md](docs/INSTALL.md).

## Data

See [DATA.md](docs/DATA.md) for details on how to download and preprocess the datasets. We provide processing scripts and instructions for all of the used datasets. Additionally, we release a synthetic dataset available [here](https://huggingface.co/datasets/aswerdlow/unidisc_hq) and the corresponding [generation scripts](unidisc/datasets/preprocessing/unidisc_dataset/README.md) as well as the raw data.

## Training

See [TRAIN.md](docs/TRAIN.md) for training commands.

## Inference

<!-- Inference demo for **TODO**.
```
TODO
``` -->
<!-- <img src="docs/todo.png" width="1000"> -->


Interactive demo:
```
python demo/server.py
python demo/client_simple_fasthtml.py
```


## Training

See [TRAINING.md](docs/TRAINING.md) for details.

## Evaluation

See [EVAL.md](docs/EVAL.md) for details.


### Citation
To cite our work, please use the following:
```
@article{TODO,
  title={TODO},
  author={TODO},
  journal={arXiv preprint arXiv:TODO},
  year={TODO}
}
```

## Credits

This repository is built on top of the following repositories:

- [MDLM](https://github.com/kuleshov-group/mdlm)
- [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X)
