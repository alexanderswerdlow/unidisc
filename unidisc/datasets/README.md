Broadly speaking, we have a few types of datasets:

1. WebDataset (preferred)

These are provided e.g., by [img2dataset](https://github.com/rom1504/img2dataset) and are in the standardized [WebDataset format](https://github.com/webdataset/webdataset) consisting of a collection of `tar` files. We generally use a modified dataloader to use an Indexed WebDataset, making things like pre-processing easier.

2. Huggingface datasets

These are loaded from HF in `dataloader.py`, typically for text datasets or smaller datasets (e.g., for evaluation).

3. Tokenized TensorDict datasets

This is how most training is done to avoid the overhead of VQ-VAE tokenization during training. The data is stored as integers on disk in a [TensorDict](https://github.com/pytorch/tensordict) container, and possibly loaded into memory during training (either in the dataloader process space or `/dev/shm`).



# Synthetic generation

Generating synthetic text/image pairs has been cited in many T2I papers as critical for efficient training. Unfortunately, almost all of these datasets are proprietary, so we opt to create our own.

We first combine text captions from the following sources:

[HPSv2](https://github.com/tgxs002/HPSv2)
[ImageReward](https://github.com/THUDM/ImageReward)
[PickScore](https://github.com/yuvalkirstain/PickScore)
[simulacra-aesthetic-captions](https://github.com/JD-P/simulacra-aesthetic-captions/tree/main)
[gecko_benchmark_t2i](https://github.com/google-deepmind/gecko_benchmark_t2i) (For evaluation)

To further diversify, we use prompt an LLM to use these as inspiration for new captions, giving it a list of random entities from wordnet to incorporate into the caption. Finally, we use Stable Diffusion 3.5 medium to generate 512x512 images for each caption.

We make this process fully distributed by having each job take unused (or less commonly used) captions, and generate images in small batches.