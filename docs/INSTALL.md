# Installation Guide

First, if you did not clone with submodules (`--recurse-submodules`), run:
```bash
git submodule update --init --recursive
```

## UV (Recommended)

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Note:** You may need to set `CUDA_HOME` and have it pointing to a valid CUDA 12.x installation. To use a different CUDA version, please change the sources in `pyproject.toml` (and the `torch`/`torchvision` versions). See [this guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for more details.

Next, run:
```bash
uv sync --no-group dev
uv sync # To install all dependencies: uv sync --all-groups
```

If it succeeded, that's it! Prefix any commands with `uv run` to use the environment.

E.g., `accelerate launch main.py` -> `uv run accelerate launch main.py`

or

`python main.py` -> `uv run python main.py`

Alternatively, you can activate the environment manually and run as follows:
```bash
uv sync
source .venv/bin/activate
python main.py
```

## Pip / Anaconda / Micromamba

### Step 1: Optional: Install Micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX="~/micromamba"
eval "$(~/bin/micromamba shell hook -s posix)"
alias conda='micromamba'

### Step 2:  Create conda environment
`conda create -n unidisc python=3.10`
`conda config --add channels conda-forge`

If using micromamba:
`micromamba config append channels conda-forge`

If using conda:
`conda config --set channel_priority flexible`

### Step 3: Setup CUDA
To use existing an existing cuda installation:
`export CUDA_HOME=...`


To install CUDA w/conda or micromamba:
```
conda install cuda cuda-nvcc -c nvidia/label/cuda-12.4.1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=$CONDA_PREFIX`
```

### Step 4: Install PyTorch

If using conda/micromamba CUDA:
`conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia/label/cuda-12.4.1 -c nvidia`

Otherwise, install from PyPI:
`pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/test/cu124`

### [Optional] Install Flash Attention
```
pip install --upgrade packaging ninja pip wheel setuptools
pip install flash-attn --no-build-isolation
```

#### [Optional] Flash Attn 3
```
git clone https://github.com/Dao-AILab/flash-attention
cd hopper; python setup.py install
```

To test: `pip install pytest; export PYTHONPATH=$PWD; pytest -q -s test_flash_attn.py`


### Step 5: Install Other Dependencies

```
pip install -r docs/reqs/requirements.txt
pip install -r docs/reqs/requirements_eval.txt
pip install --force-reinstall --no-deps -r docs/reqs/forked_requirements.txt
pip install tensordict-nightly 'git+https://github.com/huggingface/accelerate' --force-reinstall --no-deps
pip install 'git+https://github.com/huggingface/datasets' 'git+https://github.com/huggingface/transformers' 
pip install 'git+ssh://git@github.com/alexanderswerdlow/hydra.git@working_ci#egg=hydra-core'
pip install 'git+ssh://git@github.com/alexanderswerdlow/hydra.git@working_ci#egg=hydra-submitit-launcher&subdirectory=plugins/hydra_submitit_launcher'
pip install 'numpy>2.0.0'
```

### Misc / Troubleshooting
- This may be required if you don't install CUDA through conda: `conda install gcc_linux-64==12.4.0 gxx_linux-64===12.4.0`
- Other non-forked deps [only if they show as not installed]: `pip install hydra-core webdataset`
- Dependencies you may need for non-core code:


```bash
pip install flask werkzeug sentence_transformers ngrok opencv-python lpips simple_slurm typer ftfy bitsandbytes sentencepiece flask requests peft transformers deepspeed langchain langchain_groq langchain_core langchain_community langchain-openai  git+https://github.com/microsoft/mup.git
pip install fairseq --no-deps
```