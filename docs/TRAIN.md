## Small Scale Training
UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1 accelerate launch --main_process_port=$RANDOM main.py +experiments='[small_scale_train,ar]' loader.batch_size=8 wandb.name='10_11_ar' trainer.val_check_interval=100 debug=true

### Caching
UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1 accelerate launch --main_process_port=$RANDOM main.py +experiments='[small_scale_caching_train]' loader.batch_size=8 wandb.name='10_11_ar' trainer.val_check_interval=100 debug=true

## Large Scale Training

To train the large scale experiments, we recommend to set the following environment variables:
```
unset CUDA_VISIBLE_DEVICES; unset CUDA_LAUNCH_BLOCKING; unset NCCL_SOCKET_IFNAME; unset NCCL_NSOCKS_PERTHREAD; unset NCCL_SOCKET_NTHREADS; unset OMP_NUM_THREADS; unset NCCL_P2P_DISABLE; unset NCCL_P2P_LEVEL

export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1
export UNIDISC_ROOT_OUTPUT_DIR="outputs"
```

We have several stages of training. You will have to load from the previous stage's checkpoint, using either `trainer.load_from_state_dict` or by setting `HYDRA_RUN_DIR_NAME` which will automatically load the latest checkpoint/optimizer/sampler states.

1st stage:
```
accelerate launch --main_process_port=$RANDOM main.py +experiments='[large_scale_train]' debug=true loader.batch_size=8
```

2nd stage:
```
accelerate launch --main_process_port=$RANDOM main.py +experiments='[large_scale_train,large_scale_train_high_res]' debug=true loader.batch_size=4
```

3rd stage:
```
accelerate launch --main_process_port=$RANDOM main.py +experiments='[large_scale_train,large_scale_train_high_res_interleaved]' debug=true loader.batch_size=1
```

## SLURM Training
To train on SLURM, you have two options:

1. sbatch directly from the script: `sbatch scripts/train_large_scale_slurm.sh`. Please edit the script to set the correct number of nodes, gpus per node, and other parameters.

2. Use sbatch through hydra_submitit_launcher. This is recommended for smaller experiments, partially hyperparameter sweeps, although it has been setup to work for multi-node training.

First, uncomment everything under `hydra.launcher` in `configs/config.yaml`. Next, uncomment this at the top of `configs/config.yaml`:
```
- override hydra/launcher: submitit_slurm
```

Next, run the following (change to `uv pip install` if you are using uv):

```
pip install 'git+ssh://git@github.com/alexanderswerdlow/hydra.git@working_ci#egg=hydra-core'
pip install 'git+ssh://git@github.com/alexanderswerdlow/hydra.git@working_ci#egg=hydra-submitit-launcher&subdirectory=plugins/hydra_submitit_launcher'
```

To use hydra_submitit_launcher, append the following to any command:

`devices=8 nodes=1 partition=general --multirun`

You may modify `devices`/`nodes`/`partition` based on your set. See `hydra.launcher` in `configs/config.yaml` to set additional SLURM parameters.


Both of the above methods use 1 task per node. Some SLURM clusters prefer to use 1 task per GPU. Please see this amazing guide for more details: `https://github.com/stas00/ml-engineering`. In short, be careful in making this change as there are many subtle differences (e.g., passing signals between processes, how checkpointing/requeing/error handling works, etc.)

## TPU Training

TODO: Add more documentation for TPU training. Our codebase is setup to use TPUs through `torch_xla`, taking advantage of SPMD.

Misc TPU Notes:
- SPMD has a very confusing setup for SPMD and pretends that each node is a single device. See `decoupled_utils.py` for some of the logic used to handle this. Moreover, getting the proper rank can only be done after spawning SPMD, so we need to handle this as a lot of code needs the device rank on import.

## Attention Kernels
The codebase currently supports the following:

- PyTorch SDPA (Including CUDNN Flash, Regular Flash, etc.)
- PyTorch FlexAttention (For interleaved/caching training)
- Flash Attention 2/3 (With varying kernels, e.g., packed/non-packed/varlen depending on the use case)
- TorchXLA SPMD FlashAttention (For TPU training)

Generally, we use PyTorch SDPA (preferrably the CUDDN Kernel which you can force with `UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1`) and FlexAttention for all interleaved/caching training, setting `trainer.compile=true` to improve MFU. We found this to be similar in speed to Flash Attention 2, which at the time of development did not have good compile support.

# Config Notes
- `data.enable_cuda_in_tensordict_collate` requires `loader.num_workers = 0`
- Enable `data.move_tensordict_to_shm` to speed up dataloading (keeping `data.keep_tensordict_on_disk = true`), assuming you have enough system memory. Separately, you can disable `data.keep_tensordict_on_disk`, but this will load the entire tensordict into each dataloader worker process (e.g., given 2 GPUs and 4 workers, this will load 8 tensordicts into system memory) which is not possible on larger datasets. Optionally, you can set `+data.shm_path='/path/to/data'` to use a custom path, e.g., to use a scratch disk instead of system memory.
- You may need to set `NCCL_IB_DISABLE=1` or `NCCL_P2P_DISABLE=1` depending on the system configuration. Setting `NCCL_P2P_LEVEL=NVL` is recommended if the system has NVLink.
- To use `data.enable_cuda_in_tensordict_collate=true`, you must also set `data.force_mp_spawn=false` and `loader.num_workers>0`.
- Resuming from checkpoint can be done in multiple ways. Double check the log output to verify the correct checkpoint is being loaded. The tl;dr is the following: If you use `hydra_submitit_launcher` or set `HYDRA_RUN_DIR_NAME`, it will automatically load the latest checkpoint/optimizer/sampler states.
- To resume from weights only set: `trainer.load_from_state_dict="/path/to/weights.bin"`