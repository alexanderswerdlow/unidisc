#!/bin/bash
#SBATCH --job-name=unidisc
#SBATCH --partition=preempt
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=64G
#SBATCH --constraint=L40S
#SBATCH --time=31-00:00:00
#SBATCH --output=outputs/logs/%x-%j-%N.out
#SBATCH --error=outputs/logs/%x-%j-%N.out
#SBATCH --requeue

printenv

echo "Hostname: $(hostname)"
echo "ibstatus: $(ibstatus)"
echo "ibdev2netdev: $(ibdev2netdev)"
echo "rdma device: $(rdma link)"
echo "hostnames: $(scontrol show hostnames $SLURM_JOB_NODELIST)"

export LOGLEVEL=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo "environment: $(env | grep NCCL)"

unset CUDA_VISIBLE_DEVICES
unset CUDA_LAUNCH_BLOCKING
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_DISABLE
unset NCCL_NSOCKS_PERTHREAD
unset NCCL_SOCKET_NTHREADS
unset OMP_NUM_THREADS
unset NCCL_P2P_LEVEL

ulimit -l
ulimit -a

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1
export UNIDISC_DISABLE_APEX_RMSNORM=1
export UNIDISC_ROOT_OUTPUT_DIR="outputs"
export HYDRA_RUN_DIR_NAME='large_scale_v0'

# accelerate
num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
srun --label accelerate launch \
    --multi_gpu \
    --rdzv_backend c10d \
    --machine_rank $SLURM_NODEID \
    --num_processes $num_processes \
    --num_machines $SLURM_NNODES \
    --dynamo_backend no \
    --mixed_precision no \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    main.py experiments='[large_scale_train,large_scale_train_high_res_interleaved]' nodes=2