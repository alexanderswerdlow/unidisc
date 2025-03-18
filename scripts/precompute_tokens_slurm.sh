#!/bin/bash
#SBATCH --job-name=precompute_tokens
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%A_%a_%n_log.out
#SBATCH --signal=B:USR2@600

echo "ibstatus: $(ibstatus)"
echo "ibdev2netdev: $(ibdev2netdev)"
echo "rdma device: $(rdma link)"

unset NCCL_P2P_LEVEL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

export LOGLEVEL=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo "environment: $(env | grep NCCL)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"

trap 'echo "SIGUSR2"; \
if [ -n "$SLURM_ARRAY_JOB_ID" ]; then echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"; fi; \
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"; fi; \
# ps auxww | grep $USER; \
pid=$(pgrep -u $USER -f "python.*(accelerate|torchrun|deepspeed|distributed\.run).*"); \
echo "Found parent PIDs: $pid"; \
for p in $pid; do \
    echo "Parent PID has cmd: $(ps -p $p -o cmd=)"; \
    children=$(pgrep -P $p); \
    echo "Children: $children"; \
    if [ -n "$children" ]; then \
    for child in $children; do \
        ppid=$(ps -o ppid= -p $child | tr -d " ")
        if [ "$ppid" -eq "$p" ]; then
        echo "Killing direct child process: PID $child with cmd: $(ps -p $child -o cmd=)"
        kill -USR2 $child &
        else
        echo "Skipping non-direct child process: PID $child with PPID $ppid"
        fi
    done; \
    echo "Sent kill signals to children of $p"; \
    else \
    echo "No children found for $p"; \
    fi; \
done; \
wait;' SIGUSR2

num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
echo "num_processes: $num_processes"
srun --label accelerate launch \
    --rdzv_backend c10d \
    --machine_rank $SLURM_NODEID \
    --num_processes $num_processes \
    --num_machines $SLURM_NNODES \
    --dynamo_backend no \
    --mixed_precision no \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    "$@"
