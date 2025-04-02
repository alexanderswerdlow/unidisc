#!/bin/bash

logfile="demo.log"

# Redirect output to log file while still printing to console
exec > >(tee -a "$logfile") 2>&1

find /home/appuser/app/ckpts -type f -o -type d | sort

PYTHONUNBUFFERED=1 uv run python demo/client.py &

PYTHONUNBUFFERED=1 UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1 uv run python demo/server.py experiments='[large_scale_train,large_scale_train_high_res_interleaved,eval_unified,large_scale_high_res_interleaved_inference]' \
    trainer.load_from_state_dict="/home/appuser/app/ckpts/unidisc_interleaved/unidisc_interleaved.pt" \
    model.use_custom_vae_ckpt="/home/appuser/app/ckpts/unidisc_interleaved/vq_ds16_t2i.pt" &

wait