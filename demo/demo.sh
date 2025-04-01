#!/bin/bash

ls -la /home/appuser/app/ckpts
ls -la /home/appuser/app/ckpts/unidisc_interleaved

PYTHONUNBUFFERED=1 uv run python demo/client.py &

PYTHONUNBUFFERED=1 UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1 uv run python demo/server.py experiments='[large_scale_train,large_scale_train_high_res_interleaved,eval_unified,large_scale_high_res_interleaved_inference]' \
trainer.load_from_state_dict="/home/appuser/app/ckpts/unidisc_interleaved/unidisc_interleaved.pt" model.use_custom_vae_ckpt="/home/appuser/app/ckpts/unidisc_interleaved/vq_ds16_t2i.pt" &

wait