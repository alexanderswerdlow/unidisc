#!/bin/bash

UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=1 uv run python demo/server.py experiments='[large_scale_train,large_scale_train_high_res_interleaved,eval_unified,large_scale_high_res_interleaved_inference]' \
trainer.load_from_state_dict="/home/appuser/app/pytorch_model_fsdp.bin" &

uv run python demo/client.py