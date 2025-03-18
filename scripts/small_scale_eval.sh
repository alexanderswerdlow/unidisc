# MIN_WAIT=60 MAX_WAIT=300 bash scripts/osync.sh --on-changes --initiator=/home/aswerdlo/hdd/data/unidisc/ckpts/sync --target=ssh://mprabhud@grogu//grogu/user/mprabhud/aswerdlo/unidisc/ckpts/sync

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=$CONDA_PREFIX
export UNIDISC_FORCE_CUDNN_SPDA_CONTEXT=0
export NUM_GPUS=${NUM_GPUS:-4}
export CONSTRAINT="L40|L40S|A100_40GB|A100_80GB|6000Ada|A6000|A4500"
export MEM_PER_GPU=32
export CPUS_PER_GPU=8

export CKPT_DIR='/home/aswerdlo/repos/unidisc_arxiv'

RUN_NAR=${RUN_NAR:-0}
RUN_AR=${RUN_AR:-0}
RUN_CC=${RUN_CC:-0}
RUN_DB=${RUN_DB:-0}
RUN_FLICKR=${RUN_FLICKR:-0}
RUN_COCO=${RUN_COCO:-0}
RUN_MEDIUM=${RUN_MEDIUM:-0}

common_args=(\
debug=true \
model=$([[ "$RUN_MEDIUM" -eq 1 ]] && echo "medium" || echo "small") \
loader.eval_batch_size=$([[ "$RUN_MEDIUM" -eq 1 ]] && echo "3" || echo "24") \
trainer.compile=true \
+trainer.forced_keys='[eval.cfg,eval.unconditional_fid,sampling.predictor,data.fid_dataset,sampling.sampling_step_frac]' \
model.force_optimized_native_attn=false \
wandb.project='unidisc-jan-eval-ablations' \
partition=preempt \
wandb.tags='[11_12_fid_ar_v2]' \
eval.fid_samples=16384 \
sampling.predictor=maskgit \
sampling.sampling_step_frac='0.05' \
eval.cfg=2 \
trainer.compile=false \
slurm_name="${USER}_ablations_nar" \
mem_per_gpu=$MEM_PER_GPU \
cpus_per_gpu=$CPUS_PER_GPU \
devices=$NUM_GPUS \
constraint=$CONSTRAINT \
partition=general)

common_a_args=(\
+experiments='[small_scale_train,paired_standalone_fid_eval,master_eval,fid_hf]' data.fid_dataset="sayakpaul/coco-30-val-2014")

common_b_args=(\
+experiments='[small_scale_train,paired_standalone_fid_eval,master_eval,fid_hf]' data.fid_dataset="nlphuji/flickr30k")

common_c_args=(\
+experiments='[small_scale_train,paired_standalone_fid_eval,master_eval,fid_cc12m]')

common_d_args=(\
+experiments='[small_scale_train,paired_standalone_fid_eval,master_eval,fid_datacomp1b]')

if [ "$RUN_MEDIUM" -eq 1 ]; then
    NAR_CKPT="$CKPT_DIR/300m_nar.safetensors"
    AR_CKPT="$CKPT_DIR/300m_ar.safetensors"
else
    AR_CKPT="$CKPT_DIR/115m_ar.safetensors"
    NAR_CKPT="$CKPT_DIR/115m_nar.safetensors"
fi

echo "RUN_AR: ${RUN_AR}, RUN_NAR: ${RUN_NAR}, RUN_MEDIUM: ${RUN_MEDIUM}"
echo "RUN_CC: ${RUN_CC}, RUN_DB: ${RUN_DB}, RUN_FLICKR: ${RUN_FLICKR}, RUN_COCO: ${RUN_COCO}"
echo "NAR_CKPT: ${NAR_CKPT}"
echo "AR_CKPT: ${AR_CKPT}"

if [ "$RUN_AR" -eq 1 ]; then
    if [ "$RUN_COCO" -eq 1 ]; then
        python main.py "${common_a_args[@]}" "${common_args[@]}" $@ parameterization=ar trainer.compile=false wandb.name="1_2_ar_60k" \
        trainer.load_from_state_dict="$AR_CKPT" --multirun > /dev/null 2>&1 &
    fi

    if [ "$RUN_FLICKR" -eq 1 ]; then
        python main.py "${common_b_args[@]}" "${common_args[@]}" $@ parameterization=ar trainer.compile=false wandb.name="1_2_ar_60k" \
        trainer.load_from_state_dict="$AR_CKPT" --multirun > /dev/null 2>&1 &
    fi

    if [ "$RUN_CC" -eq 1 ]; then
        echo "RUN_CC: ${RUN_CC}"
        python main.py "${common_c_args[@]}" "${common_args[@]}" $@ parameterization=ar trainer.compile=false wandb.name="1_2_ar_60k" \
        trainer.load_from_state_dict="$AR_CKPT" --multirun
    fi

    if [ "$RUN_DB" -eq 1 ]; then
        python main.py "${common_d_args[@]}" "${common_args[@]}" $@ parameterization=ar trainer.compile=false wandb.name="1_2_ar_60k" \
        trainer.load_from_state_dict="$AR_CKPT" --multirun > /dev/null 2>&1 &
    fi
fi

if [ "$RUN_NAR" -eq 1 ]; then
    if [ "$RUN_COCO" -eq 1 ]; then
        python main.py "${common_a_args[@]}" "${common_args[@]}" $@ wandb.name="1_2_nar_325k" \
        trainer.load_from_state_dict="$NAR_CKPT" --multirun > /dev/null 2>&1 &
    fi

    if [ "$RUN_FLICKR" -eq 1 ]; then
        python main.py "${common_b_args[@]}" "${common_args[@]}" $@ wandb.name="1_2_nar_325k" \
        trainer.load_from_state_dict="$NAR_CKPT" --multirun > /dev/null 2>&1 &
    fi

    if [ "$RUN_CC" -eq 1 ]; then
        python main.py "${common_c_args[@]}" "${common_args[@]}" $@ wandb.name="1_2_nar_325k" \
        trainer.load_from_state_dict="$NAR_CKPT" --multirun > /dev/null 2>&1 &
    fi

    if [ "$RUN_DB" -eq 1 ]; then
        python main.py "${common_d_args[@]}" "${common_args[@]}" $@ wandb.name="1_2_nar_325k" \
        trainer.load_from_state_dict="$NAR_CKPT" --multirun > /dev/null 2>&1 &
    fi
fi
