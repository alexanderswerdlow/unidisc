# Cambrian

First, download the Cambrian dataset from [here](https://huggingface.co/datasets/cambrian/cambrian-10m).

```bash
huggingface-cli download nyu-visionx/Cambrian-10M --local-dir . --repo-type dataset --resume-download
```

Next, untar all the *.tar.gz files.

Next, precompute the tokens:

**_Note:_** If you are on a SLURM cluster, you can replace `accelerate launch` with:

```bash
sbatch --time=2-00:00:00 --array=0-100%25 --cpus-per-gpu=12 --mem-per-gpu=100G --nodes=1 --gpus-per-node=1 --partition=preempt --job-name=cambrian_precompute_tokens scripts/precompute_tokens_slurm.sh
```

**_Note:_** If you want to only generate a subset of the tokens, append e.g., `data.n_train_samples=200` to the command.

Finally, to tokenize the dataset, run:

```bash
accelerate launch models/datasets/precompute_tokens.py +experiments='[generated_images,tokenize,vq16_t2i]' data.token_output_dir="/path/to/token_output_dir" data.resolution=512 data.use_chameleon=false model.img_length=3072 data.block_size=3072 loader.batch_size=16 data.train='cambrian' data.raw_data_dir='/path/to/cambrian/jsons/Cambrian10M.jsonl' +model.text_vocab_size=32001 data.img_token_shift=32001 +data.use_identity_collate=true loader.num_workers=2 data.split_dataset=true +data.save_tmp_interval=3600 +data.use_slow_tokenizer=true +data.add_image_token=true
```

Now that the tokenization is complete, if it was done over multiple GPUs/nodes, you must combine the tensordicts on disk. If the to

```bash
python models/datasets/combine_token_dicts.py "/path/to/token_output_dir" --move_files --delete_after_combining --mem_efficient
```

**_Note:_** You may wish to add the `--allow_tmp` flag to the command if the tokenization was only partially completed (e.g., due to a SLURM job being preempted). In this case, the tokenization saves intermediate checkpoints with a `tmp_` prefix.