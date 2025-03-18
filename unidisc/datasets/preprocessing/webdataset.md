# WebDataset

Many of our datasets are stored in an intermediate format called [WebDataset](https://github.com/webdataset/webdataset). This is simply a tar file containing a set of files in a specific format. Although typically WebDataset is used for distributed training where data repitition is not a large issue (e.g., with an IterableDataset), for some pre-processing tasks it is easier to have an indexed dataset.

For this reason, you must run the following command to create an index, once inside the directory containing the tar files:

**Note:** you may want/need to install the fork with `pip install git+ssh://git@github.com/alexanderswerdlow/webdataset.git@wip` to get a faster `widsindex` command.

```bash
widsindex create *.tar
```


To read a WebDataset, you can use the following code:

```python
import webdataset as wds
import braceexpand
from tqdm import tqdm
shards = braceexpand.braceexpand('/scratch/data_cc3m/cc3m-train-{0000..0575}.tar')
dataset = wds.WebDataset(shards, shardshuffle=True).shuffle(5000)

for b in tqdm(dataset):
    pass
```




## Tokenization

Next, precompute the tokens:

**_Note:_** If you are on a SLURM cluster, you can replace `accelerate launch` with:

```bash
sbatch --time=2-00:00:00 --array=0-100%25 --cpus-per-gpu=12 --mem-per-gpu=100G --nodes=1 --gpus-per-node=1 --partition=preempt --job-name=cambrian_precompute_tokens scripts/precompute_tokens_slurm.sh
```

**_Note:_** If you want to only generate a subset of the tokens, append e.g., `data.n_train_samples=200` to the command.

**_Note:_** Set `data.block_size=128` if you want a different maximum token length.

**_Note:_** `model.text_vocab_size` and `data.img_token_shift` are based on the text tokenizer used, in this case `Llama-2-7b-hf`.

**_Note:_** Set the resolution as desired (e.g., 256, 512, 1024, etc.).

Finally, to tokenize the dataset, run:

```bash
accelerate launch models/datasets/precompute_tokens.py +experiments='[webdataset,tokenize,vq16_t2i]' data.token_output_dir="/path/to/token_output_dir" data.resolution=512 data.use_chameleon=false loader.batch_size=16 data.raw_data_dir='/path/to/cambrian/jsons/Cambrian10M.jsonl' +model.text_vocab_size=32001 data.img_token_shift=32001 +data.use_identity_collate=true loader.num_workers=2 data.split_dataset=true +data.save_tmp_interval=3600 +data.use_slow_tokenizer=true +data.add_image_token=true
```

Now that the tokenization is complete, if it was done over multiple GPUs/nodes, you must combine the tensordicts on disk. If the to

```bash
python models/datasets/combine_token_dicts.py "/path/to/token_output_dir" --move_files --delete_after_combining --mem_efficient
```

**_Note:_** You may wish to add the `--allow_tmp` flag to the command if the tokenization was only partially completed (e.g., due to a SLURM job being preempted). In this case, the tokenization saves intermediate checkpoints with a `tmp_` prefix.