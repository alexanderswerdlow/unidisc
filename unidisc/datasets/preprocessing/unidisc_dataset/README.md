# UniDisc Dataset

The dataset generation consists of several parts:

1. Combine seed prompts from multiple prior datasets.
2. Send off generation jobs, each of which use an LLM to augment the prompts and pass these to a diffusion model to generate images.
3. Postprocess the generated files on disk to create a parquet metadata file and finally convert the images to a WebDataset.

## Seed Prompts

We acquire seed prompts from the following datasets:

- [ImageRewardDB](https://huggingface.co/datasets/THUDM/ImageRewardDB/)
Please see [process_image_reward.py](./combine_prompts/process_image_reward.py) for the script used to process this dataset.
- [simulacra-aesthetic-captions](https://github.com/JD-P/simulacra-aesthetic-captions)
Please see [process_sac.py](./combine_prompts/process_sac.py) for the script used to process this dataset.
- [PickScore](https://github.com/yuvalkirstain/PickScore)
Please see [process_pickscore.py](./combine_prompts/process_pickscore.py) for the script used to process this dataset.
- [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2)
Please download and concatenate the `json` files in [this](https://huggingface.co/datasets/ymhao/HPDv2/tree/main/benchmark) folder.
- [Gecko Benchmark](https://huggingface.co/datasets/google-deepmind/gecko_benchmark_t2i) (For validation prompts only)
Please see [process_gecko.py](./combine_prompts/process_gecko.py) for the script used to process this dataset.

## Generation

### LLM Prompting
First, you will need to setup the LLM prompting code. We found smaller models such as `gpt-4o-mini` to not work as well as larger models and even with smaller models, we did not have sufficient funds to generate ~2.3 billion tokens. Thus, we use `langchain` to use multiple free LLMs along with local Ollama instances and paid backup APIs to distribute traffic.

To install and run Ollama:

```bash
curl -L https://ollama.com/download/ollama-linux-amd64 -o $HOME/bin/ollama
chmod +x $HOME/bin/ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then, you will need to set the corresponding Ollama server hostname/port in [llm_prompting.py](./generate/llm_prompting.py). Moreover, the code supports a round-robin mechanism, allowing you to further balance the load across multiple hosts.

Please see [llm_prompting.py](./generate/llm_prompting.py) for the code used to setup the LLM calling.

### Server
Next, the primary generation code consists of a client and server architecture to properly assign jobs. This allows for better distribution of workloads and for robust failure handling. Jobs may die for any number of reasons (pre-emption, bad GPUs, disk failures, etc.).

Please see [image_server.py](./generate/image_server.py) for the code used to setup the job server.

You may run the server using the following SLURM command:

**Note**: The client hardcodes the server hostname in this setting, so you must match `main-node` to the `hosting_node` in [generate/generate_images.py](./generate/generate_images.py).

```bash
sbatch --job-name='image_server' --mem=16G --cpus-per-task=4 --nodelist=main-node --time=4320 --partition=general --wrap "python $UNIDISC_DIR/unidisc/datasets/prompts/image_server.py" --output=$UNIDISC_DIR/outputs/generate_images/image_server.out --error=$UNIDISC_DIR/outputs/generate_images/image_server.out
```

or standalone (again, you must properly set the `hosting_node` in [generate/generate_images.py](./generate/generate_images.py)):

```bash
python $UNIDISC_DIR/unidisc/datasets/prompts/image_server.py
```

### Client

This will dispatch an sbatch array job with 128 simultaneous jobs and 1000 total jobs:

**Note**: In this case, we are using the `HPDv2.json` file we have generated. You can use any other json file you specify.

```bash
python $UNIDISC_DIR/unidisc/datasets/prompts/generate_images.py HPDv2.json --expected_samples_per_index=200 --num_workers=128 --num_chunks=1000 --max_chunk_size=512 --use_slurm --compile=True &
```

To monitor outputs:

```bash
tail -f -n1000 $UNIDISC_DIR/outputs/generate_images/image_server.out
cd $UNIDISC_DIR/outputs/generate_images && /bin/ls -t | head -10 | xargs tail -n 100
find $UNIDISC_DIR/outputs/generate_images/generated_images -type f -name "*.json" | wc -l
```


## Postprocessing

To create the parquet metadata file, run the following command:

```bash
python $UNIDISC_DIR/unidisc/datasets/preprocessing/unidisc_dataset/postprocess_dataset/convert_json_to_parquet.py /path/to/data /path/to/output.parquet
```

To package the images into a WebDataset, run the following command:

```bash
python $UNIDISC_DIR/unidisc/datasets/preprocessing/unidisc_dataset/postprocess_dataset/convert_parquet_to_wds.py --parquet_file /path/to/parquet/file --base_dir /path/to/image/folder --output_dir /path/to/output/webdataset --num_workers=64
```

