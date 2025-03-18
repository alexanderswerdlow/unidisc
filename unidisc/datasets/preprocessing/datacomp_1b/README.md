# Datacomp 1b

We use ReComp DataComp1B for a set of re-captioned, high-quality image/text pairs.

Please download the metadata from [here](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B):

```bash
huggingface-cli download UCSC-VLAA/Recap-DataComp-1B --repo-type dataset --local-dir .
```

Then optionally split the parquet files into smaller chunks:

```bash
python split_parquet.py
```

Then download the actual images into a WebDataset format using [img2dataset](https://github.com/rom1504/img2dataset). Change the resolution, number of processes, number of thread, and input/output folders as needed.

```bash
input_folder='/path/to/recap-datacomp-1b/data/train_data_split/split_0'
img2dataset --url_list "$input_folder"  --input_format "parquet" \
--url_col "url" --caption_col "re_caption" --output_format webdataset \
--output_folder recap_datacomp_1b_data --processes_count 16 --thread_count 128 \
--save_additional_columns '["org_caption"]' --enable_wandb True --image_size 256 --output_folder "/scratch/data/datacomp_1b_${input_folder##*/}" --resize_mode center_crop
```

Please see the [WebDataset](../webdataset.md) for more information on how to further process and then tokenize the WebDataset.