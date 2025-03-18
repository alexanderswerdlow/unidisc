# PixelProse

To download the metadata, run:

```bash
huggingface-cli download tomg-group-umd/pixelprose --repo-type dataset --local-dir .
```


To download the images, run:

```bash
input_folder='/path/to/pixelprose/data'
img2dataset --url_list "$input_folder"  --input_format "parquet" \
--url_col "url" --caption_col "vlm_caption" --output_format webdataset \
--output_folder pixelprose_data --processes_count 16 --thread_count 32 \
--save_additional_columns '["original_caption", "uid"]' --enable_wandb True --image_size 256 --output_folder "/path/to/output/folder" --resize_mode center_crop
```

Please see the [WebDataset](../webdataset.md) for more information on how to further process and then tokenize the WebDataset.