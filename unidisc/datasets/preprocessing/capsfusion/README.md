
# CapsFusion

First, download the parquet files from [here](https://huggingface.co/datasets/BAAI/CapsFusion-120M/tree/main).

```bash
huggingface-cli download BAAI/CapsFusion-120M --local-dir . --repo-type dataset --resume-download
```

Next, download the images using `img2dataset`.

```bash
URL_DIR=/path/to/capsfusion/parquet
RAW_IMG_DIR=/path/to/capsfusion/wds

mkdir -p $RAW_IMG_DIR
img2dataset \
    --input_format=parquet \
    --url_list=$URL_DIR \
    --output_folder=$RAW_IMG_DIR \
    --processes_count=32 \
    --image_size=512 \
    --resize_mode=keep_ratio \
    --resize_only_if_bigger=True \
    --output_format=webdataset \
    --url_col=image_url \
    --caption_col=capsfusion \
    --enable_wandb=True 2>&1 | tee -a caps_fusion_img_download.log
```

Please see the [WebDataset](../webdataset.md) for more information on how to further process and then tokenize the WebDataset.