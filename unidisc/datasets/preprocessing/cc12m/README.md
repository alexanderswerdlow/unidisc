# CC12M

We used the very helpful [cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds) dataset to avoid having to download the images from the original source.

## Downloading the dataset

```bash
huggingface-cli download pixparse/cc12m-wds --local-dir . --repo-type datasetd
huggingface-cli download pixparse/cc3m-wds --local-dir . --repo-type dataset
```

widsindex create *train*.tar