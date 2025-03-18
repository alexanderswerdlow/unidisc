# JourneyDB

To download the dataset, run:

```bash
huggingface-cli download JourneyDB/JourneyDB --repo-type dataset --local-dir . --include "data/train/train_anno.jsonl.tgz"
huggingface-cli download JourneyDB/JourneyDB --repo-type dataset --local-dir . --include "data/train/train_anno_realease_repath.jsonl.tgz"
```

To convert the dataset to a WebDataset, run:

```bash
python unidisc/datasets/preprocessing/journeydb/create_wds.py
```

Please see the [WebDataset](../webdataset.md) for more information on how to further process and then tokenize the WebDataset.