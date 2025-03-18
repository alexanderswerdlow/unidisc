## Eval
To run evaluation, run the following command:
```
RUN_CC=1 RUN_DB=1 RUN_FLICKR=1 RUN_COCO=1 RUN_MEDIUM=1 RUN_AR=1 RUN_NAR=1 NUM_GPUS=1 bash scripts/small_scale_eval.sh
```

`RUN_CC`, `RUN_DB`, `RUN_FLICKR`, `RUN_COCO` all control which datasets to evaluate on.

`RUN_MEDIUM` controls whether to run experiments on the 100M or 300M ckpts.

`RUN_AR` and `RUN_NAR` control whether to run the AR and NAR ckpts.
