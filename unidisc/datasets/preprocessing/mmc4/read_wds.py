import webdataset as wds
import braceexpand
from torch.utils.data import IterableDataset
from webdataset import gopen
from itertools import islice
dataset = wds.WebDataset("/home/aswerdlo/hdd/data/diffusion/mmc4/core/img2dataset_imgs/07878.tar")

for sample in islice(dataset, 0, 3):
    for key, value in sample.items():
        print(key, repr(value)[:50])
    print()
    breakpoint()