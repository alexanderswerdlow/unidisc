from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import glob
import json
import os
import subprocess
import tarfile
from pathlib import Path
from viztracer import VizTracer

import webdataset as wds
from tqdm import tqdm
import tempfile
import shutil
from io import BytesIO
from PIL import Image
import socket

prefix_path = Path("/scratch")

output_tar_dir = prefix_path / Path("journeydb_wds/train")
output_tar_dir.mkdir(parents=True, exist_ok=True)
output_tar_path = str(output_tar_dir) + f"/output_dataset_%06d.tar"
input_tgz_path = prefix_path / Path("journeydb/data/train/train_anno_realease_repath.jsonl.tgz")
tgz_dir = prefix_path / Path("journeydb/data/train/imgs")

jsonl_data = []
print(f"Opening {input_tgz_path}")
with tarfile.open(input_tgz_path, "r:gz") as tar:
    for member in tar.getmembers():
        if member.isfile() and member.name.endswith('.jsonl'):
            f = tar.extractfile(member)
            for line in f:
                jsonl_data.append(json.loads(line))

tgz_files = list(tgz_dir.glob("*.tgz"))
prefix_to_tgz = {tgz_file.stem: tgz_file for tgz_file in tgz_files}
cached_tgz_files = {prefix: tarfile.open(tgz_file, "r:gz") for prefix, tgz_file in prefix_to_tgz.items()}

print(f"Extracted {len(jsonl_data)} samples")
prefix_to_samples = {}
for sample in jsonl_data:
    img_path = sample["img_path"].removeprefix("./")
    prefix = img_path.split('/')[0]
    if prefix not in prefix_to_samples:
        prefix_to_samples[prefix] = []
    prefix_to_samples[prefix].append(sample)

print(f"Prefix to samples: {len(prefix_to_samples)}")
profile = False
max_samples = 3 if profile else None

mem_path = Path('/dev/shm/aswerdlo')
mem_path.mkdir(parents=True, exist_ok=True)
resolution = 1024

def process_prefix(samples, tgz_file_path, output_tar_path, mem_path, max_samples, worker_id):
    with tarfile.open(tgz_file_path, "r:gz") as tgz:
        tmpdirname = tempfile.mkdtemp(dir=mem_path)
        try:
            print(f"Extracting {tgz_file_path} to {tmpdirname}")
            tgz.extractall(path=tmpdirname)
            output_path = output_tar_path.removesuffix('.tar') + f"_{worker_id}.tar"
            print(f"Extracted {tgz_file_path} to {tmpdirname}, writing to {output_path}")
            with wds.ShardWriter(output_path, maxsize=500*1024*1024) as sink:
                for idx, sample in tqdm(enumerate(samples), desc=f'Worker {worker_id}', total=len(samples)):
                    if max_samples is not None and idx >= max_samples:
                        break

                    if idx == 0 or idx % 1000 == 0:
                        print(f"Worker {worker_id} processed {idx} samples")

                    img_path = sample["img_path"].removeprefix("./")
                    file_path = os.path.join(tmpdirname, img_path)
                    if os.path.exists(file_path):
                        try:
                            img = Image.open(file_path)
                            width, height = img.size
                            if width > height:
                                left = (width - height) / 2
                                top = 0
                                right = (width + height) / 2
                                bottom = height
                            else:
                                left = 0
                                top = (height - width) / 2
                                right = width
                                bottom = (height + width) / 2
                            img = img.crop((left, top, right, bottom))
                            img = img.resize((resolution, resolution), Image.LANCZOS)

                            img_byte_arr = BytesIO()
                            img.save(img_byte_arr, format='JPEG', quality=95)
                            img_data = img_byte_arr.getvalue()

                            key = Path(img_path).stem
                            sample_dict = {
                                "__key__": key,
                                "txt": sample['Task2']["Caption"],
                                "jpg": img_data
                            }
                            sink.write(sample_dict)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Skipping bad sample {file_path}: {e}")
        finally:
            shutil.rmtree(tmpdirname)

with VizTracer(output_file="result2.json") if profile else nullcontext():
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for worker_id, (prefix, samples) in enumerate(tqdm(sorted(prefix_to_samples.items()))):
            if prefix not in prefix_to_tgz:
                print(f"Prefix {prefix} not found in tgz files")
                continue
            tgz_file_path = prefix_to_tgz[prefix]
            futures.append(executor.submit(process_prefix, samples, tgz_file_path, output_tar_path, mem_path, max_samples, worker_id))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing prefix: {e}")

# Rename the tar files to remove the worker_id prefix
for worker_id in range(len(prefix_to_samples)):
    for tar_file in glob.glob(str(output_tar_dir / f"output_dataset_{worker_id:06d}_*.tar")):
        new_name = tar_file.replace(f"_{worker_id:06d}_", "_")
        counter = 0
        while os.path.exists(new_name):
            counter += 1
            new_name = new_name.replace(".tar", f"_{counter:06d}.tar")
        os.rename(tar_file, new_name)

for tgz in cached_tgz_files.values():
    tgz.close()

tar_files = glob.glob(str(output_tar_dir / "*.tar"))

if tar_files:
    os.chdir(output_tar_dir)
    command = ["widsindex", "create"] + [os.path.basename(f) for f in tar_files]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("widsindex command executed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running widsindex command: {e.returncode}\n{e.stderr}")
    os.chdir(Path(__file__).parent)
else:
    print("No tar files found in the output directory.")

print("Testing WebDataset reading:")

dataset = wds.WebDataset(tar_files).decode("rgb")
for i, sample in enumerate(dataset):
    if i >= 5:  # Print details for the first 5 samples
        break
    print(f"Sample {i + 1}:")
    print(f"Key: {sample['__key__']}")
    print(f"Image size: {sample['jpg'].size}")
    print(f"Text: {sample['txt']}")
    print()