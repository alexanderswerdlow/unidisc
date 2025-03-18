import pandas as pd
from pathlib import Path
import tarfile
import glob
import pickle
from tqdm import tqdm
import json
import multiprocessing as mp

mmc4_mapping_parquet = "/path/mmc4/fewer_faces/concatenated_mmc4.parquet"

mapping = pd.read_parquet(mmc4_mapping_parquet)
print("Finished loading mapping")
mapping = mapping[['url', 'tar_filepath', 'key']]
mapping = mapping.set_index("url")
mapping = mapping.sort_values('tar_filepath')
print("Finished sorting mapping")

def process_tar_file(tar_filepath):
    try:
        with tarfile.open(tar_filepath) as tar:
            return tar_filepath, set(tar.getnames())
    except:
        return tar_filepath, set()

def get_cache():
    from constants import UNIDISC_DIR
    
    cache_path = UNIDISC_DIR / "archive" / "tar_contents_cache.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    _tar_contents_cache = {}
    unique_tar_filepaths = mapping['tar_filepath'].unique()
    
    # Use all available CPU cores
    with mp.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_tar_file, unique_tar_filepaths),
            total=len(unique_tar_filepaths),
            desc="Building tar contents cache"
        ))
    
    # Convert results to dictionary
    _tar_contents_cache = dict(results)
            
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(_tar_contents_cache, f)
        
    return _tar_contents_cache

_tar_contents_cache = get_cache() 
jsonl_files = glob.glob("/path/mmc4/fewer_faces/shards/*.jsonl")
output_dir = Path("/path/mmc4/fewer_faces/filtered_shards")
output_dir.mkdir(parents=True, exist_ok=True)

data_items = []
for file in tqdm(jsonl_files, desc="Reading JSONL files"):
    with open(file, 'r') as f:
        for line in tqdm(f, desc=f"Processing {file}"):
            data = json.loads(line)
            has_valid_image = False
            tar_filepath = None  # Will hold the tar_filepath of the first valid image

            for image_info in data["image_info"]:
                try:
                    mapped_to_ = mapping.loc[image_info["raw_url"]]
                    if isinstance(mapped_to_, pd.Series):
                        mapped_to_ = [mapped_to_]
                    elif isinstance(mapped_to_, pd.DataFrame):
                        mapped_to_ = [row for _, row in mapped_to_.iterrows()]
                    else:
                        mapped_to_ = [mapped_to_]
                    for mapped_to in mapped_to_:
                        tar_filepath_candidate = mapped_to["tar_filepath"]
                        relative_tar_filepath_candidate = None
                        if "relative_tar_filepath" in mapped_to:
                            relative_tar_filepath_candidate = mapped_to["relative_tar_filepath"]

                        key = mapped_to["key"]
                        if f"{key}.jpg" in _tar_contents_cache[tar_filepath_candidate]:
                            has_valid_image = True
                            tar_filepath = tar_filepath_candidate
                            image_info["tar_filepath"] = tar_filepath
                            if relative_tar_filepath_candidate is not None:
                                image_info["relative_tar_filepath"] = relative_tar_filepath_candidate
                            image_info["key"] = key
                            break
                    if has_valid_image:
                        break
                except KeyError:
                    continue

            if has_valid_image and tar_filepath is not None:
                data_items.append((tar_filepath, line))

# Sort data_items by tar_filepath
data_items.sort(key=lambda x: x[0])
chunk_size = len(data_items) // 200
output_files = []
for i in range(0, len(data_items), chunk_size):
    chunk = data_items[i:i+chunk_size]
    output_path = output_dir / f"sorted_shard_{i//chunk_size:05d}.jsonl"
    with open(output_path, 'w') as f_out:
        for _, line in chunk:
            f_out.write(line)
    output_files.append(output_path)

print(f"Finished writing sorted data into {len(output_files)} files.")