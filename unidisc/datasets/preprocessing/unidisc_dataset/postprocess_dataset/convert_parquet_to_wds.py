import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import webdataset as wds
from pathlib import Path
from PIL import Image
from io import BytesIO
import os

def process_chunk(chunk, shard_id, base_dir, output_tar_path):
    base_dir = Path(base_dir)
    output_tar_path = output_tar_path.replace("output_dataset_", f"output_dataset_{shard_id}_")
    print(f"Processing shard {shard_id} with {len(chunk)} samples, {output_tar_path}")
    with wds.ShardWriter(output_tar_path, maxsize=500*1024*1024) as sink:
        for _, row in chunk.iterrows():
            # Construct the image path
            # image_path = base_dir / row['__key__']
            image_path = row['image_path']
            if not Path(image_path).exists(): assert False
            try:
                # Load the image
                with Image.open(image_path) as img:
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_data = img_byte_arr.getvalue()
                # Create a unique key for each sample
                key = Path(image_path).stem

                # Prepare the sample dictionary
                sample = {
                    '__key__': key,
                    'jpg': img_data,
                    'txt': row['caption'],
                    'meta.json': row.drop(['__key__', 'caption']).to_dict()
                }
                # Write the sample to the shard
                sink.write(sample)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

            
def main(parquet_file, base_dir, output_dir, num_workers=8):
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)
    print(f"Dataframe loaded with {len(df)} rows.")
    print(f"Columns: {df.columns.tolist()}")

    # Ensure the output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tar_path = str(output_dir) + "/output_dataset_%06d.tar"

    # Split the DataFrame into chunks for each worker
    df_split = np.array_split(df, num_workers)

    # Use a ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for shard_id, chunk in enumerate(df_split):
            futures.append(executor.submit(process_chunk, chunk, shard_id, base_dir, output_tar_path))

        # Collect results and handle exceptions
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet_file', type=str, required=True, help="Path to the Parquet file.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory where images are located.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output WebDataset tar files.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()
    main(args.parquet_file, args.base_dir, args.output_dir, args.num_workers)
