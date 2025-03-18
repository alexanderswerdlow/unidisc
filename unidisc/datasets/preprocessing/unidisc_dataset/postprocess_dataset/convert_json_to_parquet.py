import json
from pathlib import Path
from typing import Optional
import pandas as pd
import typer
from tqdm import tqdm
import socket

def main(directories: list[Path], output_path: Optional[Path] = None):
    data = []
    for directory in directories:
        for i, json_file in tqdm(enumerate(directory.glob('*.json'))):
            try:
                with json_file.open('r') as f:
                    metadata = json.load(f)
                    image_filename = json_file.with_suffix('.jpg').name
                    metadata['__key__'] = str((directory / image_filename).relative_to(directory.parent))
                    metadata["caption"] = metadata["augmented_prompt"]
                    metadata["subdirectory"] = str(directory.relative_to(directory.parent))
                    data.append(metadata)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(data)
    df['idx'] = df.index
    hostname = socket.gethostname()
    df['cluster'] = 'cluster_name'
    df = df[df['image_path'].notna() & (df['image_path'] != '')]
    df.to_parquet(output_path, index=False)
    print(f"Metadata has been saved to {output_path}")

if __name__ == "__main__":
    typer.run(main)