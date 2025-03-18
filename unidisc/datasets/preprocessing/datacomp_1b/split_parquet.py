from pathlib import Path
import shutil
import random

def split_parquet_files(input_folder, output_folder, max_files_per_folder=100):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    parquet_files = list(input_folder.glob('*.parquet'))
    
    random.shuffle(parquet_files)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for i in range(0, len(parquet_files), max_files_per_folder):
        subfolder_name = f"subfolder_{i // max_files_per_folder + 1}"
        subfolder_path = output_folder / subfolder_name
        
        subfolder_path.mkdir(parents=True, exist_ok=True)
        
        for file in parquet_files[i:i + max_files_per_folder]:
            shutil.move(str(file), str(subfolder_path / file.name))

# Example usage
input_folder = '/path/to/recap-datacomp-1b/data/train_data'
output_folder = '/path/to/recap-datacomp-1b/data/train_data_split'
split_parquet_files(input_folder, output_folder)