import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import glob

def load_and_concatenate_parquet_files(file_patterns, relative_to=None):
    # Initialize an empty list to hold dataframes
    dataframes = []
    
    # Iterate through all file patterns
    for pattern in file_patterns:
        # Expand the glob pattern
        parquet_files = glob.glob(pattern)
        
        for file in tqdm(parquet_files):
            # Load the parquet file
            try: 
                df = pd.read_parquet(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            
            # Extract shard id from the file name
            shard_id = Path(file).stem
            
            # Add the shard id as a new column
            df['img2dataset_shard_id'] = shard_id
            df['tar_filepath'] = str(Path(file).with_suffix(".tar"))
            if args.relative_to:
                df['relative_tar_filepath'] = str(Path(file).with_suffix(".tar").relative_to(args.relative_to))
            
            # Append the dataframe to the list
            dataframes.append(df)
    
    # Concatenate all dataframes
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate Parquet files from multiple paths or glob patterns")
    parser.add_argument("file_patterns", nargs='+', type=str, help="Paths or glob patterns for Parquet files")
    parser.add_argument("-o", "--output", type=str, default="concatenated_mmc4.parquet", help="Output file name")
    parser.add_argument("--relative-to", type=str, help="Base path to make tar_filepath relative to (optional)")
    args = parser.parse_args()

    concatenated_df = load_and_concatenate_parquet_files(args.file_patterns, args.relative_to)
    output_file = Path(args.output)
    concatenated_df.to_parquet(output_file, index=False)
    print(f"Concatenated data saved to {output_file}")
    print(f"Total rows: {len(concatenated_df)}")
