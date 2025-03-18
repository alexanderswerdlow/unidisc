from datasets import load_dataset
import json

# Load both datasets
dataset_v1 = load_dataset('yuvalkirstain/pickapic_v1_no_images', split='train+validation+test')
dataset_v2 = load_dataset('yuvalkirstain/pickapic_v2_no_images', split='train+validation+test')

# Create a set to store unique captions
unique_captions = set()

# Add captions from v1 dataset
for item in dataset_v1:
    unique_captions.add(item['caption'])

# Add captions from v2 dataset
for item in dataset_v2:
    unique_captions.add(item['caption'])

# Convert set to list
caption_list = list(unique_captions)

# Write to JSON file
with open('unique_captions.json', 'w') as f:
    json.dump(caption_list, f, indent=2)

print(f"Total unique captions: {len(caption_list)}")