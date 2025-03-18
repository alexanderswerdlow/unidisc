from datasets import load_dataset
import json
import re

dataset = load_dataset("parquet", data_files={'test': 'https://huggingface.co/datasets/THUDM/ImageRewardDB/resolve/main/metadata-test.parquet', 'train': 'https://huggingface.co/datasets/THUDM/ImageRewardDB/resolve/main/metadata-train.parquet', 'validation': 'https://huggingface.co/datasets/THUDM/ImageRewardDB/resolve/main/metadata-validation.parquet'}, split='train+validation+test')

unique_captions = set()

for item in dataset:
    unique_captions.add(item['prompt'])

unique_captions = list(unique_captions)

def correct_data(text):
    # Replace 4, 3, and 2 digit numbers with spaces in between
    text = re.sub(r'(\d) (\d) (\d) (\d)', r'\1\2\3\4', text)
    text = re.sub(r'(\d) (\d) (\d)', r'\1\2\3', text)
    text = re.sub(r'(\d) (\d)', r'\1\2', text)
    # Remove spaces between a single digit and the letter k
    text = re.sub(r'(\d) k', r'\1k', text)
    # Replace 2 digits followed by 2 characters and a space
    text = re.sub(r'(\d) (\d)(\w\w) ', r'\1\2\3 ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    # Replace "( (" with "((" and "[ [" with "[["
    text = re.sub(r'\( \(', r'((', text)
    text = re.sub(r'\[ \[', r'[[', text)
    # Replace ") )" with "))" and "] ]" with "]]"
    text = re.sub(r'\) \)', r'))', text)
    text = re.sub(r'\] \]', r']]', text)

    text = re.sub(r'\( \(', r'(', text)
    text = re.sub(r'\[ \[', r'[', text)
    # Replace ") )" with ")" and "] ]" with "]"
    text = re.sub(r'\) \)', r')', text)
    text = re.sub(r'\] \]', r']', text)

    text = re.sub(r'(\d) (\w) ', r'\1\2 ', text)

    return text


    
corrected_data = [correct_data(item) for item in unique_captions]

with open('image_reward.json', 'w') as f:
    json.dump(list(corrected_data), f)

print(f"Total unique captions: {len(list(unique_captions))}")