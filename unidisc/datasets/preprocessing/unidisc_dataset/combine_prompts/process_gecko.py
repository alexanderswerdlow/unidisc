import pandas as pd
import json

# https://github.com/google-deepmind/gecko_benchmark_t2i/blob/main/prompts.csv
df = pd.read_csv('prompts.csv')

def clean_prompt(prompt):
    ascii_prompt = ''.join(char for char in prompt if ord(char) < 128)
    cleaned_prompt = ascii_prompt.replace('\n', ' ').strip()
    return cleaned_prompt


df['prompt'] = df['prompt'].apply(clean_prompt)
df = df.sample(frac=1).reset_index(drop=True)
sampled_list = df['prompt'].tolist()

with open('sampled_prompts.json', 'w') as f:
    json.dump(sampled_list, f)