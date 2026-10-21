import json
import pandas as pd
from pathlib import Path

project_dir=Path(__file__).resolve().parent.parent
dataset_path=project_dir/'dataset'/'sarcasm_Headlines_Dataset.json'
output_path=project_dir/'dataset'/'cleared_sarcasm_Dataset.csv'

with open(dataset_path, 'r', encoding='utf-8') as files:
    data = [json.loads(line) for line in files]

df = pd.DataFrame(data)
df['headline'] = df['headline'].astype(str).str.strip()
df=df.drop_duplicates(subset=["headline"])

df.to_csv(output_path, index=False, encoding='utf-8')

print("Cleaned dataset saved to:", output_path)
print("Total rows after cleaning:", len(df))
print("\nLabel Distribution:")
print(df['is_sarcastic'].value_counts().sort_index())
