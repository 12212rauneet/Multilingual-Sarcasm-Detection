import json
from pathlib import Path

dataset_path = Path(__file__).resolve().parent.parent / 'dataset'/"sarcasm_Headlines_Dataset.json"

with open(dataset_path, 'r', encoding='utf-8') as files:
    data = [json.loads(line) for line in files]
print("Total Headlines:", {len(data)})
print("\nFirst 3 Headlines:")
for item in data[:3]:
    print(item)
sarcasm_count = sum(
    1 for item in data if item ['is_sarcastic'] ==1
)
not_sarcasm_count = sum(
    1 for item in data if item ['is_sarcastic'] ==0
)
print("\nSarcastic Headlines:", sarcasm_count)
print("Non-Sarcastic Headlines:", not_sarcasm_count)
print("Total Checked:",sarcasm_count + not_sarcasm_count)


headline= [item["headline"] for item in data]
labels= [item["is_sarcastic"] for item in data]

empty_count = sum(1 for item in headline if not item.strip())
duplicate_count = len(headline) - len(set(headline))
invalid_labels = sum(1 for label in labels if label not in (0,1))

print("Data Quality Check:")
print("Empty Headlines:", empty_count)
print("Duplicate Headlines:", duplicate_count)
print("Invalid Labels:", invalid_labels)


