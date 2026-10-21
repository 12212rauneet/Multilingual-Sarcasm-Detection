import pandas as pd
from pathlib import Path

project_dir=Path(__file__).resolve().parent.parent
dataset_path=project_dir/'dataset'/'human_labelled_features.csv'


df = pd.read_csv(dataset_path)


print("Human label counts:")
print(df["human_labelled"].value_counts(dropna=False))

print("\nUnique human labels:")
print(df["human_labelled"].unique())


