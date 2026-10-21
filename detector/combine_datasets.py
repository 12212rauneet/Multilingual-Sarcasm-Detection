
import pandas as pd
from pathlib import Path

# Project folder ka path
project_dir = Path(__file__).resolve().parent.parent

# Dono datasets ke paths
english_path = project_dir / "dataset" / "cleared_sarcasm_Dataset.csv"
hinglish_path = project_dir / "dataset" / "cleaned_hinglish_dataset.csv"

# Combined dataset ka output path
output_path = project_dir / "dataset" / "combined_sarcasm_dataset.csv"

# Datasets load 
english_df = pd.read_csv(english_path)
hinglish_df = pd.read_csv(hinglish_path)

# Dono datasets ke columns same hone chahiye
english_df = english_df[["headline", "is_sarcastic"]]
hinglish_df = hinglish_df[["headline", "is_sarcastic"]]

# Datasets combine karo
combined_df = pd.concat(
    [english_df, hinglish_df],
    ignore_index=True
)

# Empty text aur duplicate comments remove
combined_df["headline"] = combined_df["headline"].astype(str).str.strip()
combined_df = combined_df[combined_df["headline"] != ""]
combined_df = combined_df.drop_duplicates(subset=["headline"])

# Combined dataset save karo
combined_df.to_csv(output_path, index=False, encoding="utf-8")

print("Combined dataset saved:", output_path)
print("Total comments:", len(combined_df))
print("\nLabel distribution:")
print(combined_df["is_sarcastic"].value_counts().sort_index())
print("\nDataset preview:")
print(combined_df.head())
