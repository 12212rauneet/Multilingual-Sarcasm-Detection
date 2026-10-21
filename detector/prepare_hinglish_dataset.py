
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

input_path = project_dir / "dataset" / "human_labelled_features.csv"
output_path = project_dir / "dataset" / "cleaned_hinglish_dataset.csv"

df = pd.read_csv(input_path)

# Keep only the human-labelled columns
df = df[["comment", "human_labelled"]].copy()

# Remove missing values
df = df.dropna(subset=["comment", "human_labelled"])

# Rename columns
df = df.rename(columns={
    "comment": "headline",
    "human_labelled": "label"
})

# Clean text and labels
df["headline"] = df["headline"].astype(str).str.strip()
df["label"] = (
    df["label"].astype(str).str.strip().str.lower()
)

# Standardize both non-sarcastic label formats
df["label"] = df["label"].replace({
    "non_sarcastic": "non-sarcastic"
})

# Keep valid labels and non-empty comments
df = df[
    (df["headline"] != "") &
    (df["label"].isin(["sarcastic", "non-sarcastic"]))
].copy()

# Convert labels to numeric values
df["is_sarcastic"] = df["label"].map({
    "non-sarcastic": 0,
    "sarcastic": 1
})

# Remove duplicate comments
df = df.drop_duplicates(subset=["headline"])

# Save cleaned dataset
df[["headline", "is_sarcastic"]].to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print("Cleaned dataset saved:", output_path)
print("Total comments:", len(df))
print("\nLabel distribution:")
print(df["is_sarcastic"].value_counts().sort_index())
