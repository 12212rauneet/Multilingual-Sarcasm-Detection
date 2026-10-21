import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Project paths
project_dir = Path(__file__).resolve().parent.parent
detector_dir = project_dir / "detector"

# Load trained model and vectorizer
model = joblib.load(detector_dir / "sarcasm_model.pkl")
vectorizer = joblib.load(detector_dir / "tfidf_vectorizer.pkl")

# Dataset paths
english_path = project_dir / "dataset" / "cleared_sarcasm_Dataset.csv"
hinglish_path = project_dir / "dataset" / "cleaned_hinglish_dataset.csv"

# Load datasets
english_df = pd.read_csv(english_path)
hinglish_df = pd.read_csv(hinglish_path)

# Evaluate each language
datasets = {
    "English": english_df,
    "Hinglish": hinglish_df
}

for language, df in datasets.items():
    X = df["headline"].astype(str)
    y = df["is_sarcastic"]

    X_vectorized = vectorizer.transform(X)
    predictions = model.predict(X_vectorized)

    print(f"\n{'=' * 15} {language} Evaluation {'=' * 15}")
    print("Total examples:", len(df))
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))