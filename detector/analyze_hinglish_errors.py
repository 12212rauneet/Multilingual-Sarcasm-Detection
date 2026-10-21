import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Project paths
project_dir = Path(__file__).resolve().parent.parent
detector_dir = project_dir / "detector"

# 2. Load Hinglish dataset
dataset_path = project_dir / "dataset" / "cleaned_hinglish_dataset.csv"
df = pd.read_csv(dataset_path)

# 3. Recreate the same Hinglish test split
_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["is_sarcastic"]
)

# 4. Load the saved word-level model
model = joblib.load(detector_dir / "sarcasm_model.pkl")
vectorizer = joblib.load(detector_dir / "tfidf_vectorizer.pkl")

# 5. Predict the test examples
x_test = test_df["headline"].astype(str)
y_test = test_df["is_sarcastic"]

x_test_features = vectorizer.transform(x_test)
predictions = model.predict(x_test_features)

# 6. Collect incorrect predictions
results = test_df.copy()
results["predicted_label"] = predictions

errors = results[
    results["is_sarcastic"] != results["predicted_label"]
].copy()

# 7. Show error summary
print("Total Hinglish test examples:", len(test_df))
print("Incorrect predictions:", len(errors))
print("Error rate:", round(len(errors) / len(test_df), 4))
print("\n===== Error Type Counts =====")

false_negatives = errors[
    (errors["is_sarcastic"] == 1) &
    (errors["predicted_label"] == 0)
]

false_positives = errors[
    (errors["is_sarcastic"] == 0) &
    (errors["predicted_label"] == 1)
]

print("False negatives:", len(false_negatives))
print("False positives:", len(false_positives))






print("\n===== Sample Incorrect Predictions =====")
print(errors[["headline", "is_sarcastic", "predicted_label"]].head(30).to_string(index=False))

# 8. Save errors for further analysis
output_path = project_dir / "dataset" / "hinglish_prediction_errors.csv"
errors.to_csv(output_path, index=False)

print("\nError analysis saved to:", output_path)