import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion

# 1. Project paths
project_dir = Path(__file__).resolve().parent.parent

english_path = project_dir / "dataset" / "cleared_sarcasm_Dataset.csv"
hinglish_path = project_dir / "dataset" / "cleaned_hinglish_dataset.csv"

# 2. Load datasets
english_df = pd.read_csv(english_path)
hinglish_df = pd.read_csv(hinglish_path)

# 3. Split each language separately
english_train, english_test = train_test_split(
    english_df,
    test_size=0.2,
    random_state=42,
    stratify=english_df["is_sarcastic"]
)

hinglish_train, hinglish_test = train_test_split(
    hinglish_df,
    test_size=0.2,
    random_state=42,
    stratify=hinglish_df["is_sarcastic"]
)

# 4. Combine training data only
train_df = pd.concat(
    [english_train, hinglish_train],
    ignore_index=True
)

x_train = train_df["headline"].astype(str)
y_train = train_df["is_sarcastic"]

# 5. Combine word-level and character-level features
vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            max_features=20000,
            ngram_range=(3, 5),
            min_df=2
        )
    )
])

x_train_features = vectorizer.fit_transform(x_train)

# 6. Train the model
model = LogisticRegression(
    C=1.0,
    max_iter=1000
)

model.fit(x_train_features, y_train)

print("Hybrid model training completed!")
print("Training examples:", len(x_train))

# 7. Evaluate English test data
english_x = english_test["headline"].astype(str)
english_y = english_test["is_sarcastic"]

english_features = vectorizer.transform(english_x)
english_pred = model.predict(english_features)

print("\n===== English Hybrid Model =====")
print("Accuracy:", round(accuracy_score(english_y, english_pred), 4))
print(classification_report(
    english_y, english_pred, zero_division=0
))

# 8. Evaluate Hinglish test data
hinglish_x = hinglish_test["headline"].astype(str)
hinglish_y = hinglish_test["is_sarcastic"]

hinglish_features = vectorizer.transform(hinglish_x)
hinglish_pred = model.predict(hinglish_features)

print("\n===== Hinglish Hybrid Model =====")
print("Accuracy:", round(accuracy_score(hinglish_y, hinglish_pred), 4))
print(classification_report(
    hinglish_y, hinglish_pred, zero_division=0
))

# 9. Analyze Hinglish prediction errors
hinglish_results = hinglish_test.copy()
hinglish_results["predicted_label"] = hinglish_pred

errors = hinglish_results[
    hinglish_results["is_sarcastic"] !=
    hinglish_results["predicted_label"]
].copy()

false_negatives = errors[
    (errors["is_sarcastic"] == 1) &
    (errors["predicted_label"] == 0)
]

false_positives = errors[
    (errors["is_sarcastic"] == 0) &
    (errors["predicted_label"] == 1)
]

print("\n===== Hybrid Model Error Analysis =====")
print("Total Hinglish test examples:", len(hinglish_test))
print("Incorrect predictions:", len(errors))
print("Error rate:", round(len(errors) / len(hinglish_test), 4))
print("False negatives:", len(false_negatives))
print("False positives:", len(false_positives))

# Save incorrect predictions
output_path = project_dir / "dataset" / "hinglish_hybrid_errors.csv"
errors.to_csv(output_path, index=False)

print("Errors saved to:", output_path)