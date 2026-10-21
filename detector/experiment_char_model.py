
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

# 5. Convert text into character-level features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=20000,
    min_df=2
)

x_train_tfidf = vectorizer.fit_transform(x_train)

# 6. Train the model
model = LogisticRegression(
    C=1.0,
    max_iter=1000
)

model.fit(x_train_tfidf, y_train)

print("Character-level model training completed!")
print("Training examples:", len(x_train))

# 7. Evaluate English test data
english_x = english_test["headline"].astype(str)
english_y = english_test["is_sarcastic"]

english_pred = model.predict(vectorizer.transform(english_x))

print("\n===== English Character Model =====")
print("Accuracy:", round(accuracy_score(english_y, english_pred), 4))
print(classification_report(english_y, english_pred, zero_division=0))

# 8. Evaluate Hinglish test data
hinglish_x = hinglish_test["headline"].astype(str)
hinglish_y = hinglish_test["is_sarcastic"]

hinglish_pred = model.predict(vectorizer.transform(hinglish_x))

print("\n===== Hinglish Character Model =====")
print("Accuracy:", round(accuracy_score(hinglish_y, hinglish_pred), 4))
print(classification_report(hinglish_y, hinglish_pred, zero_division=0))
