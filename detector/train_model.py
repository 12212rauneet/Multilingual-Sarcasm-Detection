import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


####projet path
project_dir=Path(__file__).resolve().parent.parent
english_path = project_dir / 'dataset' / 'cleared_sarcasm_Dataset.csv'
hinglish_path = project_dir / 'dataset' / 'cleaned_hinglish_dataset.csv'

english_df = pd.read_csv(english_path)
hinglish_df = pd.read_csv(hinglish_path)

english_train, english_test = train_test_split(
    english_df,
    test_size=0.2,
    random_state=42,
    stratify=english_df['is_sarcastic']
)

hinglish_train, hinglish_test = train_test_split(
    hinglish_df,
    test_size=0.2,
    random_state=42,
    stratify=hinglish_df['is_sarcastic']
)

train_df = pd.concat(
    [english_train, hinglish_train],
    ignore_index=True
)

x_train = train_df['headline'].astype(str)
y_train = train_df['is_sarcastic']

print("English training examples:", len(english_train))
print("Hinglish training examples:", len(hinglish_train))
print("Total training examples:", len(train_df))
print("English test examples:", len(english_test))
print("Hinglish test examples:", len(hinglish_test))


print("Total training examples:", len(x_train))
print("English test examples:", len(english_test))
print("Hinglish test examples:", len(hinglish_test))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion
import joblib

#conver headlines into numerical features


# Combine word-level and character-level TF-IDF features
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

x_train_tfidf = vectorizer.fit_transform(x_train)


#train the machine learning model

model = LogisticRegression(
    C=1.0,
    max_iter=1000
)


model.fit(x_train_tfidf, y_train)

print("\nModel Training Completed!")
print("Training examples:", len(x_train))
print("Model trained successfully!")





# Evaluate on unseen English test data
english_x_test = english_test['headline'].astype(str)
english_y_test = english_test['is_sarcastic']

english_x_test_tfidf = vectorizer.transform(english_x_test)
english_predictions = model.predict(english_x_test_tfidf)

print("\n===== English Test Results =====")
print("Accuracy:", round(
    accuracy_score(english_y_test, english_predictions), 4
))
print(classification_report(
    english_y_test, english_predictions, zero_division=0
))


# Evaluate on unseen Hinglish test data
hinglish_x_test = hinglish_test['headline'].astype(str)
hinglish_y_test = hinglish_test['is_sarcastic']

hinglish_x_test_tfidf = vectorizer.transform(hinglish_x_test)
hinglish_predictions = model.predict(hinglish_x_test_tfidf)

print("\n===== Hinglish Test Results =====")
print("Accuracy:", round(
    accuracy_score(hinglish_y_test, hinglish_predictions), 4
))
print(classification_report(
    hinglish_y_test, hinglish_predictions, zero_division=0
))



import joblib

model_path = project_dir/'detector'/'sarcasm_model.pkl'
vectorizer_path = project_dir/'detector'/'tfidf_vectorizer.pkl'

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nModel saved sucessfully:")
print("Model path:", model_path)
print("Vectorizer path:", vectorizer_path)

