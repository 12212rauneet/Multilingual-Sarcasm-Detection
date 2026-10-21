# SarcasmAI — Multilingual Sarcasm Detection
## 🚀 Live Demo

[Click here to use SarcasmAI](https://multilingual-sarcasm-detection.onrender.com/)

A machine learning web application that detects sarcasm in English and Hinglish text using TF-IDF feature extraction and Logistic Regression.

## Features

- Detects sarcastic and non-sarcastic text.
- Supports English and Hinglish input.
- Combines word-level and character-level TF-IDF features.
- Provides an interactive web interface built with Django.
- Includes cleaned datasets and model evaluation results.

## Tech Stack

- **Language:** Python
- **Web Framework:** Django
- **Machine Learning:** Scikit-learn
- **Text Features:** TF-IDF, word and character n-grams
- **Model:** Logistic Regression
- **Data Processing:** Pandas
- **Model Persistence:** Joblib

## Model Performance

| Dataset | Accuracy |
|---|---:|
| English | 86.58% |
| Hinglish | 64.41% |

These are exploratory results from the current evaluation splits, which were also used during model experimentation. They should not be interpreted as an unbiased final benchmark.

## Project Structure

```text
Multilingual-Sarcasm-Detection/
├── config/
├── detector/
│   ├── templates/
│   │   └── detector/
│   │       └── home.html
│   ├── views.py
│   ├── train_model.py
│   ├── sarcasm_model.pkl
│   └── tfidf_vectorizer.pkl
├── dataset/
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Multilingual-Sarcasm-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Django server

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.

## Train the Model

To train the hybrid TF-IDF model and save its artifacts, run:

```bash
python detector/train_model.py
```

## Limitations

- Sarcasm depends on context, tone, and cultural references.
- Performance differs between English and Hinglish.
- The model may misclassify slang, regional language variations, and emoji-heavy text.
- Predictions are not guaranteed to reflect the author's intended meaning.

## Future Improvements

- Improve Hinglish and regional-language detection.
- Evaluate on a separate, untouched test dataset.
- Add confidence scores with appropriate calibration.
- Expand the dataset and analyze common prediction errors.

## Author

Created as a machine learning and web development project using Python and Django.
