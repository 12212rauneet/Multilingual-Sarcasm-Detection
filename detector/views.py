import joblib
from pathlib import Path
from django.shortcuts import render

# Load the trained model and vectorizer
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "sarcasm_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")


def home(request):
    result = None
    user_text = ""

    if request.method == "POST":
        user_text = request.POST.get("text", "").strip()

        if user_text:
            text_vector = vectorizer.transform([user_text])
            prediction = model.predict(text_vector)[0]

            if prediction == 1:
                result = "Sarcastic 😏"
            else:
                result = "Not Sarcastic 🙂"

    return render(
        request,
        "detector/home.html",
        {
            "result": result,
            "user_text": user_text,
        }
    )