
from pathlib import Path
import re
import joblib

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load saved files (no training at runtime)
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
models = joblib.load(MODEL_DIR / "trained_models.pkl")
metrics = joblib.load(MODEL_DIR / "model_metrics.pkl")
stats = joblib.load(MODEL_DIR / "dataset_stats.pkl")

print("TruthLens models loaded successfully!")


def preprocess(text):
    text = str(text).lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
    ]
    return " ".join(tokens)


def predict_news(title, article, model_name):
    combined = f"{title} {article}"
    processed = preprocess(combined)
    vector = vectorizer.transform([processed])

    model = models.get(model_name, models["Naive Bayes"])
    prediction = int(model.predict(vector)[0])

    result = "REAL" if prediction == 1 else "FAKE"

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = 0

    return result, round(confidence, 2)


@app.route("/")
def home():
    return render_template(
        "index.html",
        total_articles=stats["total_articles"],
        real_articles=stats["real_articles"],
        fake_articles=stats["fake_articles"],
        metrics=metrics
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    request_data = request.get_json()

    if not request_data:
        return jsonify({"error": "Invalid request."}), 400

    title = request_data.get("title", "")
    article = request_data.get("article", "")
    model_name = request_data.get("model", "Naive Bayes")

    if not title and not article:
        return jsonify({
            "error": "Please enter a news headline or article."
        }), 400

    if model_name not in models:
        model_name = "Naive Bayes"

    result, confidence = predict_news(
        title, article, model_name
    )

    return jsonify({
        "result": result,
        "confidence": confidence,
        "model": model_name
    })


@app.route("/api/metrics")
def api_metrics():
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(debug=False)
