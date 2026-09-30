from flask import Flask, render_template, request, jsonify
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def preprocess(text):
    """
    Clean and preprocess news text without using NLTK.
    This is Vercel-friendly because it does not download
    any external NLP resources at runtime.
    """

    text = str(text).lower()

    # Extract alphabetic words
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)

    # Remove English stopwords
    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(tokens)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading dataset...")

true_df = pd.read_csv("Dataset/True.csv")
fake_df = pd.read_csv("Dataset/Fake.csv")

# Labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine datasets
data = pd.concat(
    [true_df, fake_df],
    ignore_index=True
)

# Handle missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine headline + article
data["combined_text"] = (
    data["title"] + " " + data["text"]
)


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

print("Preprocessing dataset...")

data["processed_text"] = data["combined_text"].apply(
    preprocess
)


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X = data["processed_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# MODELS
# --------------------------------------------------

models = {

    "Naive Bayes":
        MultinomialNB(),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Passive Aggressive":
        PassiveAggressiveClassifier(
            random_state=42,
            max_iter=1000
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

print("Training models...")

metrics = {}

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    metrics[name] = {

        "accuracy": round(
            accuracy_score(
                y_test,
                predictions
            ) * 100,
            2
        ),

        "precision": round(
            precision_score(
                y_test,
                predictions
            ) * 100,
            2
        ),

        "recall": round(
            recall_score(
                y_test,
                predictions
            ) * 100,
            2
        ),

        "f1": round(
            f1_score(
                y_test,
                predictions
            ) * 100,
            2
        )
    }


# --------------------------------------------------
# DEFAULT MODEL
# --------------------------------------------------

default_model = models["Naive Bayes"]


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_news(title, article, model_name):

    # Combine headline + article
    combined = f"{title} {article}"

    # Preprocess
    processed = preprocess(combined)

    # Convert to TF-IDF
    vector = vectorizer.transform(
        [processed]
    )

    # Select model
    model = models.get(
        model_name,
        default_model
    )

    # Make prediction
    prediction = int(
        model.predict(vector)[0]
    )

    # REAL = 1
    # FAKE = 0

    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    # Confidence
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            vector
        )[0]

        confidence = max(
            probabilities
        ) * 100

    else:

        # Passive Aggressive does not provide
        # predict_proba by default.
        confidence = 0

    return result, round(
        confidence,
        2
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    total_articles = len(data)

    real_articles = int(
        (data["label"] == 1).sum()
    )

    fake_articles = int(
        (data["label"] == 0).sum()
    )

    return render_template(
        "index.html",

        total_articles=total_articles,

        real_articles=real_articles,

        fake_articles=fake_articles,

        metrics=metrics
    )


# --------------------------------------------------
# API — PREDICT
# --------------------------------------------------

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    request_data = request.get_json()

    if not request_data:
        return jsonify({
            "error": "Invalid request."
        }), 400

    title = request_data.get(
        "title",
        ""
    )

    article = request_data.get(
        "article",
        ""
    )

    model_name = request_data.get(
        "model",
        "Naive Bayes"
    )

    # Check input
    if not title and not article:

        return jsonify({
            "error":
            "Please enter a news headline or article."
        }), 400

    # Prediction
    result, confidence = predict_news(
        title,
        article,
        model_name
    )

    # Return JSON response
    return jsonify({

        "result": result,

        "confidence": confidence,

        "model": model_name
    })


# --------------------------------------------------
# API — METRICS
# --------------------------------------------------

@app.route("/api/metrics")
def api_metrics():

    return jsonify(metrics)


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "\n==================================="
    )

    print(
        " TruthLens Fake News Detector"
    )

    print(
        "==================================="
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "===================================\n"
    )

    app.run(
        debug=False
    )