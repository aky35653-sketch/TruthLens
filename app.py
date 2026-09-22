from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
# NLTK
# --------------------------------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

true_df = pd.read_csv("Dataset/True.csv")
fake_df = pd.read_csv("Dataset/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

data = pd.concat([true_df, fake_df], ignore_index=True)

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["combined_text"] = data["title"] + " " + data["text"]


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def preprocess(text):
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(tokens)


print("Preprocessing dataset...")

data["processed_text"] = data["combined_text"].apply(preprocess)


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

X_train_tfidf = vectorizer.fit_transform(X_train)
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

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    metrics[name] = {
        "accuracy": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),

        "precision": round(
            precision_score(y_test, predictions) * 100, 2
        ),

        "recall": round(
            recall_score(y_test, predictions) * 100, 2
        ),

        "f1": round(
            f1_score(y_test, predictions) * 100, 2
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

    combined = f"{title} {article}"

    processed = preprocess(combined)

    vector = vectorizer.transform([processed])

    model = models.get(
        model_name,
        default_model
    )

    prediction = int(model.predict(vector)[0])

    # REAL = 1
    # FAKE = 0

    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    # Confidence
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(vector)[0]

        confidence = max(probabilities) * 100

    else:

        confidence = 0

    return result, round(confidence, 2)


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

@app.route("/api/predict", methods=["POST"])
def api_predict():

    request_data = request.get_json()

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

    if not title and not article:

        return jsonify({
            "error": "Please enter a news headline or article."
        }), 400

    result, confidence = predict_news(
        title,
        article,
        model_name
    )

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

    print("\n===================================")
    print(" TruthLens Fake News Detector")
    print("===================================")
    print("Open: http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        debug=True
    )