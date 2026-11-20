
import re
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    LogisticRegression
)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# =====================================
# 2. TEXT PREPROCESSING
# =====================================

def preprocess(text):

    text = str(text).lower()

    tokens = re.findall(r'\b[a-zA-Z]+\b', text)

    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(tokens)


# =====================================
# 3. LOAD DATASET
# =====================================

print("Loading dataset...")

true_df = pd.read_csv(DATASET_DIR / "True.csv")
fake_df = pd.read_csv(DATASET_DIR / "Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

data = pd.concat(
    [true_df, fake_df],
    ignore_index=True
)

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["combined_text"] = (
    data["title"] + " " + data["text"]
)


# =====================================
# 4. PREPROCESS DATA
# =====================================

print("Preprocessing dataset...")

data["processed_text"] = (
    data["combined_text"].apply(preprocess)
)

X = data["processed_text"]
y = data["label"]


# =====================================
# 5. TRAIN TEST SPLIT
# =====================================

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================
# 6. TF-IDF VECTORIZATION
# =====================================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =====================================
# 7. INITIALIZE ML MODELS
# =====================================

models = {

    "Naive Bayes": MultinomialNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Passive Aggressive": PassiveAggressiveClassifier(
        random_state=42,
        max_iter=1000
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

}


# =====================================
# 8. TRAIN AND EVALUATE MODELS
# =====================================

metrics = {}

print("\nTraining models...\n")

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


# =====================================
# 9. SAVE TRAINED MODELS
# =====================================

print("\nSaving trained models...")

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl",
    compress=3
)

joblib.dump(
    models,
    MODEL_DIR / "trained_models.pkl",
    compress=3
)

joblib.dump(
    metrics,
    MODEL_DIR / "model_metrics.pkl",
    compress=3
)


# =====================================
# 10. DISPLAY RESULTS
# =====================================

print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")

for name, result in metrics.items():

    print(f"\n{name}")

    print("Accuracy:", result["accuracy"], "%")
    print("Precision:", result["precision"], "%")
    print("Recall:", result["recall"], "%")
    print("F1 Score:", result["f1"], "%")


print("\nSaved files inside:", MODEL_DIR)

print("\nAll three model files created successfully!")
