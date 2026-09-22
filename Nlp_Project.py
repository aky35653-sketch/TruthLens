

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 2. DOWNLOAD NLTK RESOURCES
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

# IMPORTANT:
# True.csv = REAL NEWS = label 1
# Fake.csv = FAKE NEWS = label 0

true_df = pd.read_csv("Dataset/True.csv")
fake_df = pd.read_csv("Dataset/Fake.csv")


# ============================================================
# 4. ADD LABELS
# ============================================================

true_df["label"] = 1
fake_df["label"] = 0


# ============================================================
# 5. COMBINE DATASETS
# ============================================================

data = pd.concat(
    [true_df, fake_df],
    ignore_index=True
)


print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 6. CHECK DATASET
# ============================================================

print("\nDataset columns:")
print(data.columns.tolist())

print("\nLabel distribution:")
print(data["label"].value_counts())

print("\nMissing values:")
print(data[["title", "text"]].isnull().sum())


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")


# ============================================================
# 8. COMBINE TITLE + ARTICLE TEXT
# ============================================================

data["content"] = (
    data["title"].astype(str)
    + " "
    + data["text"].astype(str)
)


# Convert everything to lowercase

data["content"] = data["content"].str.lower()


# ============================================================
# 9. OPTIONAL NLP TOKENIZATION
# ============================================================

print("\nTokenizing text...")

data["content_tokens"] = data["content"].apply(
    nltk.word_tokenize
)


# ============================================================
# 10. REMOVE STOPWORDS
# ============================================================

print("Removing stopwords...")

stop_words = set(stopwords.words("english"))

data["content_tokens"] = data["content_tokens"].apply(
    lambda words: [
        word
        for word in words
        if word.isalpha() and word not in stop_words
    ]
)


# Convert tokens back to text

data["processed_text"] = data["content_tokens"].apply(
    lambda words: " ".join(words)
)


# ============================================================
# 11. DEFINE X AND y
# ============================================================

X = data["processed_text"]
y = data["label"]


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 13. TF-IDF VECTORIZATION
# ============================================================

print("\nCreating TF-IDF features...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000
)


# IMPORTANT:
# Fit ONLY on training data

X_train_tfidf = tfidf_vectorizer.fit_transform(
    X_train
)


# Transform test data using the SAME vectorizer

X_test_tfidf = tfidf_vectorizer.transform(
    X_test
)


print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ============================================================
# 14. FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(model, model_name):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Train model
    model.fit(
        X_train_tfidf,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test_tfidf
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    classification_rep = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    # Print results

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nClassification Report:")
    print(classification_rep)

    return {
        "model": model,
        "predictions": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion
    }


# ============================================================
# 15. MULTINOMIAL NAIVE BAYES
# ============================================================

naive_bayes = MultinomialNB()

nb_results = evaluate_model(
    naive_bayes,
    "MULTINOMIAL NAIVE BAYES"
)


# ============================================================
# 16. DECISION TREE
# ============================================================

decision_tree = DecisionTreeClassifier(
    random_state=42
)

dt_results = evaluate_model(
    decision_tree,
    "DECISION TREE"
)


# ============================================================
# 17. PASSIVE AGGRESSIVE CLASSIFIER
# ============================================================

passive_aggressive = PassiveAggressiveClassifier(
    random_state=42
)

pa_results = evaluate_model(
    passive_aggressive,
    "PASSIVE AGGRESSIVE CLASSIFIER"
)


# ============================================================
# 18. LOGISTIC REGRESSION
# ============================================================

logistic_regression = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_results = evaluate_model(
    logistic_regression,
    "LOGISTIC REGRESSION"
)


# ============================================================
# 19. RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_results = evaluate_model(
    random_forest,
    "RANDOM FOREST"
)


# ============================================================
# 20. STORE ALL RESULTS
# ============================================================

results = {
    "Multinomial Naive Bayes": nb_results,
    "Decision Tree": dt_results,
    "Passive Aggressive": pa_results,
    "Logistic Regression": lr_results,
    "Random Forest": rf_results
}


# ============================================================
# 21. PRINT FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

for name, result in results.items():

    print(
        f"{name:30s} "
        f"Accuracy: {result['accuracy']:.4f} | "
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f} | "
        f"F1: {result['f1']:.4f}"
    )


# ============================================================
# 22. CREATE COMPARISON DATAFRAME
# ============================================================

comparison_data = []

for name, result in results.items():

    comparison_data.append({
        "Algorithm": name,
        "Accuracy": result["accuracy"],
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1 Score": result["f1"]
    })


comparison_df = pd.DataFrame(
    comparison_data
)


print("\nModel Comparison:")
print(comparison_df)


# ============================================================
# 23. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Algorithm"],
    comparison_df["Accuracy"]
)

plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Algorithm Comparison - Accuracy")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 24. PRECISION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Algorithm"],
    comparison_df["Precision"]
)

plt.xlabel("Algorithm")
plt.ylabel("Precision")
plt.title("Algorithm Comparison - Precision")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 25. RECALL GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Algorithm"],
    comparison_df["Recall"]
)

plt.xlabel("Algorithm")
plt.ylabel("Recall")
plt.title("Algorithm Comparison - Recall")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 26. F1 SCORE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Algorithm"],
    comparison_df["F1 Score"]
)

plt.xlabel("Algorithm")
plt.ylabel("F1 Score")
plt.title("Algorithm Comparison - F1 Score")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 27. CONFUSION MATRIX - NAIVE BAYES
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    nb_results["confusion"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multinomial Naive Bayes - Confusion Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 28. PREDICTION FUNCTION
# ============================================================

def predict_news(title_text, article_text=""):

    # Combine title + article

    combined_text = (
        str(title_text)
        + " "
        + str(article_text)
    )

    # Lowercase

    combined_text = combined_text.lower()

    # Tokenization

    tokens = nltk.word_tokenize(
        combined_text
    )

    # Stopword removal

    tokens = [
        word
        for word in tokens
        if word.isalpha()
        and word not in stop_words
    ]

    # Convert back to string

    processed_text = " ".join(tokens)

    # TF-IDF transformation

    tfidf_vector = tfidf_vectorizer.transform(
        [processed_text]
    )

    # Prediction using Naive Bayes

    prediction = naive_bayes.predict(
        tfidf_vector
    )[0]

    # Probability

    probabilities = naive_bayes.predict_proba(
        tfidf_vector
    )[0]

    confidence = max(probabilities) * 100

    # Result

    if prediction == 1:

        result = "REAL"

    else:

        result = "FAKE"

    return result, confidence


# ============================================================
# 29. TEST USER PREDICTIONS
# ============================================================

print("\n")
print("=" * 80)
print("TESTING NEWS PREDICTION")
print("=" * 80)


title_1 = "Donald Trump Sends Out Embarrassing New Year Message"

result_1, confidence_1 = predict_news(
    title_1
)

print("\nTitle 1:")
print(title_1)

print("Prediction:", result_1)
print(f"Confidence: {confidence_1:.2f}%")


title_2 = (
    "As U.S. budget fight looms, Republicans "
    "flip their fiscal script"
)

result_2, confidence_2 = predict_news(
    title_2
)

print("\nTitle 2:")
print(title_2)

print("Prediction:", result_2)
print(f"Confidence: {confidence_2:.2f}%")


# ============================================================
# 30. FINISHED
# ============================================================

print("\n")
print("=" * 80)
print("FAKE NEWS DETECTION PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)