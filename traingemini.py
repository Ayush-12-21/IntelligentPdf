"""
Train heading classifier from a single CSV or JSON file.

Expected columns / keys:
    text   – the line
    label  – 1 for heading, 0 for body
"""
import os, json, pickle, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = "training_data"
CSV_PATH  = os.path.join(DATA_DIR, "labels.csv")
JSON_PATH = os.path.join(DATA_DIR, "labels.json")
MODEL_OUT = "heading_model.pkl"

# -----------------------------------------------------------
def load_training_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    elif os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as fp:
            df = pd.DataFrame(json.load(fp))
    else:
        raise FileNotFoundError(
            "Neither labels.csv nor labels.json found in training_data/"
        )
    if {"text", "label"} - set(df.columns):
        raise ValueError("File must contain 'text' and 'label' fields")
    return df["text"].tolist(), df["label"].tolist()

# -----------------------------------------------------------
print("Loading labeled data …")
texts, labels = load_training_data()

# quick sanity
print(f"Samples: {len(texts)} (headings={sum(labels)}, body={len(labels)-sum(labels)})")

# simple TF-IDF → RandomForest pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        max_features=10_000,
        stop_words="english"
    )),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)

print("Training …")
pipe.fit(X_train, y_train)

print("\n=== Evaluation ===")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

# save
with open(MODEL_OUT, "wb") as fp:
    pickle.dump(pipe, fp)
print(f"Model saved → {MODEL_OUT}")
