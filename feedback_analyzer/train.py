import os
import re
import string
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'feedback_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'tfidf_vectorizer.pkl')


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


def train_model():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} feedback samples")

    print("\nClass distribution:")
    print(df['sentiment'].value_counts())

    df['feedback_clean'] = df['feedback'].apply(preprocess_text)

    X = df['feedback_clean']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight='balanced',
            random_state=42
        ))
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    sample_texts = [
        "This is absolutely wonderful and I love it!",
        "This is terrible and a waste of money.",
        "It is okay, nothing special really."
    ]
    print("\nQuick predictions on sample texts:")
    for text in sample_texts:
        pred = pipeline.predict([preprocess_text(text)])[0]
        proba = pipeline.predict_proba([preprocess_text(text)])[0]
        classes = pipeline.classes_
        print(f"  '{text}' -> {pred} (confidence: {max(proba):.2%})")

    return pipeline


if __name__ == '__main__':
    train_model()
