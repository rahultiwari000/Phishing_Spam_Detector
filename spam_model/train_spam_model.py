# spam_model/train_spam_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Define file paths
DATA_PATH = os.path.join("datasets", "spam_emails.csv")
MODEL_PATH = os.path.join("spam_model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("spam_model", "spam_vectorizer.pkl")


def load_dataset():
    """Load and clean the spam dataset."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'text' not in df.columns:
        raise ValueError("Dataset must have a 'text' column with email contents.")
    
    if 'label_num' in df.columns:
        df['label'] = df['label_num']
    elif 'label' not in df.columns:
        raise ValueError("Dataset must have a 'label' or 'label_num' column.")

    print(f"✅ Loaded dataset with {len(df)} emails")
    return df[['text', 'label']]


def train_model():
    """Train and evaluate a spam email classifier."""
    df = load_dataset()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes model
    print("🚀 Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model + vectorizer
    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_model()
