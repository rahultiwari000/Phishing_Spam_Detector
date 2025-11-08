
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from url_features import extract_features


DATA_PATH = os.path.join("datasets", "phishing_site_urls.csv")
MODEL_PATH = os.path.join("phishing_model", "phishing_model.pkl")
SCALER_PATH = os.path.join("phishing_model", "phishing_scaler.pkl")


def load_dataset():
    """Loads the phishing dataset and handles different Kaggle formats."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    
    if 'label' in df.columns or 'class' in df.columns or 'phishing' in df.columns:
        label_col = [c for c in df.columns if c in ('label', 'class', 'phishing')][0]
        url_col = [c for c in df.columns if 'url' in c][0]
        df = df[[url_col, label_col]]
        df.columns = ['url', 'label']

   
    elif df.shape[1] == 1:
        df.columns = ['url']
        df['label'] = 1
    else:
        raise ValueError("Cannot detect columns. Please ensure dataset has 'url' and 'label' columns.")

    print(f"✅ Loaded dataset with {len(df)} rows")
    return df


def add_safe_samples(df, num_samples=500):
    """Adds some safe URLs (non-phishing) for balance."""
    safe_urls = [
        "https://www.google.com",
        "https://www.amazon.com",
        "https://www.microsoft.com",
        "https://www.wikipedia.org",
        "https://www.apple.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.linkedin.com",
        "https://www.nytimes.com",
        "https://www.instagram.com",
        "https://www.facebook.com",
        "https://www.reddit.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.tesla.com"
    ]
    safe_df = pd.DataFrame({"url": safe_urls * (num_samples // len(safe_urls)), "label": 0})
    full_df = pd.concat([df, safe_df], ignore_index=True)
    return full_df


def extract_all_features(df):
    """Converts all URLs into feature dictionaries."""
    print("🔍 Extracting features from URLs (this may take a minute)...")
    features = df['url'].apply(extract_features).apply(pd.Series)
    features['label'] = df['label']
    print("✅ Feature extraction complete.")
    return features


def train_model():
    df = load_dataset()
    df = add_safe_samples(df)
    features = extract_all_features(df)

    X = features.drop('label', axis=1)
    y = features['label']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

  
    print("🚀 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)


    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

  
    dump(model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"💾 Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    train_model()
