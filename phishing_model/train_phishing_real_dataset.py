

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump


DATA_PATH = os.path.join("datasets", "Phishing_Legitimate_full.csv")
MODEL_PATH = os.path.join("phishing_model", "phishing_real_model.pkl")
SCALER_PATH = os.path.join("phishing_model", "phishing_real_scaler.pkl")


def train_model():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded successfully with shape: {df.shape}")

    
    if 'CLASS_LABEL' not in df.columns:
        raise ValueError("❌ Could not find 'CLASS_LABEL' column in dataset.")
    
    y = df['CLASS_LABEL']
    df = df.drop(columns=['CLASS_LABEL'], errors='ignore')

   
    drop_cols = ['id', 'Index', 'Result', 'Unnamed: 0']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

   
    X = df.select_dtypes(include=[np.number])
    print(f"🔢 Selected {X.shape[1]} numeric feature columns for training.")

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    print("🚀 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
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
    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    train_model()
