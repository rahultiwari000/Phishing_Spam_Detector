import streamlit as st
import pandas as pd
import joblib
import os
import warnings
from phishing_model.url_features import extract_features

warnings.filterwarnings("ignore")

# ----------------------------------
# 🔧 Load Models
# ----------------------------------
PHISH_MODEL_PATH = os.path.join("phishing_model", "phishing_model.pkl")
PHISH_SCALER_PATH = os.path.join("phishing_model", "phishing_scaler.pkl")
PHISH_REAL_MODEL_PATH = os.path.join("phishing_model", "phishing_real_model.pkl")
PHISH_REAL_SCALER_PATH = os.path.join("phishing_model", "phishing_real_scaler.pkl")
SPAM_MODEL_PATH = os.path.join("spam_model", "spam_model.pkl")
SPAM_VECTORIZER_PATH = os.path.join("spam_model", "spam_vectorizer.pkl")

phish_model = joblib.load(PHISH_MODEL_PATH)
phish_scaler = joblib.load(PHISH_SCALER_PATH)
phish_real_model = joblib.load(PHISH_REAL_MODEL_PATH)
phish_real_scaler = joblib.load(PHISH_REAL_SCALER_PATH)
spam_model = joblib.load(SPAM_MODEL_PATH)
spam_vectorizer = joblib.load(SPAM_VECTORIZER_PATH)

# ----------------------------------
# 🧠 Initialize Page State
# ----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ----------------------------------
# 🧩 Detection Functions
# ----------------------------------
def check_url(url: str):
    try:
        features = extract_features(url)
        X = pd.DataFrame([features])
        X_scaled = phish_scaler.transform(X)
        pred = phish_model.predict(X_scaled)[0]
        confidence = phish_model.predict_proba(X_scaled)[0][pred] * 100
        if pred == 1:
            return f"🚨 Suspicious Website Detected", f"This URL appears to be malicious. Confidence: {confidence:.1f}%"
        else:
            return f"✅ Safe Website", f"This URL seems legitimate and safe to visit. Confidence: {confidence:.1f}%"
    except Exception as e:
        return "⚠️ Error", f"Unable to analyze URL: {e}"

def check_email(email_text: str):
    try:
        X = spam_vectorizer.transform([email_text])
        pred = spam_model.predict(X)[0]
        if pred == 1:
            return "📛 Spam Email Detected", "This email appears to be spam or potentially malicious."
        else:
            return "✉️ Legitimate Email", "This email seems safe and not spam."
    except Exception as e:
        return "⚠️ Error", f"Unable to analyze email: {e}"

def check_real_features(values_raw):
    try:
        values = [float(v) for v in values_raw.split(",") if v.strip()]
        expected = phish_real_model.n_features_in_
        if len(values) > expected:
            values = values[-expected:]
        X = pd.DataFrame([values])
        X_scaled = phish_real_scaler.transform(X)
        pred = phish_real_model.predict(X_scaled)[0]
        if pred == 1:
            return "🚨 Phishing Row Detected", "This dataset entry seems to represent a phishing website."
        else:
            return "✅ Legitimate Row", "This dataset entry is marked safe."
    except Exception as e:
        return "⚠️ Error", f"Error analyzing dataset row: {e}"

# ----------------------------------
# 🖥️ PAGE 1 - HOME (Dashboard)
# ----------------------------------
if st.session_state.page == "home":
    st.markdown("""
        <h2 style='color:#00c4ff; margin-bottom:0;'>🛡️ Detect & Secure</h2>
        <h1 style='font-weight:700; margin-top:5px;'>Hybrid Phishing & Spam Detector</h1>
        <p style='font-size:15px; opacity:0.8; margin-bottom:25px;'>
            Analyze URLs, emails, and dataset rows with a unified, security-first workflow.
        </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔗 Check URL"):
            st.session_state.page = "url"
            st.rerun()
    with col2:
        if st.button("📧 Check Email"):
            st.session_state.page = "email"
            st.rerun()
    with col3:
        if st.button("📊 Check Dataset Row"):
            st.session_state.page = "dataset"
            st.rerun()

    st.markdown("""
        <div style='text-align:justify; font-size:15px; background:rgba(255,255,255,0.05);
                    padding:18px; border-radius:10px; margin-top:35px;'>
        🌐 This application uses <b>Machine Learning</b> and <b>heuristic algorithms</b> 
        to detect phishing websites and spam emails.  
        You can scan website URLs, analyze email text, or evaluate dataset rows 
        to understand phishing patterns and enhance security awareness.
        </div>
        <div style='text-align:right; margin-top:40px; color:rgba(255,255,255,0.6); font-size:13px;'>
            🔧 Build by <b>Rahul Tiwari</b>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------
# 🌐 PAGE 2 - URL ANALYZER
# ----------------------------------
elif st.session_state.page == "url":
    st.title("🌍 Website URL Checker")
    url = st.text_input("Enter the website URL:")
    if st.button("Analyze URL"):
        heading, result = check_url(url)
        st.success(heading)
        st.info(result)
    if st.button("⬅️ Back"):
        st.session_state.page = "home"
        st.rerun()

# ----------------------------------
# 📧 PAGE 3 - EMAIL ANALYZER
# ----------------------------------
elif st.session_state.page == "email":
    st.title("📧 Email Spam Detector")
    email_text = st.text_area("Paste the email content below:")
    if st.button("Analyze Email"):
        heading, result = check_email(email_text)
        st.success(heading)
        st.info(result)
    if st.button("⬅️ Back"):
        st.session_state.page = "home"
        st.rerun()

# ----------------------------------
# 📊 PAGE 4 - DATASET ANALYZER
# ----------------------------------
elif st.session_state.page == "dataset":
    st.title("📊 Phishing Dataset Row Checker")
    row_data = st.text_area("Paste one row of numeric dataset values:")
    if st.button("Analyze Row"):
        heading, result = check_real_features(row_data)
        st.success(heading)
        st.info(result)
    if st.button("⬅️ Back"):
        st.session_state.page = "home"
        st.rerun()
