# 🛡️ Hybrid Phishing & Spam Detector  

A **Machine Learning + Heuristic-based security detection system** that identifies **phishing websites, spam emails, and unsafe dataset entries** using a hybrid approach — combining trained ML models, domain reputation analysis, and text-based heuristics.

---

## 🚀 Features  

✅ **Website URL Scanner** – Detects malicious or phishing websites using trained models on extracted URL features.  
✅ **Email Spam Classifier** – Analyzes message content to flag spam or phishing emails.  
✅ **Dataset Analyzer** – Evaluates real-world phishing datasets for research and visualization.  
✅ **Streamlit UI** – A clean, modern, and interactive dashboard for real-time security checks.  
✅ **Hybrid Detection** – Combines Machine Learning + heuristic rules + domain reputation insights.

---

## 🧠 Tech Stack  

- **Language:** Python 3  
- **Framework:** Streamlit  
- **Libraries:** scikit-learn, pandas, numpy, joblib  
- **Models Used:**  
  - Logistic Regression (Phishing URL detection)  
  - Naive Bayes / Logistic Regression (Spam email classification)

---

## 🗂️ Folder Structure  

Phishing_Spam_Detector/
│
├── datasets/
│ ├── phishing_site_urls.csv
│ ├── Phishing_Legitimate_full.csv
│ ├── spam_emails.csv
│ └── domains_reputation.csv
│
├── phishing_model/
│ ├── phishing_model.pkl
│ ├── phishing_real_model.pkl
│ ├── phishing_real_scaler.pkl
│ ├── phishing_scaler.pkl
│ ├── train_phishing_model.py
│ ├── train_phishing_real_dataset.py
│ └── url_features.py
│
├── spam_model/
│ ├── spam_model.pkl
│ ├── spam_vectorizer.pkl
│ └── train_spam_model.py
│
├── app.py # Streamlit Web App
├── requirements.txt # Dependencies
└── README.md # Project Documentation

yaml
Copy code

---

## 🌐 How It Works  

1. **URL Analysis:** Extracts structural features (e.g., length, subdomains, special characters) and predicts phishing probability.  
2. **Email Spam Detection:** Uses text vectorization and trained models to identify spam messages.  
3. **Dataset Row Check:** Evaluates pre-engineered feature datasets to validate phishing classifications.  
4. **Heuristic Filters:** Enhances model decisions with rule-based reputation checks.  

---

## 🧩 Example Outputs  

- **✅ Safe Website:**  
  “This website appears legitimate and safe.”  

- **🚨 Suspicious Website:**  
  “This URL is flagged as phishing or unsafe. Proceed with caution.”  

- **📛 Spam Email:**  
  “This message shows characteristics of spam or fraud.”  

---

## 🧰 Requirements  

All required Python libraries are listed in `requirements.txt`.  
Main dependencies include:
```text
streamlit
pandas
numpy
scikit-learn
joblib
