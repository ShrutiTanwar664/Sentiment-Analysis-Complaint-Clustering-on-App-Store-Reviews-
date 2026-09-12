# 📱 App Review Sentiment Analysis & Complaint Clustering

An end-to-end NLP pipeline that scrapes, analyzes, and clusters **100,000+ Google Play Store reviews** to classify sentiment and automatically surface recurring user complaints across major Indian apps.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project analyzes user reviews across **6 app categories** — Shopping, Delivery, FinTech, Social, Productivity, and Media/Travel — to answer two questions:

1. **Is a review positive or negative?** → Solved using an **LSTM deep learning model**
2. **What are users actually complaining about?** → Solved using **TF-IDF + K-Means clustering** to auto-discover complaint taxonomies from unstructured text

The result is a data-driven view of *why* users leave negative reviews, broken down by app and by category.

---

## 🔁 Pipeline


Google Play Store
       │
       ▼
 1. Web Scraping        →  google-play-scraper → 100K+ reviews → CSV
       │
       ▼
 2. Preprocessing        →  Text cleaning, tokenization, sequence padding
       │
       ▼
 3. Sentiment Model      →  LSTM (TensorFlow/Keras) → Positive / Negative
       │
       ▼
 4. Complaint Clustering →  TF-IDF + K-Means → 9 Complaint Taxonomies
       │
       ▼
 5. Visualization        →  Heatmaps, distribution & category-wise charts


---

## 📊 Results

### Sentiment Classification (LSTM)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative (0) | 0.95 | 0.94 | 0.95 | 22,701 |
| Positive (1) | 0.83 | 0.84 | 0.84 | 7,390 |
| **Accuracy** | | | **0.92** | 30,091 |

**Confusion Matrix**

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 21,451 | 1,250 |
| **Actual Positive** | 1,188 | 6,202 |

### Complaint Taxonomy Clustering (TF-IDF + K-Means)

Clustered **20,000+ negative reviews** into **9 core complaint categories**, including:

- Support Escalation Bottlenecks
- Customer Support Reachability
- Return & Refund Issues
- Delivery Partner & Order Delay
- Bank Account Deduction Failures
- Payment Gateway Errors
- Login Bugs & Authentication Issues
- Streaming & Subscription Bugs
- Ride Booking & Destination Failure

**Key insights by category:**
- 💳 **FinTech** (CRED) → dominated by Customer Support Reachability & Payment Gateway Errors
- 🚚 **Delivery** (Swiggy, Zepto, Zomato) → dominated by Support Escalation Bottlenecks
- 🛍️ **Shopping** (Flipkart, Amazon, AJIO, Nykaa) → mostly Customer Support Reachability & Return/Refund Issues
- 🎬 **Media/Travel** (JioCinema, MakeMyTrip) → Live Broadcast Crashes & Ticket/Flight Booking Failures

Complaint volume was analyzed across **10 top-tier apps**: AJIO, Amazon India, CRED, Flipkart, JioCinema, MakeMyTrip, Nykaa, Swiggy, Zepto, and Zomato.

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| **Language** | Python |
| **Web Scraping** | `google-play-scraper` |
| **Deep Learning** | TensorFlow, Keras (LSTM) |
| **ML / Clustering** | scikit-learn (TF-IDF, K-Means) |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 📂 Repository Structure


├── data/                  # Scraped & cleaned review CSVs
├── notebooks/             # EDA, preprocessing, LSTM & clustering notebooks
├── visuals/               # Generated charts, heatmaps, confusion matrix
└── README.md


## 🚀 Future Improvements

- Extend to multi-class sentiment (positive / neutral / negative)
- Replace TF-IDF with transformer-based embeddings (BERT) for richer clustering
- Deploy as an interactive dashboard (Streamlit) for live complaint monitoring
  

---

## 🙋 Author

Built by Shruti Tanwar 
