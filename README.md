# Complaint Intelligence Engine (CIE)

**An Unsupervised NLP Decision-Support System for Indian E-Commerce Complaint Analytics**

The **Complaint Intelligence Engine (CIE)** is an end-to-end, twelve-stage machine learning pipeline that transforms raw multilingual Indian e-commerce reviews into actionable operational intelligence. It gives operations and customer-experience teams a repeatable workflow from data ingestion through to prioritization, routing, retention risk quantification, and emerging-issue detection.

## 🚀 Key Features

* **Multilingual & Hinglish Native:** Uses language detection and Multilingual Sentence-BERT to encode English and Hindi/Hinglish scripts into the same semantic embedding space.
* **Zero Manual Labelling:** The core NLP layer leverages K-Means and HDBSCAN to discover complaint archetypes unsupervised.
* **End-to-End Pipeline:** A fully automated pipeline converting unstructured reviews into severity queues, routing assignments, churn risk signals, drafted responses, and drift alerts.
* **Interactive Dashboard:** Includes a 10-module Streamlit decision-support dashboard for live exploration, anomaly tracking, and action taking.

## 📊 Data Sources
* **Google Play Store:** Live reviews fetched via `google-play-scraper` across five major platforms: Myntra, Meesho, Nykaa, Flipkart, and Amazon India.
* **Reddit (Optional):** Keyword-filtered posts ingested via PRAW from relevant subreddits (requires API credentials).

## 📈 Results & Key Findings

The following table summarizes the corpus metrics, clustering discoveries, and machine learning model performances from the documented pipeline run:

| Metric / Task | Model / Method | Score / Value | Key Notes |
| :--- | :--- | :--- | :--- |
| **Final Corpus Size** | - | 2,253 reviews | Across 5 platforms; ~8.7% Hinglish |
| **Clustering Themes** | K-Means | 4 optimal clusters | Delivery Failure, Payment Fraud, Fake Product, App Crash, Quality Mismatch, Positive |
| **Critical Alerts** | Isolation Forest + HDBSCAN | ~113 reviews flagged | Tier 1 severity; Spike detection FPR ~1.2% |
| **Severity Triage** | Logistic Regression | 1.0 (Weighted F1) | *Near-perfect score due to weak-label alignment; requires strict holdouts* |
| **Complaint Routing** | LinearSVC (TF-IDF) | 0.936 (Macro F1) | Routes to 5 operational teams |
| **Churn Risk** | Gradient Boosting | 1.0 (PR-AUC) | Identifies silent, high-risk dissatisfied users |
| **Auto-Responder** | template_v3 | 0.590 (Mean Quality) | Drafts evaluated for empathy, specificity, and actionability |
| **Topic Drift Monitor**| Centroid Shift + JS Divergence| 0.148 (Alert Rate) | Alerts triggered in 4 out of 27 tracked weeks |

## 🧠 Pipeline Architecture

The CIE is orchestrated via two main layers across 12 automated steps:

### 1. Core NLP & Unsupervised Layer (Steps 01-07)
* **01 Ingestion:** Fetches and deduplicates raw reviews.
* **02 Preprocessing:** Cleans text, retains Hinglish, and extracts TF-IDF features.
* **03 Embedding:** Generates 384-dimensional dense vectors using `paraphrase-multilingual-MiniLM-L12-v2`.
* **04 Dimensionality Reduction:** Denoises and reduces via PCA and UMAP (50d → 10d).
* **05 Clustering:** K-Means + HDBSCAN partitions data to discover themes.
* **06 Anomaly Detection:** Isolation Forest + HDBSCAN flags Tier-1 critical reviews and detects volume spikes.
* **07 Visualisation:** Generates static exports of heatmaps, timelines, and distributions.

### 2. Supervised Extension Layer (Steps 08-12)
* **08 Severity:** Predicts escalation risk using programmatic weak labelling.
* **09 Router:** Classifies complaints into operational queues using K-Means clusters as training targets.
* **10 Churn:** Quantifies churn risk based on recent silent dissatisfaction and severity.
* **11 Auto-Responder:** Generates context-aware, quality-scored response drafts.
* **12 Drift Monitor:** Computes Embedding Centroid Shifts and JS Divergence to detect emerging issues.

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Data Processing & Stats:** Pandas, NumPy, SciPy, Statsmodels
* **NLP & Embeddings:** Sentence-Transformers (SBERT), langdetect, PyTorch
* **Machine Learning:** Scikit-Learn, UMAP, HDBSCAN, LightGBM, Imbalanced-Learn
* **App & Visualization:** Streamlit, Plotly, Matplotlib, Seaborn

## 💻 Installation & Usage

1. **Set up the environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
