# Complaint Intelligence Engine

End-to-end complaint intelligence and decision-support platform for Indian e-commerce reviews (Google Play + optional Reddit), with unsupervised discovery + supervised model benchmarking + Streamlit operations dashboard.

## Features

- Live ingestion from Myntra, Meesho, Nykaa, Flipkart, Amazon India + Reddit
- Hinglish-aware preprocessing and language detection
- Multilingual Sentence-BERT embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- PCA -> UMAP -> t-SNE dimensionality reduction
- K-Means + HDBSCAN clustering
- Isolation Forest + One-Class SVM anomaly detection
- Weekly complaint spike detection (rolling z-score)
- Supervised severity triage model comparison and selection
- Complaint auto-routing model comparison and selection
- Churn risk proxy modeling and watchlist
- Auto-response generator with quality scoring and best-generator selection
- Topic drift detection across weekly windows
- 10-tab Streamlit decision-support dashboard

## Project Structure

- `01_ingest.py` - Data ingestion and merge
- `02_preprocess.py` - Cleaning, features, TF-IDF
- `03_embed.py` - Sentence-BERT embeddings
- `04_reduce.py` - PCA, UMAP, t-SNE
- `05_cluster.py` - K-Means, HDBSCAN, profiles
- `06_anomaly.py` - Anomalies, Tier 1 critical, spikes
- `07_visualize.py` - Static chart exports
- `08_severity_modeling.py` - Weak labels, severity model comparison, best model select
- `09_router_modeling.py` - Cluster-to-router supervised training + model comparison
- `10_churn_modeling.py` - Churn risk proxy modeling + model comparison
- `11_response_modeling.py` - Response generator comparison + quality scoring
- `12_drift_modeling.py` - Topic drift monitoring and alerts
- `run_pipeline.py` - One-command end-to-end run
- `ml_utils.py` - Shared benchmarking utilities + selected model registry
- `app.py` - Streamlit dashboard

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` (optional for authenticated Reddit):

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

## Run End-to-End Pipeline

```bash
python run_pipeline.py
```

## Run Streamlit

```bash
streamlit run app.py
```

## Extension Outputs

- `reports/metrics_severity.csv`
- `reports/metrics_router.csv`
- `reports/metrics_churn.csv`
- `reports/metrics_response.csv`
- `reports/metrics_drift.csv`
- `models/selected_models.json`
- `data/final_reviews_scored.csv`
- `data/final_reviews_routed.csv`
- `data/final_reviews_churn.csv`
- `data/final_reviews_response.csv`
- `data/topic_drift_report.csv`

## GitHub Repository Creation

```bash
git init
git add .
git commit -m "Initial commit: complaint intelligence engine end-to-end pipeline"
gh repo create complaint-intelligence-engine --public --source . --remote origin --push
```

## Streamlit Deployment

1. Push project to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** and select your repo.
4. Set entrypoint as `app.py`.
5. Add secrets (if needed) in Streamlit app settings.
6. Deploy and share public URL.
