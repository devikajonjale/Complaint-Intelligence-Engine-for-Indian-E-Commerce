# Complaint Intelligence Engine

Unsupervised NLP pipeline for discovering complaint archetypes in Indian e-commerce reviews (Google Play + Reddit), with anomaly detection and Streamlit dashboard.

## Features

- Live ingestion from Myntra, Meesho, Nykaa, Flipkart, Amazon India + Reddit
- Hinglish-aware preprocessing and language detection
- Multilingual Sentence-BERT embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- PCA -> UMAP -> t-SNE dimensionality reduction
- K-Means + HDBSCAN clustering
- Isolation Forest + One-Class SVM anomaly detection
- Weekly complaint spike detection (rolling z-score)
- 4-page Streamlit dashboard

## Project Structure

- `01_ingest.py` - Data ingestion and merge
- `02_preprocess.py` - Cleaning, features, TF-IDF
- `03_embed.py` - Sentence-BERT embeddings
- `04_reduce.py` - PCA, UMAP, t-SNE
- `05_cluster.py` - K-Means, HDBSCAN, profiles
- `06_anomaly.py` - Anomalies, Tier 1 critical, spikes
- `07_visualize.py` - Static chart exports
- `run_pipeline.py` - One-command end-to-end run
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
