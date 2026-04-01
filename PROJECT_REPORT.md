# Complaint Intelligence Engine - Project Report

## 1. Project Overview

The **Complaint Intelligence Engine** is an unsupervised NLP analytics pipeline for Indian e-commerce complaints. It ingests review data, preprocesses multilingual text (including Hinglish), generates semantic embeddings, performs dimensionality reduction and clustering, flags anomalies, detects weekly spikes, and produces dashboard-ready outputs and visualizations.

## 2. Objectives

- Collect complaint-related review data from e-commerce platforms.
- Build an end-to-end unsupervised intelligence pipeline.
- Discover complaint archetypes using clustering.
- Detect risky/critical outlier complaints using anomaly models.
- Produce visual and tabular outputs for stakeholder decision support.

## 3. Tech Stack

- **Language:** Python 3.12
- **Core libraries:** pandas, numpy, scipy, scikit-learn
- **NLP/Embeddings:** sentence-transformers, torch, langdetect
- **Reduction/Clustering:** PCA, UMAP, t-SNE, K-Means, HDBSCAN
- **Anomaly detection:** Isolation Forest, One-Class SVM
- **Visualization/UI:** matplotlib, seaborn, plotly, streamlit

## 4. Repository Structure

- `01_ingest.py`: Live data ingestion and merge
- `02_preprocess.py`: Cleaning, feature engineering, TF-IDF
- `03_embed.py`: Sentence-BERT embedding generation
- `04_reduce.py`: PCA/UMAP/t-SNE reduction
- `05_cluster.py`: K-Means/HDBSCAN clustering and profiling
- `06_anomaly.py`: Anomaly scoring and spike analytics
- `07_visualize.py`: Static figure exports
- `run_pipeline.py`: End-to-end orchestrator
- `app.py`: Streamlit dashboard

## 5. Execution Summary

Environment setup and full execution were performed in a Python virtual environment.

### 5.1 Command Execution

- Virtual environment created: `.venv`
- Dependencies installed from `requirements.txt`
- End-to-end run executed with:
  - `.\.venv\Scripts\python run_pipeline.py`
- Dashboard startup verified with:
  - `.\.venv\Scripts\streamlit run app.py --server.headless true --server.port 8501`

### 5.2 Pipeline Runtime Outcome

The complete pipeline ran successfully (`Pipeline completed successfully.`) and produced all major artifacts.

## 6. Data and Model Outcomes

### 6.1 Ingestion and Cleaning

- Google Play reviews fetched: **4768**
- Reddit fetch failed due missing/unauthenticated credentials (401 responses)
- Post merge + clean dataset: **2389 rows**
- After preprocessing and de-duplication: **2253 rows**

### 6.2 Text/Feature Engineering

- Hinglish reviews identified: **196 (8.7%)**
- TF-IDF matrix shape: **2253 x 2642**

### 6.3 Embeddings and Reduction

- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- Embedding shape: **(2253, 384)**
- PCA(50) cumulative explained variance: **0.7976**
- Reduced spaces produced:
  - `umap_10.npy` (10D)
  - `tsne_2d.npy` (2D)

### 6.4 Clustering and Anomaly

- Selected K-Means cluster count: **k=4** (max silhouette)
- KMeans vs HDBSCAN ARI: **0.6748**
- Anomaly step completed; warning indicated insufficient positive rows for robust One-Class SVM fitting, so fallback defaults were used.

## 7. Generated Artifacts

### 7.1 Data Outputs (`data/`)

- `raw_reviews.csv`
- `cleaned_reviews.csv`
- `tfidf_matrix.npz`, `tfidf_vocab.json`
- `embeddings.npy`
- `pca_50.npy`, `umap_10.npy`, `tsne_2d.npy`
- `clustered_reviews.csv`, `cluster_profiles.csv`, `k_selection.csv`
- `final_reviews.csv`, `spike_report.csv`, `variance_report.csv`

### 7.2 Model Outputs (`models/`)

- `pca_model.pkl`
- `umap_model.pkl`
- `kmeans.pkl`
- `hdbscan.pkl`
- `isoforest.pkl`
- `ocsvm.pkl`

### 7.3 Figure Outputs (`figures/`)

- `k_selection_silhouette.png`
- `platform_volume.png`
- `tier1_by_platform.png`

## 8. Operational Notes

- Reddit collection is currently limited because credentials were not provided; the pipeline handled this gracefully and continued using Google Play data.
- On Windows, package installation showed intermittent `WinError 32` file lock interruptions; re-running installation allowed core runtime dependencies to become usable.
- Hugging Face cache warns about missing symlink support on Windows unless Developer Mode/admin mode is enabled.

## 9. Project Strengths

- Full end-to-end modular pipeline from ingestion to insight artifacts.
- Strong multilingual embedding strategy suitable for mixed language complaints.
- Combined clustering and anomaly layers provide both pattern discovery and risk triage.
- Streamlit dashboard support for stakeholder-facing analytics.

## 10. Recommendations

- Add Reddit API credentials via `.env` for complete multi-source intelligence.
- Add automated tests and CI checks for each pipeline stage.
- Add a deterministic sample dataset for offline reproducible runs.
- Add error-retry logic and richer logging around external API failures.
- Consider GPU-optional acceleration path for embedding generation.

## 11. Final Status

Project execution status: **SUCCESSFUL (pipeline + dashboard startup validated)**.
