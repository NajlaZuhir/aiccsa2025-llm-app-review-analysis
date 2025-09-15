# Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM

**Research Implementation for AICCSA 2025**

This repository contains the implementation for the paper **“Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM”** (AICCSA 2025). The framework addresses the limits of traditional star-rating systems by using Large Language Models (LLMs) to capture nuanced feedback that numeric ratings often miss.

---

## 🎯 Research Objectives

Star ratings alone rarely reflect the full story in review text. Our LLM-based framework bridges this gap by:

- **Extracting rich insights** from free-form reviews (aspects, sentiments, recommendations).
- **Capturing linguistic nuances** (hedging, sarcasm, contrastive praise/critique).
- **Providing actionable feedback** for product teams.
- **Enabling interactive analysis** through evidence-grounded Q&A.

---

## 🧩 Framework Overview

**Core Architecture.** A modular, hybrid multi-stage pipeline where each component runs independently or as an end-to-end system, combining traditional NLP baselines with advanced LLM techniques.

- **M1 — Discrepancy Analysis (Baseline):**  
  VADER-based sentiment → mapped to 1–5 scale → absolute difference vs. star rating.

- **M2 — ABSA + Recommendation Mining:**  
  Structured prompting with domain examples to extract **(aspect, sentiment, recommendation)** triples.

- **M3 — LLM-Enhanced Topic Modeling:**  
  BERTopic for clusters, with LLM-generated topic **labels** and **summaries**.

- **M4 — RAG-Based Conversational QA:**  
  Interactive, **evidence-backed** question answering over large review sets.

### Multi-LLM Architecture

- **openai** — primary for complex reasoning & structured outputs  
- **LLaMA-2** — open-source alternative for accessibility & comparison  
- **Mistral** — efficient, high-quality responses  
- **Unified Interface** — seamless provider switching via a standardized `ChatFn` interface

### Multi-Dataset Validation

- Evaluated on [AWARE](https://zenodo.org/records/5528481), 
  [Google Play](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews), 
  and [Spotify](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update) review corpora
  - **Scalable** from small samples to **80K+** reviews with intelligent caching

---

## 🚀 Quick Start

### Installation

1) **Clone the repository**
2) **Install dependencies**
```bash
pip install -r requirements.txt
```
3) **Set environment variables**  
Create a `.env` in the project root:
```env
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
LLAMA_API_KEY=...
```

---

## 📁 Core Architecture (Project Structure)

```
aiccsa2025-llm-app-review-analysis/      # Repository root
│
├── app_reviews_pipeline/                # Main pipeline package
│   ├── __init__.py
│   ├── llm_config.py                    # Multi-provider LLM configuration
│   ├── preprocessing.py                 # Data cleaning & preparation
│   ├── prompt_optimize.py               # Automated prompt optimization
│   ├── quality_judge.py                 # LLM output quality evaluation
│   ├── run_pipeline.py                  # Pipeline orchestration (CLI)
│   ├── user_selection.py                # Interactive CLI utilities
│   │
│   ├── M1_Discrepancy/
│   │   ├── discrepancy.py               # VADER sentiment & discrepancy detection
│   │   ├── discrepancy_plots.ipynb      # Notebook version
│   │   └── __init__.py
│   │
│   ├── M2_Absa_recommendation/
│   │   ├── absa_recommendation.py       # ABSA pipeline
│   │   ├── absa_prompts.py              # ABSA prompts
│   │   ├── absa_LLM_helpers.py          # ABSA utilities
│   │   └── __init__.py
│   │
│   ├── M3_Topic_modeling/
│   │   ├── topic_modeling.py            # BERTopic + LLM labeling
│   │   ├── topic_prompts.py             # Topic labeling prompts
│   │   └── __init__.py
│   │
│   └── M4_Rag_qa/
│       ├── rag_qa.py                    # RAG-based Q&A
│       ├── rag_prompt.py                # RAG prompts/templates
│       ├── rag_qus_samples.txt          # Sample questions
│       └── __init__.py
│
├── data/
│   ├── raw/
│   │   ├── AWARE_Comprehensive.csv
│   │   ├── spotify_reviews.csv
│   │   └── google_play_reviews.csv
│   └── processed/
│       ├── aware_clean.csv
│       ├── spotify_clean.csv
│       ├── google_play_clean.csv
│       └── *_stats.json                 # Dataset statistics
│
├── outputs/
│   ├── absa/
│   ├── discrepancy/
│   ├── topic_modeling/
│   ├── rag_cache/
│   └── prompt_dumps/
│
├── .env
├── requirements.txt
└── README.md
```

---

## 🎮 Usage Guide

### 1) 🧹 Data Preprocessing
Clean and normalize any dataset placed under `data/raw/`:
```bash
python app_reviews_pipeline/preprocessing.py   --dataset aware   # options: aware | spotify | google_play
```
**Output:** Clean CSVs in `data/processed/` + summary stats in `*_stats.json`.

**Interactive Flow (when using `run_pipeline.py`):**
1. Choose dataset (AWARE, Spotify, Google Play)  
2. Select LLM provider and model  
3. Set sample size or use full dataset

### 2) ▶️ Run the full pipeline (interactive)
```bash
python app_reviews_pipeline/run_pipeline.py
```

### 3) Run individual modules

- **M1 — Discrepancy Analysis**
  ```bash
  python -m app_reviews_pipeline.M1_Discrepancy.discrepancy     --input data/processed/aware_clean.csv     --outdir outputs/discrepancy
  ```

- **M2 — ABSA + Recommendation Mining**
  ```bash
  python -m app_reviews_pipeline.M2_Absa_recommendation.absa_recommendation     --input data/processed/spotify_clean.csv     --provider openai --model gpt-4o     --outdir outputs/absa
  ```

- **M3 — Topic Modeling (BERTopic + LLM labels)**
  ```bash
  python -m app_reviews_pipeline.M3_Topic_modeling.topic_modeling     --input data/processed/google_play_clean.csv     --nr-topics 20 --embedding-model all-mpnet-base-v2     --outdir outputs/topic_modeling
  ```

- **M4 — RAG QA**
  ```bash
  python -m app_reviews_pipeline.M4_Rag_qa.rag_qa     --input data/processed/spotify_clean.csv     --cache-dir outputs/rag_cache
  ```

---

## 📊 Outputs

- **`outputs/discrepancy/`** — per-review sentiment, mapped scores, and rating/text gaps  
- **`outputs/absa/`** — extracted triples `(aspect, sentiment, recommendation)`  
- **`outputs/topic_modeling/`** — cluster labels, summaries, visualizations  
- **`outputs/rag_cache/`** — embeddings and indices for QA  
- **`outputs/prompt_dumps/`** — auto-optimized prompts for reproducibility

---

## 🧪 Research Impact

- **Academic Contribution:** LLM framework surpassing traditional baselines on nuanced opinion mining.  
- **Practical Value:** Actionable, feature-level insights beyond star ratings.  
- **Methodological Innovation:** Structured prompting with automated optimization.  
- **Scalability:** Robust from small samples to tens of thousands of reviews.

---

## 💡 Notes

- Put your raw CSVs in `data/raw/`.  
- Preprocessing writes cleaned files to `data/processed/`.  
- Ensure `.env` is configured before running LLM-dependent modules.
