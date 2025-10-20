# Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM

This repository contains the implementation for the paper **[Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM](https://arxiv.org/pdf/2509.20953v1)** accepted to **[AICCSA 2025](https://lnkd.in/gUzgpBZd)**. The framework addresses the limits of traditional star-rating systems by using Large Language Models (LLMs) to capture nuanced feedback that numeric ratings often miss.

---

## 🎯 Research Objectives

Star ratings alone rarely reflect the full story in review text. Our LLM-based framework bridges this gap by:

- **Extracting rich insights** from free-form reviews (aspects, sentiments, recommendations).
- **Capturing linguistic nuances** (hedging, sarcasm, contrastive praise/critique).
- **Providing actionable feedback** for product teams.
- **Enabling interactive analysis** through evidence-grounded Q&A.

---

## 🧩 Framework Overview

![Proposed Modular Framework for LLM-Based Sentiment Analysis](aiccsa2025-llm-app-review-analysis
/framework.png)


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

- **OpenAI** — primary for complex reasoning & structured outputs  
- **Llama 2** — open-source alternative for accessibility & comparison  
- **Mistral AI** — efficient, high-quality responses  
- **Unified Interface** — seamless provider switching via a standardized `ChatFn` interface

### Multi-Dataset Validation
- Evaluated on three heterogeneous app review corpora: [AWARE](https://zenodo.org/records/5528481), [Google Play](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews) and [Spotify](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update)
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

## 📁 Project Structure

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
├── .env
├── requirements.txt
└── README.md
```

---

## 🎮 Usage Guide

### 1) 🧹 Data Preprocessing
Clean and normalize any dataset placed under `data/raw/`:
```bash
python app_reviews_pipeline/preprocessing.py
```
**Output:** Clean CSVs in `data/processed/` + summary stats in `*_stats.json`.

### 2) ▶️ Run the full pipeline
```bash
python app_reviews_pipeline/run_pipeline.py
```
**Interactive Flow:**
1. Select one of the four Modules (M1-M4)
2. Choose dataset (AWARE, Spotify, Google Play)  
3. Select LLM provider and model  
4. Set sample size or use full dataset
   
**Note:** you can also run the modules individually.

---

## 📊 Outputs

- **`outputs/discrepancy/`** — per-review sentiment, mapped scores, and rating/text gaps  
- **`outputs/absa/`** — extracted triples `(aspect, sentiment, recommendation)`  
- **`outputs/topic_modeling/`** — cluster labels, summaries, visualizations  
- **`Rag outputs:`** —  in the terminal
- **Note:** outputs folder gets created in the Main pipeline folder 
---

## 🧪 Research Impact

- **Academic Contribution:** LLM framework surpassing traditional baselines.  
- **Practical Value:** Actionable, feature-level insights beyond star ratings.  
- **Methodological Innovation:** Structured prompting with automated optimization.  
- **Scalability:** Robust from small samples to tens of thousands of reviews.

---

## 💡 Notes

- Put your raw CSVs in `data/raw/`.  
- Preprocessing writes cleaned files to `data/processed/`.  
- Ensure `.env` is configured before running LLM-dependent modules.
- Feedback and contributions are welcome.
