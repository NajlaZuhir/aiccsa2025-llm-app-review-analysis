"""
Topic Modeling with BERTopic and LLM-assisted labeling.

Run from repo root:
  python -m app_reviews_pipeline.M3_Topic_modeling.topic_modeling

Requires:
  pip install bertopic hdbscan umap-learn scikit-learn tqdm python-dotenv sentence-transformers plotly
"""

# ===========================
# Imports & Configuration
# ===========================
import os
import sys
import json
import random
import re
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# ===========================
# Path Configuration
# ===========================
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]          # .../app_reviews_pipeline
THIS_DIR = HERE.parent              # .../app_reviews_pipeline/M3_Topic_modeling
for p in (PKG_ROOT, THIS_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# ===========================
# Custom Imports
# ===========================
from user_selection import (  # type: ignore
    choose_dataset,
    choose_provider_and_model,
    choose_sample_size,
    choose_prompt_tuning,
)
from llm_config import get_llm, ChatFn  # type: ignore
import logging
logging.getLogger("BERTopic").setLevel(logging.WARNING)

# prompts import works whether called as module or script
try:
    from M3_Topic_modeling.topic_prompts import (  # type: ignore
        TOPIC_LABEL_PROMPT,
        TOPIC_SUMMARY_PROMPT,
        TOPIC_QUALITY_PROMPT,
    )
except Exception:
    from topic_prompts import (  # type: ignore
        TOPIC_LABEL_PROMPT,
        TOPIC_SUMMARY_PROMPT,
        TOPIC_QUALITY_PROMPT,
    )

# prompt optimizer (package import + fallback)
try:
    from app_reviews_pipeline.prompt_optimize import optimize_prompt  # type: ignore
except Exception:
    try:
        from prompt_optimize import optimize_prompt  # type: ignore
    except Exception:
        optimize_prompt = None  # tuning remains optional

load_dotenv()


# ===========================
# Helper Functions
# ===========================
def _extract_json_from_response(response: str) -> str:
    """
    Extract JSON from LLM response that might be wrapped in markdown code blocks.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Clean JSON string
    """
    # Strip whitespace
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith('```json'):
        # Find the end of the code block
        end_marker = response.find('```', 7)  # Start looking after '```json'
        if end_marker != -1:
            response = response[7:end_marker].strip()
    elif response.startswith('```'):
        # Handle generic code blocks
        end_marker = response.find('```', 3)
        if end_marker != -1:
            response = response[3:end_marker].strip()
    
    return response

def _safe_format(template: str, **fields) -> str:
    """Format a prompt but replace any unknown {placeholders} with ''."""
    def repl(m):
        k = m.group(1)
        return str(fields.get(k, ""))
    return re.sub(r"\{([A-Za-z0-9_]+)\}", repl, template)


# ===========================
# LLM Labeling Functions
# ===========================
def llm_label_for_keywords(
    keywords: List[str],
    chat_fn: ChatFn,
    max_tokens: int = 24,
    label_prompt: str = TOPIC_LABEL_PROMPT,
) -> str:
    prompt = _safe_format(label_prompt, keywords_csv=", ".join(keywords))
    messages = [
        {"role": "system", "content": "Return only the label. No quotes."},
        {"role": "user", "content": prompt},
    ]
    try:
        out = chat_fn(messages, temperature=0.2, max_tokens=max_tokens).strip()
        return out or "Topic"
    except Exception:
        return "Topic"


def llm_summary_for_examples(
    examples: List[str],
    chat_fn: ChatFn,
    max_tokens: int = 120,
    summary_prompt: str = TOPIC_SUMMARY_PROMPT,
) -> str:
    block = "\n- " + "\n- ".join(x.strip().replace("\n", " ")[:220] for x in examples[:5])
    prompt = _safe_format(summary_prompt, sample_block=block)
    messages = [
        {"role": "system", "content": "Return 1–2 sentences. No bullets, no quotes."},
        {"role": "user", "content": prompt},
    ]
    try:
        out = chat_fn(messages, temperature=0.2, max_tokens=max_tokens).strip()
        return out or ""
    except Exception:
        return ""

def evaluate_topic_quality(
    topic_id: int,
    label: str,
    summary: str,
    keywords: List[str],
    examples: List[str],
    chat_fn: ChatFn,
    quality_prompt: str = TOPIC_QUALITY_PROMPT,
) -> Dict:
    """
    Use LLM to evaluate topic quality based on multiple criteria.
    
    Args:
        topic_id: Topic identifier
        label: Topic label (assigned by LLM)
        summary: Topic summary (from LLM)
        keywords: List of top keywords
        examples: List of example reviews
        chat_fn: LLM chat function
        quality_prompt: Prompt template for quality evaluation
        
    Returns:
        Dict with quality scores, reasons, and suggestions
    """
    # Format examples for prompt
    example_block = "\n".join(f"{i+1}. {x[:300]}" for i, x in enumerate(examples[:5]))
    
    # Format prompt
    prompt = _safe_format(
        quality_prompt,
        label=label,
        summary=summary,
        keywords=", ".join(keywords),
        examples=example_block
    )
    
    messages = [
        {"role": "system", "content": "Return valid JSON only. Include all fields from the schema."},
        {"role": "user", "content": prompt},
    ]
    
    error_result = {
        "topic_id": topic_id,
        "scores": {
            "coherence": {"score": 0, "reason": "Evaluation failed", "suggestion": None},
            "distinctiveness": {"score": 0, "reason": "Evaluation failed", "suggestion": None},
            "summary_quality": {"score": 0, "reason": "Evaluation failed", "suggestion": None},
            "label_clarity": {"score": 0, "reason": "Evaluation failed", "suggestion": None},
            "evidence_strength": {"score": 0, "reason": "Evaluation failed", "suggestion": None}
        },
        "overall_score": 0,
        "primary_issues": ["Evaluation failed"],
    }
    
    try:
        result = chat_fn(messages, temperature=0.1, max_tokens=1000)
        if not isinstance(result, str):
            return error_result
            
        try:
            clean_json = _extract_json_from_response(result)
            data = json.loads(clean_json)
            # Validate required fields
            required = ["scores", "overall_score", "primary_issues"]
            if not all(k in data for k in required):
                return error_result
                
            data["topic_id"] = topic_id
            return data
        except json.JSONDecodeError:
            return error_result
    except Exception as e:
        error_result["primary_issues"] = [f"Evaluation failed: {str(e)}"]
        error_result["error"] = str(e)
        return error_result



# ===========================
# Main Function
# ===========================
def main():
    print("\n=== Topic Modeling (BERTopic) ===")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # ===========================
    # Data Loading & Setup
    # ===========================
    # 1) Dataset (clean CSV with 'review_id' + 'review_text')
    ds, csv_path = choose_dataset()
    df = pd.read_csv(csv_path, usecols=["review_id", "review_text"])

    # 2) Provider + model (shared UX)
    provider, model = choose_provider_and_model()
    chat_fn = get_llm(provider, model)

    # 3) Sample size
    sample_n = choose_sample_size(len(df))
    if sample_n:
        df = df.sample(n=sample_n, random_state=42)
    docs = df["review_text"].astype(str).tolist()

    # ===========================
    # Output Directory Setup
    # ===========================
    out_dir = Path(f"outputs/topic_modeling/{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================
    # Initial Topic Run & Quality Check
    # ===========================
    print("\nRunning initial topic analysis...")
    # Use safe settings for initial run
    safe_model = BERTopic(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=12, min_samples=None, prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=1.0),
        calculate_probabilities=True,
        verbose=False,
    )
    
    initial_docs = docs[:min(500, len(docs))]  # Sample for quality check
    try:
        topics, probs = safe_model.fit_transform(initial_docs)
        
        # Get initial topic info and run LLM labeling
        topic_info = safe_model.get_topic_info()
        valid_topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]
        
        if valid_topic_ids:
            by_topic = {}
            for i, t in enumerate(topics):
                by_topic.setdefault(int(t), []).append(i)
                
            # Do basic labeling for quality check
            labeled_rows = []
            for tid in valid_topic_ids:
                words = safe_model.get_topic(tid) or []
                keywords = [w for (w, _) in words][:10]
                doc_ids = by_topic.get(int(tid), [])
                try:
                    reps = safe_model.get_representative_docs(tid) or []
                except:
                    reps = []
                examples = (reps if reps else [initial_docs[i] for i in doc_ids[:5]])[:5]
                
                label = llm_label_for_keywords(keywords, chat_fn=chat_fn, label_prompt=TOPIC_LABEL_PROMPT)
                summary = llm_summary_for_examples(examples, chat_fn=chat_fn, summary_prompt=TOPIC_SUMMARY_PROMPT)
                
                labeled_rows.append({
                    "topic_id": int(tid),
                    "label": label,
                    "summary": summary,
                    "top_keywords": ", ".join(keywords),
                    "document_count": len(doc_ids),
                    "examples": " | ".join(e.replace("\n", " ")[:300] for e in examples),
                })
            
            topic_df = pd.DataFrame(labeled_rows)
            assign_df = pd.DataFrame({
                "review_id": range(len(initial_docs)),  # Temporary IDs for quality check
                "topic_id": [int(t) for t in topics],
                "topic_prob": [float(max(p)) if hasattr(p, '__len__') else float(p) for p in probs],
            })
            
            # Quality check (always proceed with optimization)
            try:
                from quality_judge import needs_optimization
                needs_opt, reason = needs_optimization("topic", (topic_df, assign_df))
                print(f"\n[Quality Check] {reason}")
            except Exception as e:
                print(f"[Quality Check] Skipped: {e}")
        else:
            print("\n[Quality Check] No valid topics found in initial run")
    except Exception as e:
        print(f"\n[Quality Check] Failed: {e}")
        enable_tune = choose_prompt_tuning(default=False)

    # ===========================
    # Prompt Optimization
    # ===========================
    print("\nOptimizing and evaluating prompts...")
    label_p, summary_p = TOPIC_LABEL_PROMPT, TOPIC_SUMMARY_PROMPT
    
    if optimize_prompt is None:
        print("[PromptTuner] Optimizer not available; using default prompts.")
    else:
        # Optimize label prompt
        label_result = optimize_prompt(
            dataset=ds,
            sample_texts=docs[:500],
            chat_fn=chat_fn,
            base_prompt=TOPIC_LABEL_PROMPT,
            provider=provider,
            model=model,
            enabled=True,
            prompt_type="topic_label"
        )
        label_p = label_result[0] if isinstance(label_result, tuple) else label_result
        if isinstance(label_result, tuple):
            print("\n[Topic Label] Evaluation:")
            print(f"Selected: {'Optimized' if label_result[1]['recommendation'] == 'b' else 'Base'} prompt")
            print(f"Confidence: {label_result[1].get('confidence', 0)*100:.1f}%")
            print(f"Reason: {label_result[1].get('explanation', 'N/A')}")
        
        # Optimize summary prompt
        summary_result = optimize_prompt(
            dataset=ds,
            sample_texts=docs[:500],
            chat_fn=chat_fn,
            base_prompt=TOPIC_SUMMARY_PROMPT,
            provider=provider,
            model=model,
            enabled=True,
            prompt_type="topic_summary"
        )
        summary_p = summary_result[0] if isinstance(summary_result, tuple) else summary_result
        if isinstance(summary_result, tuple):
            print("\n[Topic Summary] Evaluation:")
            print(f"Selected: {'Optimized' if summary_result[1]['recommendation'] == 'b' else 'Base'} prompt")
            print(f"Confidence: {summary_result[1].get('confidence', 0)*100:.1f}%")
            print(f"Reason: {summary_result[1].get('explanation', 'N/A')}")
        
        # Save the tuned prompts
        (out_dir / "tuned_prompts.json").write_text(
            json.dumps({
                "label_prompt": label_p,
                "summary_prompt": summary_p,
                "label_evaluation": label_result[1] if isinstance(label_result, tuple) else None,
                "summary_evaluation": summary_result[1] if isinstance(summary_result, tuple) else None
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


    # ===========================
    # BERTopic Model Configuration
    # ===========================
    print("\nFitting BERTopic…")

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42)
    
    # Make min_df adaptive so small samples still produce topics


    n_docs = len(docs)
    if n_docs < 300:
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=1.0)
    else:
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                    min_df=max(2, int(0.005*n_docs)), max_df=0.95)

    max_df = 0.95
    min_df_val = max(2, int(0.005 * n_docs))  # ≥2 docs or 0.5%

    # guard: ensure min_df ≤ floor(max_df * n_docs)
    max_allowed = int(math.floor(max_df * n_docs)) if isinstance(max_df, float) else int(max_df)
    min_df_val = min(min_df_val, max(1, max_allowed))

    # --- vectorizer (safe default for small samples) ---
    safe_vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,     # always valid
        max_df=1.0,   # avoids max_df < min_df check for tiny topic-doc sets
    )

    hdb = HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True)  # more granular clusters

    # ===========================
    # BERTopic Model Training
    # ===========================
    topic_model = BERTopic(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        umap_model=umap_model,
        hdbscan_model=hdb,
        vectorizer_model=safe_vectorizer,
        calculate_probabilities=True,
        verbose=False,
    )

    # Try fit; if you later switch back to stricter df thresholds and it fails,
    # fall back automatically to the safe vectorizer.
    try:
        topics, probs = topic_model.fit_transform(docs)
    except ValueError as e:
        if "max_df corresponds" in str(e):
            print("(retry) Falling back to safe vectorizer (min_df=1, max_df=1.0).")
            safe_vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=1.0)
            topic_model = BERTopic(
                embedding_model="sentence-transformers/all-mpnet-base-v2",
                umap_model=umap_model,
                hdbscan_model=hdb,
                vectorizer_model=safe_vectorizer,
                calculate_probabilities=True,
                verbose=True,
            )
            topics, probs = topic_model.fit_transform(docs)
        else:
            raise

    # ===========================
    # Topic Analysis & Reduction
    # ===========================
    topic_info = topic_model.get_topic_info()
    valid_topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]
    print(f"\nDiscovered {len(valid_topic_ids)} topics (excluding outliers).")
    if not valid_topic_ids:
        print("No non-outlier topics discovered. Try lowering min_cluster_size or sampling more docs.")
        return

    raw = input("Reduce to N (< current) or press Enter to skip: ").strip()
    if raw:
        try:
            n = int(raw)
            if n < len(valid_topic_ids):
                topic_model = topic_model.reduce_topics(docs, nr_topics=n)
                topics, probs = topic_model.transform(docs)  # recompute for reduced model
                topic_info = topic_model.get_topic_info()
                valid_topic_ids = [t for t in topic_info["Topic"].tolist() if t != -1]
                print(f"Reduced to {len(valid_topic_ids)} topics.")
            else:
                print(f"(skip) {n} ≥ {len(valid_topic_ids)}.")
        except Exception as e:
            print(f"(note) reduction skipped: {e}")

    # ===========================
    # Topic-Document Mapping
    # ===========================
    by_topic: Dict[int, List[int]] = {}
    for i, t in enumerate(topics):
        by_topic.setdefault(int(t), []).append(i)

    # ===========================
    # LLM Topic Labeling & Summarization
    # ===========================
    labeled_rows = []
    quality_scores = []
    print("\nLabeling, summarizing & evaluating topics with LLM…")
    for tid in tqdm(valid_topic_ids, unit="topic", leave=False, dynamic_ncols=True):
        words = topic_model.get_topic(tid) or []
        keywords = [w for (w, _) in words][:10]
        doc_ids = by_topic.get(int(tid), [])

        # Prefer BERTopic’s representative docs; fallback to first few assigned docs
        try:
            reps = topic_model.get_representative_docs(tid) or []
        except Exception:
            reps = []
        examples = (reps if reps else [docs[i] for i in doc_ids[:5]])[:5]

        # Get initial label and summary
        label = llm_label_for_keywords(keywords, chat_fn=chat_fn, label_prompt=label_p)
        summary = llm_summary_for_examples(examples, chat_fn=chat_fn, summary_prompt=summary_p)
        
        # Evaluate topic quality
        quality = evaluate_topic_quality(
            topic_id=int(tid),
            label=label,
            summary=summary,
            keywords=keywords,
            examples=examples,
            chat_fn=chat_fn
        )
        quality_scores.append(quality)
        
        # Use improved label/summary if suggested
        if quality.get("suggested_label") and quality.get("scores", {}).get("label_clarity", {}).get("score", 10) < 7:
            label = quality["suggested_label"]
        if quality.get("suggested_summary") and quality.get("scores", {}).get("summary_quality", {}).get("score", 10) < 7:
            summary = quality["suggested_summary"]

        labeled_rows.append({
            "topic_id": int(tid),
            "label": label,
            "summary": summary,
            "top_keywords": ", ".join(keywords),
            "document_count": len(doc_ids),
            "examples": " | ".join(e.replace("\n", " ")[:300] for e in examples),
            "quality_score": quality.get("overall_score", 0),
            "quality_issues": "; ".join(quality.get("primary_issues", [])),
            "coherence": quality.get("scores", {}).get("coherence", {}).get("score", 0),
            "distinctiveness": quality.get("scores", {}).get("distinctiveness", {}).get("score", 0),
        })

    topics_df = pd.DataFrame(labeled_rows).sort_values("document_count", ascending=False)

    # ===========================
    # Document-Topic Assignment
    # ===========================
    probs_list = []
    for p in probs:
        try:
            probs_list.append(float(np.max(p)))
        except Exception:
            try:
                probs_list.append(float(p))
            except Exception:
                probs_list.append(float("nan"))

    assign_df = pd.DataFrame({
        "review_id": df["review_id"].values,
        "topic_id": [int(t) for t in topics],
        "topic_prob": probs_list,
    })

    # ===========================
    # Output Generation
    # ===========================
    try:
        (out_dir / "topics.html").write_text(topic_model.visualize_topics().to_html(), encoding="utf-8")
        (out_dir / "barchart.html").write_text(topic_model.visualize_barchart().to_html(), encoding="utf-8")
        (out_dir / "hierarchy.html").write_text(topic_model.visualize_hierarchy().to_html(), encoding="utf-8")
    except Exception as e:
        print(f"(note) Could not save HTML visuals: {e}")

    # Save topics data in both CSV and JSONL formats
    topics_path = out_dir / f"{ds}_topics_labeled.csv"
    topics_jsonl_path = out_dir / f"{ds}_topics_labeled.jsonl"
    
    # Save CSV
    topics_df.to_csv(topics_path, index=False)
    
    # Save JSONL
    with open(topics_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in topics_df.iterrows():
            topic_data = row.to_dict()
            # Parse examples into a list for better JSON structure
            if 'examples' in topic_data:
                topic_data['examples'] = topic_data['examples'].split(' | ')
            if 'top_keywords' in topic_data:
                topic_data['top_keywords'] = [k.strip() for k in topic_data['top_keywords'].split(',')]
            f.write(json.dumps(topic_data, ensure_ascii=False) + '\n')

    # Save quality analysis
    quality_df = pd.DataFrame([{
        'topic_id': q['topic_id'],
        'overall_score': q.get('overall_score', 0),
        'primary_issues': '; '.join(q.get('primary_issues', [])),
        'coherence': q.get('scores', {}).get('coherence', {}).get('score', 0),
        'distinctiveness': q.get('scores', {}).get('distinctiveness', {}).get('score', 0),
        'summary_quality': q.get('scores', {}).get('summary_quality', {}).get('score', 0),
        'label_clarity': q.get('scores', {}).get('label_clarity', {}).get('score', 0),
        'evidence_strength': q.get('scores', {}).get('evidence_strength', {}).get('score', 0),
    } for q in quality_scores])
    
    quality_path = out_dir / f"{ds}_topic_quality.csv"
    quality_jsonl_path = out_dir / f"{ds}_topic_quality.jsonl"
    
    # Save CSV
    quality_df.to_csv(quality_path, index=False)
    
    # Save JSONL
    with open(quality_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in quality_df.iterrows():
            quality_data = row.to_dict()
            # Convert numpy/pandas types to native Python types for JSON serialization
            quality_data = {k: float(v) if isinstance(v, (np.floating, pd.Series)) else v 
                          for k, v in quality_data.items()}
            # Split primary issues into a list if it exists
            if 'primary_issues' in quality_data and isinstance(quality_data['primary_issues'], str):
                quality_data['primary_issues'] = [i.strip() for i in quality_data['primary_issues'].split(';')]
            f.write(json.dumps(quality_data, ensure_ascii=False) + '\n')

    # Quality summary
    print("\n=== Quality Summary ===")
    try:
        score_columns = [c for c in quality_df.columns 
                        if c.endswith('_score') or 
                        c in ['coherence', 'distinctiveness', 'evidence_strength']]
        if score_columns:
            avg_scores = quality_df[score_columns].mean()
            print(f"Average Scores:")
            for metric, score in avg_scores.items():
                if pd.notna(score):  # Only print valid scores
                    print(f"  • {metric.replace('_', ' ').title()}: {score:.1f}")
            
            # Only check for low quality if overall_score exists
            if 'overall_score' in quality_df.columns:
                low_quality = quality_df[quality_df['overall_score'] < 7]['topic_id'].tolist()
                if low_quality:
                    print(f"\nTopics Needing Improvement: {len(low_quality)}")
                    print(f"  Topic IDs: {', '.join(map(str, low_quality))}")
                else:
                    print("\nAll topics meet quality thresholds (score ≥ 7)")
        else:
            print("No quality metrics available")
    except Exception as e:
        print(f"Could not generate quality summary: {str(e)}")

    print("\nSaved files:")
    print(f"  • Topics CSV:     {topics_path}")
    print(f"  • Topics JSONL:   {topics_jsonl_path}")
    print(f"  • Quality CSV:    {quality_path}")
    print(f"  • Quality JSONL:  {quality_jsonl_path}")
    print(f"Provider/Model: {provider} / {model}")


if __name__ == "__main__":
    main()
