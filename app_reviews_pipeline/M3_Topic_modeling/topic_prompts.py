# topic_prompts.py

# ── Topic Quality Evaluation ────────────────────────────────────────────
TOPIC_QUALITY_PROMPT = """
Evaluate the quality of this topic from app reviews by scoring these criteria (0-10):

TOPIC INFO:
Label: {label}
Summary: {summary}
Keywords: {keywords}

SAMPLE REVIEWS:
{examples}

SCORING CRITERIA:
1. Coherence (0-10): How well do the keywords, reviews, and summary align with a single clear theme?
2. Distinctiveness (0-10): How unique/specific is this topic vs generic feedback?
3. Summary Quality (0-10): Is the summary clear, accurate, and supported by evidence?
4. Label Clarity (0-10): Is the label specific and representative of the content?
5. Evidence Strength (0-10): Do the examples provide strong support for this topic?

For each criterion, provide:
- Score (0-10)  
- Brief justification (1 sentence)
- Improvement suggestion if score < 7

Return as JSON with fields:
{
  "scores": {
    "coherence": {"score": N, "reason": "...", "suggestion": "..."},
    "distinctiveness": {"score": N, "reason": "...", "suggestion": "..."},
    "summary_quality": {"score": N, "reason": "...", "suggestion": "..."},
    "label_clarity": {"score": N, "reason": "...", "suggestion": "..."},
    "evidence_strength": {"score": N, "reason": "...", "suggestion": "..."}
  },
  "overall_score": N,  # weighted average
  "primary_issues": ["..."],  # key problems if any score < 7
  "suggested_label": "...",  # only if label_clarity < 7
  "suggested_summary": "..."  # only if summary_quality < 7
}
""".strip()

# ── Label (plain text) ──────────────────────────────────────────────────────
TOPIC_LABEL_PROMPT = """
You will be given the TOP KEYWORDS for one topic from app reviews.
Return a concise TOPIC LABEL (2–5 words), Title Case, no quotes or punctuation.

Rules:
- Focus on the shared concept in the keywords (features, issues, actions).
- Avoid brand/obvious words (e.g., "Spotify", "app", "music", "playlist") unless essential.
- Avoid generic labels like "General Feedback" or "Bugs" if a more specific label exists.
- No emojis, no extra text. Output the label only.

Top Keywords: {keywords_csv}
""".strip()


# ── Summary (plain text) ────────────────────────────────────────────────────
TOPIC_SUMMARY_PROMPT = """
You will be given a few representative review snippets for one topic.
Write a 1–2 sentence summary (max ~45 words) of what users say about this topic.
Return plain text only (no bullets, no quotes, no recommendations).

Rules:
- Base the summary ONLY on the given snippets; do not add facts.
- Capture the main theme and notable sub-points (e.g., sentiment, cause, effect).
- Keep neutral tone; no instructions like “should fix”.

Representative snippets:
{sample_block}
""".strip()
