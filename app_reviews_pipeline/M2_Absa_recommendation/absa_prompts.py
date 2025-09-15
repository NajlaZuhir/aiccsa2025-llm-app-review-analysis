"""
ABSA Prompts few-shot prompts. 
Each prompt is designed for a specific subtask in the ABSA pipeline.
"""

# ===========================
# Aspect Extraction Prompt :  Extracts concrete feature/function nouns (aspects) from a review sentence.
# ===========================
ASPECT_PROMPT = """
You are an expert aspect extractor for app reviews. Extract ALL meaningful features, functions, or characteristics mentioned in ONE sentence.

=== Rules ===
1) Extract ANY meaningful product aspects:
   - Features ("dark mode", "notifications")
   - Functions ("syncing", "sharing")
   - UI elements ("button", "menu")
   - Performance ("speed", "stability")
   - User experience ("usability", "design")
2) Convert verbs to nouns when needed:
   - "can't sync" → "syncing"
   - "hard to share" → "sharing"
3) Keep descriptive modifiers:
   - "quick search" (not just "search")
   - "offline mode" (not just "mode")
4) Include user experience aspects:
   - "fun", "ease-of-use", "convenience"
5) Use hyphens for multi-word terms
6) Keep ALL relevant aspects, even similar ones

=== Examples ===
Review: "The app is super slow and keeps crashing when I try to upload photos."
Aspects: ["app-performance", "app-stability", "photo-upload"]

Review: "Love the dark mode but the notifications are too frequent."
Aspects: ["dark-mode", "notification-frequency"]

Review: "Can't sync between my phone and laptop, very frustrating experience."
Aspects: ["device-syncing", "user-experience"]

Review: "The game is fun but too many ads."
Aspects: ["gameplay-enjoyment", "ad-frequency"]

=== Now you ===
Sentence: "{sentence}"

Return ONLY valid JSON (array of strings), no prose, no markdown:
[
  "aspect1",
  "aspect2"
]
""".strip()

# ===========================
# Sentiment Classification Prompt: Classifies the sentiment (Positive, Negative, Neutral) toward each extracted aspect.
# ===========================
SENTIMENT_PROMPT = """
You are an expert sentiment analyzer for app reviews. For each aspect in a sentence, determine the user's sentiment (Positive/Negative/Neutral).

Rules:
1) Classify EVERY listed aspect that's mentioned
2) Look for sentiment indicators:
   - Direct: "good", "bad", "love", "hate"
   - Indirect: "keeps crashing" (→ Negative)
   - Implied: "needs better X" (→ Negative for X)
3) Default to Neutral only if truly ambiguous
4) Consider context and tone
5) Include aspects even with mild sentiment

Examples:
Sentence: "App is fast but crashes a lot when uploading."
Aspects: ["app-speed", "app-stability", "file-upload"]
Output: {
    "app-speed": "Positive",
    "app-stability": "Negative",
    "file-upload": "Negative"
}

Sentence: "Dark mode works great, wish the font was bigger though."
Aspects: ["dark-mode", "font-size"]
Output: {
    "dark-mode": "Positive",
    "font-size": "Negative"
}

Now you:
Sentence: "{sentence}"
Aspects (JSON array): {aspects}

Return ONLY valid JSON (object mapping aspect → sentiment), no prose, no markdown:
{
  "aspect1": "Positive",
  "aspect2": "Negative"
}
""".strip()

# ===========================
# Recommendation Extraction Prompt: Extracts actionable, brief product/UX recommendations from a review sentence.
# ===========================
RECO_PROMPT = """
You are an expert product recommendations extractor. Convert user feedback into clear, actionable improvements.

Rules:
1) Extract BOTH:
   - Explicit requests ("Please add X")
   - Implied needs ("X doesn't work" → "Fix X")
2) Make recommendations:
   - Specific & actionable
   - Brief (5-10 words)
   - Start with verb
3) Include recommendations for:
   - Bug fixes
   - Feature requests
   - UX improvements
   - Performance issues
4) Convert complaints to solutions

Examples:
Review: "App keeps crashing when I try to upload photos."
Recommendations: ["Fix app crashes during photo uploads"]

Review: "Would be nice to have dark mode and better search."
Recommendations: ["Add dark mode feature", "Improve search functionality"]

Review: "Can't sync between devices and notifications are delayed."
Recommendations: ["Fix device synchronization issues", "Improve notification delivery speed"]

Review: "Great app but loading takes forever."
Recommendations: ["Optimize app loading performance"]

Now you:
Sentence: "{sentence}"

Return ONLY valid JSON (array of strings), no prose, no markdown:
[
  "recommendation 1",
  "recommendation 2"
]
""".strip()
