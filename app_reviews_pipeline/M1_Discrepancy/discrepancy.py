
# ===========================
# Imports & Configuration
# ===========================
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# --- VADER compound thresholds mapped to 1..5 stars
STAR_THRESHOLDS = (-0.6, -0.2, 0.2, 0.6)   # <=-0.6→1, <=-0.2→2, <=0.2→3, <=0.6→4, else 5
GENERATE_HEATMAP = True                    # set False if you only want the bar chart
TOP_PAD_FRAC = 0.15                        # extra headroom above the tallest bar (15%)

# --- wire up shared menus (choose_dataset / choose_sample_size)
THIS_ROOT = Path(__file__).resolve().parents[1]  # .../app_reviews_pipeline
if str(THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THIS_ROOT))
from user_selection import choose_dataset, choose_sample_size  # type: ignore

# ===========================
# Utility Functions
# ===========================

def vader_to_star(v: float) -> int:
    """Map VADER compound score to star rating (1-5)."""
    a, b, c, d = STAR_THRESHOLDS
    if v <= a: return 1
    if v <= b: return 2
    if v <= c: return 3
    if v <= d: return 4
    return 5

def _counts_1to5(series: pd.Series) -> np.ndarray:
    """Count occurrences for each star rating (1-5)."""
    vc = series.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int)
    return vc.values

def _annotate_counts(ax, x, heights, total, y_max):
    """Place count + percent slightly above each bar, padded from top."""
    offset = max(6, y_max * 0.02)  # consistent vertical offset
    for xi, h in zip(x, heights):
        pct = (h / max(total, 1)) * 100.0
        ax.text(
            xi, h + offset,
            f"{h}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9
        )

# ===========================
# Plotting Functions
# ===========================

def _make_bar_chart(ds: str,
                    actual_counts: np.ndarray,
                    vader_counts: np.ndarray,
                    total_n: int,
                    out_png: Path,
                    out_pdf: Path):
    """Create and save bar chart comparing actual and VADER ratings."""
    labels = np.array([1, 2, 3, 4, 5])
    x = np.arange(len(labels), dtype=float)
    width = 0.35

    # Pre-compute y-limit with headroom
    tallest = max(int(actual_counts.max()), int(vader_counts.max()))
    y_max = int(np.ceil(tallest * (1.0 + TOP_PAD_FRAC)))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars1 = ax.bar(x - width/2, actual_counts, width, label="Actual ratings")
    bars2 = ax.bar(x + width/2, vader_counts,  width, label="Sentiment ratings")

    ax.set_title(f"{ds.capitalize()} App Reviews: Actual vs. Sentiment Ratings (N={total_n})", pad=14)
    ax.set_xlabel("Rating bucket")
    ax.set_ylabel("Review count")
    ax.set_xticks(x, labels)

    # Apply padded y-limit and a little margin so labels never clip
    ax.set_ylim(0, y_max)
    ax.margins(y=0.02)
    ax.legend()

    # Annotate with counts + percentages (use common y_max for offset)
    _annotate_counts(ax, x - width/2, actual_counts, actual_counts.sum(), y_max)
    _annotate_counts(ax, x + width/2, vader_counts,  vader_counts.sum(), y_max)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def _make_heatmap(ds: str,
                  ctab: pd.DataFrame,
                  out_png: Path,
                  out_pdf: Path):
    """Create and save heatmap of actual vs. VADER star ratings."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(ctab.values, cmap="Blues")

    ax.set_title(f"{ds.capitalize()} — User Stars vs. VADER Stars (Counts)", pad=12)
    ax.set_xlabel("VADER stars")
    ax.set_ylabel("Actual stars")
    ax.set_xticks(range(5), [1,2,3,4,5])
    ax.set_yticks(range(5), [1,2,3,4,5])

    # Annotate each cell with count
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(ctab.values[i, j]),
                    ha="center", va="center",
                    fontsize=9, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ===========================
# Main Pipeline
# ===========================

def main():
    # 1) Pick dataset and load
    ds, csv_path = choose_dataset()
    df = pd.read_csv(csv_path)

    # Detect star column
    star_col = next((c for c in ["score", "rating", "stars", "user_rating"] if c in df.columns), None)
    if not star_col:
        raise ValueError(
            f"No star/rating column found in {csv_path}. "
            f"Expected one of: score/rating/stars/user_rating"
        )
    if "review_text" not in df.columns:
        raise ValueError("Expected a 'review_text' column in processed CSV.")

    # 2) Optional sampling
    n = choose_sample_size(len(df))
    if n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)

    # 3) VADER → compound → stars
    analyzer = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["review_text"].astype(str).map(lambda s: analyzer.polarity_scores(s)["compound"])
    df["vader_stars"] = df["vader_compound"].map(vader_to_star).astype(int)

    # Normalize user stars: numeric, clip 1..5
    actual_stars = pd.to_numeric(df[star_col], errors="coerce").round().clip(1, 5).astype("Int64")
    vader_stars = df["vader_stars"].astype("Int64")
    df["vader_stars"] = vader_stars
    df["star_discrepancy"] = (actual_stars - vader_stars).abs()

    # 4) Exports
    out_dir = Path(f"outputs/discrepancy/{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / f"{ds}_vader_discrepancy.csv"
    df_out = df.copy()
    df_out["actual_stars"] = actual_stars
    df_out.to_csv(rows_csv, index=False)

    summary = df_out["star_discrepancy"].value_counts().sort_index().rename_axis("abs_diff").reset_index(name="count")
    summary_csv = out_dir / f"{ds}_discrepancy_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # 5) Console quick stats
    mad = float(df_out["star_discrepancy"].mean())
    prop_ge1 = float((df_out["star_discrepancy"] >= 1).mean())
    prop_ge2 = float((df_out["star_discrepancy"] >= 2).mean())
    print(
        f"\nSummary ({ds}, N={len(df_out)}): "
        f"MAD={mad:.3f}, |Δ|≥1: {prop_ge1*100:.1f}%, |Δ|≥2: {prop_ge2*100:.1f}%"
    )

    # 6) Plot: bar chart counts + % (with padded y-limit)
    actual_counts = _counts_1to5(actual_stars.dropna().astype(int))
    vader_counts = _counts_1to5(vader_stars.dropna().astype(int))
    bar_png = out_dir / f"{ds}_user_vs_vader_bars.png"
    bar_pdf = out_dir / f"{ds}_user_vs_vader_bars.pdf"
    _make_bar_chart(ds, actual_counts, vader_counts, len(df_out), bar_png, bar_pdf)

    # 7) Optional heatmap (confusion counts)
    if GENERATE_HEATMAP:
        ctab = (
            pd.crosstab(actual_stars, vader_stars)
            .reindex(index=[1,2,3,4,5], columns=[1,2,3,4,5])
            .fillna(0)
            .astype(int)
        )
        heat_png = out_dir / f"{ds}_user_vs_vader_heatmap.png"
        heat_pdf = out_dir / f"{ds}_user_vs_vader_heatmap.pdf"
        _make_heatmap(ds, ctab, heat_png, heat_pdf)

    print("\nSaved:")
    print(f"  {rows_csv}")
    print(f"  {summary_csv}")
    print(f"  {bar_png}")
    if GENERATE_HEATMAP:
        print(f"  {heat_png}")

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    main()