#!/usr/bin/env python3
"""
Build NLP & sentiment-analysis-ready dataset.

Takes all raw scraped data and the three product pool files, applies the
product-pool-driven NLP pipeline, and outputs a fully enriched CSV.

Output columns added on top of the standard mention schema:
  tokens                – JSON list of cleaned/stop-word-removed tokens
  token_count           – token count
  token_str             – space-joined tokens (ML input ready)
  product_relevance_score   – vocab-overlap score vs product pool
  tfidf_product_score       – cosine similarity to product TF-IDF corpus
  brand_match           – first matched brand from product pool
  theme_match           – first matched theme
  category_match        – matched category tokens
  has_product_mention   – boolean flag
  sentiment_compound    – VADER compound score (-1 to +1)
  sentiment_positive    – VADER positive probability
  sentiment_negative    – VADER negative probability
  sentiment_neutral     – VADER neutral probability
  sentiment_label       – "positive" / "negative" / "neutral"

Usage:
  python scripts/build_nlp_dataset.py
  python scripts/build_nlp_dataset.py \\
      --csv1 "/path/to/#1.1 Product Pool...Read Me.csv" \\
      --csv2 "/path/to/#1.2 Key Product Pool...Week69.csv" \\
      --xlsx "/path/to/直播货盘格式...xlsx" \\
      --input data/raw/scraped_2026_large.csv \\
      --output data/processed/nlp_sentiment_ready_2026.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

from trend_copilot_data.product_nlp import (
    build_product_nlp_pipeline,
    prepare_nlp_ready,
    STOP_WORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("build_nlp_dataset")

# Default product pool paths (the three internal files)
DEFAULT_CSV1 = Path("/Users/bytedance/Downloads/#1.1 Product Pool for Affiliate Creator_Internal - Read Me.csv")
DEFAULT_CSV2 = Path("/Users/bytedance/Downloads/[for Video SPM]#1.2 Key Product Pool for Top Creator_Internal - Week69(0601-0607).csv")
DEFAULT_XLSX = Path("/Users/bytedance/Desktop/直播货盘格式0527-2026-06-25 15-36-38.xlsx")

DEFAULT_INPUTS = [
    _ROOT / "data" / "raw" / "scraped_2026_large.csv",
    _ROOT / "data" / "raw" / "trend_mentions_raw.csv",
    _ROOT / "outputs" / "trend_mentions_raw.csv",
]
DEFAULT_OUTPUT = _ROOT / "data" / "processed" / "nlp_sentiment_ready_2026.csv"

CHUNK_SIZE = 5_000


def load_raw_data(input_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in input_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
                log.info("Loaded %d rows from %s", len(df), path.name)
                frames.append(df)
            except Exception as exc:
                log.warning("Failed to read %s: %s", path, exc)
    if not frames:
        log.error("No input data files found. Run scrape_large_2026.py first.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "mention_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["mention_id"])
    combined = combined.reset_index(drop=True)
    log.info("Total unique rows loaded: %d", len(combined))
    return combined


def run(
    csv1: Path | None,
    csv2: Path | None,
    xlsx: Path | None,
    input_paths: list[Path],
    output_path: Path,
    text_col: str = "full_text",
) -> None:
    # ── 1. Load product pool + build NLP pipeline ────────────────────────────
    log.info("Building product NLP pipeline from product pool files…")
    pipeline = build_product_nlp_pipeline(
        readme_csv=csv1,
        key_product_csv=csv2,
        live_xlsx=xlsx,
    )

    vocab_size = len(pipeline["entities"].get("all_vocab", set()))
    brands = len(pipeline["entities"].get("brand_set", set()))
    corpus_size = len(pipeline.get("product_corpus", []))
    log.info("Pipeline ready: vocab=%d tokens, %d brands, %d corpus docs, TF-IDF=%s",
             vocab_size, brands, corpus_size,
             "enabled" if pipeline.get("vectorizer") else "disabled (sklearn missing)")

    # ── 2. Load raw mentions ─────────────────────────────────────────────────
    raw_df = load_raw_data(input_paths)
    if raw_df.empty:
        log.error("No data to process.")
        return

    if text_col not in raw_df.columns:
        # Try to build full_text from title + text
        if "title" in raw_df.columns and "text" in raw_df.columns:
            raw_df["full_text"] = (
                raw_df["title"].fillna("").astype(str)
                + " "
                + raw_df["text"].fillna("").astype(str)
            ).str.strip()
        else:
            avail_text_cols = [c for c in raw_df.columns if "text" in c.lower()]
            if avail_text_cols:
                text_col = avail_text_cols[0]
                log.warning("Using column '%s' as text input", text_col)
            else:
                log.error("No text column found. Available: %s", list(raw_df.columns[:15]))
                return

    # ── 3. Process in chunks ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(raw_df)
    log.info("Processing %d rows in chunks of %d…", total, CHUNK_SIZE)

    first_chunk = True
    for start in range(0, total, CHUNK_SIZE):
        chunk = raw_df.iloc[start : start + CHUNK_SIZE].copy()
        enriched = prepare_nlp_ready(chunk, pipeline, text_col=text_col)
        enriched.to_csv(output_path, mode="w" if first_chunk else "a",
                        header=first_chunk, index=False)
        first_chunk = False
        pct = min(100, round((start + len(chunk)) / total * 100))
        log.info("  Processed %d/%d rows (%d%%)", start + len(chunk), total, pct)

    # ── 4. Summary stats ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Output: %s", output_path)
    try:
        result = pd.read_csv(output_path, low_memory=False)
        log.info("Total rows: %d", len(result))
        if "sentiment_label" in result.columns:
            dist = result["sentiment_label"].value_counts()
            log.info("Sentiment distribution:\n%s", dist.to_string())
        if "tfidf_product_score" in result.columns:
            log.info("Avg TF-IDF product score: %.4f", result["tfidf_product_score"].mean())
            log.info("Records with product match (score>0.05): %d (%.1f%%)",
                     (result["tfidf_product_score"] > 0.05).sum(),
                     (result["tfidf_product_score"] > 0.05).mean() * 100)
        if "brand_match" in result.columns:
            matched = result["brand_match"].ne("").sum()
            log.info("Records with brand match: %d (%.1f%%)", matched, matched / len(result) * 100)
    except Exception as exc:
        log.warning("Could not read output for summary: %s", exc)

    log.info("Done. NLP-ready dataset saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NLP & sentiment-ready dataset")
    parser.add_argument("--csv1", type=Path, default=DEFAULT_CSV1,
                        help="Path to #1.1 Product Pool Read Me CSV")
    parser.add_argument("--csv2", type=Path, default=DEFAULT_CSV2,
                        help="Path to #1.2 Key Product Pool CSV")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX,
                        help="Path to live product pool XLSX")
    parser.add_argument("--input", type=Path, nargs="+",
                        default=DEFAULT_INPUTS,
                        help="Raw mention CSV files to process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output path for NLP-ready CSV")
    parser.add_argument("--text-col", default="full_text",
                        help="Column name containing the text to process")
    args = parser.parse_args()

    csv1 = args.csv1 if args.csv1 and args.csv1.exists() else None
    csv2 = args.csv2 if args.csv2 and args.csv2.exists() else None
    xlsx = args.xlsx if args.xlsx and args.xlsx.exists() else None

    if not any([csv1, csv2, xlsx]):
        log.warning("No product pool files found at default paths. "
                    "NLP features will be minimal. "
                    "Specify paths via --csv1, --csv2, --xlsx.")

    run(
        csv1=csv1,
        csv2=csv2,
        xlsx=xlsx,
        input_paths=list(args.input),
        output_path=args.output,
        text_col=args.text_col,
    )


if __name__ == "__main__":
    main()
