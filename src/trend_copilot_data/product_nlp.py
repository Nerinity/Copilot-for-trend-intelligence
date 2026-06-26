"""
Product-pool NLP pipeline.

Replaces the generic HashingVectorizer embedding with TF-IDF trained on
the three internal product pool files (product names, brands, themes, tags).
Also adds proper tokenization, stop-word removal, and VADER sentiment.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _HAS_NLTK = True
except Exception:  # pragma: no cover
    _HAS_NLTK = False
    _VADER = None

# ── Extended stop words ──────────────────────────────────────────────────────
STOP_WORDS: frozenset[str] = frozenset({
    # Articles / prepositions / conjunctions
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need",
    # Pronouns
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "him", "her", "his", "i", "my", "me",
    # Adverbs / function words
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "more", "most", "other", "some", "any", "all", "few", "many", "much",
    "such", "own", "same", "than", "too", "very", "just", "because", "if",
    "while", "although", "though", "since", "unless", "until", "when",
    "where", "which", "who", "whom", "whose", "what", "how", "why", "then",
    "here", "there", "out", "up", "down", "into", "off", "over", "under",
    "again", "further", "once", "only", "also", "about", "above", "after",
    "before", "between", "during", "through", "within", "without",
    # Common verbs
    "like", "get", "got", "getting", "use", "used", "using", "make", "made",
    "know", "think", "thought", "people", "thing", "things", "really", "want",
    "wanted", "needs", "need", "now", "well", "back", "see", "look",
    "come", "go", "take", "give", "keep", "let", "put", "set", "try", "work",
    "say", "said", "tell", "told", "ask", "asked", "seem", "feel", "find",
    # Social media noise
    "reddit", "post", "comment", "thread", "update", "edit", "deleted", "removed",
    "mod", "upvote", "downvote", "karma", "tldr", "flair", "op", "oc", "dm",
    "share", "click", "link", "url", "www", "com", "http", "https",
    # E-commerce generic filler
    "buy", "sell", "sold", "price", "cheap", "free", "shipping", "order",
    "delivery", "return", "refund", "item", "review", "reviews",
    "good", "great", "bad", "best", "worst", "amazing", "awesome",
    "new", "old", "first", "last", "one", "two", "three", "time", "day", "week",
    "month", "year", "today", "yesterday", "always", "never", "every", "next",
    # Filler adjectives
    "nice", "beautiful", "pretty", "really", "super", "totally", "completely",
    "absolutely", "definitely", "probably", "maybe", "perhaps", "actually",
    "basically", "literally", "honestly", "seriously", "personally",
})

# Columns from the product pool files that carry signal
_PRODUCT_TEXT_COLS = [
    "product_name", "shop_name", "main_theme", "sub_theme_en",
    "scenarios_tags", "function_tags", "style_tags",
    "lvl1_product_industry", "lvl2_product_industry",
    "first_category_name", "second_category_name", "third_category_name",
]

# Columns that are specifically brand / identity tokens (higher weight)
_BRAND_COLS = ["shop_name"]
_PRODUCT_NAME_COLS = ["product_name"]
_THEME_COLS = ["main_theme", "sub_theme_en"]
_CATEGORY_COLS = ["lvl1_product_industry", "lvl2_product_industry",
                  "first_category_name", "second_category_name", "third_category_name"]
_TAG_COLS = ["scenarios_tags", "function_tags", "style_tags"]


# ── Tokenizer ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Clean and tokenize text, remove stop words, return alphabetic tokens."""
    if not text:
        return []
    text = re.sub(r"https?://\S+", " ", str(text))
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = text.lower()
    tokens = re.findall(r"\b[a-z][a-z]{1,}\b", text)  # min length 2
    return [t for t in tokens if t not in STOP_WORDS]


def tokenize_to_str(text: str) -> str:
    """Tokenize and join as space-separated string (for ML input)."""
    return " ".join(tokenize(text))


# ── Product pool loading ─────────────────────────────────────────────────────

def _safe_str_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str).str.strip()
    return pd.Series([""] * len(df))


def load_product_pool_csv(path: str | Path, skip_header_rows: int = 0) -> pd.DataFrame:
    """Load a product pool CSV, handling multi-row headers."""
    raw = pd.read_csv(path, skiprows=skip_header_rows, low_memory=False)
    # If the first data row looks like a header (non-numeric in id column), skip it
    if "product_id" not in raw.columns:
        # Try reading with row 1 as header
        raw = pd.read_csv(path, header=1, low_memory=False)
    keep = [c for c in _PRODUCT_TEXT_COLS if c in raw.columns]
    if not keep:
        log.warning("No expected columns found in %s. Available: %s", path, list(raw.columns[:10]))
        return pd.DataFrame()
    df = raw[keep].copy()
    for col in keep:
        df[col] = df[col].fillna("").astype(str).str.strip()
    # Drop rows where all kept cols are empty
    df = df[df.apply(lambda r: any(bool(v) for v in r), axis=1)]
    return df.reset_index(drop=True)


def load_product_pool_xlsx(path: str | Path) -> pd.DataFrame:
    """Load a product pool XLSX file."""
    try:
        raw = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        log.error("Failed to read XLSX %s: %s", path, exc)
        return pd.DataFrame()
    keep = [c for c in _PRODUCT_TEXT_COLS if c in raw.columns]
    if not keep:
        log.warning("No expected columns found in XLSX %s. Available: %s", path, list(raw.columns[:10]))
        return pd.DataFrame()
    df = raw[keep].copy()
    for col in keep:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df.apply(lambda r: any(bool(v) for v in r), axis=1)]
    return df.reset_index(drop=True)


def load_all_product_pools(
    readme_csv: str | Path | None = None,
    key_product_csv: str | Path | None = None,
    live_xlsx: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load and merge all three product pool files.

    - readme_csv: #1.1 Product Pool for Affiliate Creator Read Me.csv (rules/taxonomy)
    - key_product_csv: #1.2 Key Product Pool for Top Creator Week69.csv (main data)
    - live_xlsx: 直播货盘格式 .xlsx (live streaming pool)
    """
    frames: list[pd.DataFrame] = []

    if key_product_csv and Path(key_product_csv).exists():
        df = load_product_pool_csv(key_product_csv, skip_header_rows=1)
        if not df.empty:
            df["source_file"] = "key_product_pool"
            frames.append(df)
            log.info("Loaded %d products from key_product_csv", len(df))

    if live_xlsx and Path(live_xlsx).exists():
        df = load_product_pool_xlsx(live_xlsx)
        if not df.empty:
            df["source_file"] = "live_product_pool"
            frames.append(df)
            log.info("Loaded %d products from live_xlsx", len(df))

    if readme_csv and Path(readme_csv).exists():
        # This file is mainly rules; try to extract category names from it
        try:
            raw = pd.read_csv(readme_csv, low_memory=False)
            cats = []
            for col in raw.columns:
                for val in raw[col].dropna().astype(str):
                    if 3 <= len(val.strip()) <= 80 and val.strip().replace(" ", "").isalpha():
                        cats.append(val.strip())
            if cats:
                rule_df = pd.DataFrame({"first_category_name": list(dict.fromkeys(cats))})
                rule_df["source_file"] = "readme_rules"
                frames.append(rule_df)
                log.info("Extracted %d category hints from readme_csv", len(rule_df))
        except Exception as exc:
            log.warning("Could not parse readme_csv %s: %s", readme_csv, exc)

    if not frames:
        log.warning("No product pool files loaded.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in _PRODUCT_TEXT_COLS:
        if col not in combined.columns:
            combined[col] = ""
    return combined.reset_index(drop=True)


# ── Entity extraction ────────────────────────────────────────────────────────

def _split_tags(text: str) -> list[str]:
    """Split comma/semicolon/pipe delimited tag strings."""
    parts = re.split(r"[,;|/\(\)\[\]]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_product_entities(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extract product vocabulary entities from product pool DataFrame.

    Returns a dict with:
      brand_set: set of full brand name strings (lowercase)
      brand_tokens: set of individual brand tokens
      product_name_tokens: set of tokens from product names
      theme_set: set of full theme strings
      theme_tokens: set of tokens from themes
      category_tokens: set of category tokens
      tag_tokens: set of tag tokens
      all_vocab: union of all above
      product_corpus: list of per-product text strings (for TF-IDF)
    """
    brand_set: set[str] = set()
    brand_tokens: set[str] = set()
    product_name_tokens: set[str] = set()
    theme_set: set[str] = set()
    theme_tokens: set[str] = set()
    category_tokens: set[str] = set()
    tag_tokens: set[str] = set()
    product_corpus: list[str] = []

    for _, row in df.iterrows():
        # Brand / shop
        brand_raw = str(row.get("shop_name", "") or "").strip()
        if brand_raw and brand_raw not in ("nan", "NULL", ""):
            brand_set.add(brand_raw.lower())
            brand_tokens.update(tokenize(brand_raw))

        # Product name
        pname = str(row.get("product_name", "") or "").strip()
        if pname and pname not in ("nan", "NULL", ""):
            product_name_tokens.update(tokenize(pname))

        # Themes
        for col in _THEME_COLS:
            theme_raw = str(row.get(col, "") or "").strip()
            if theme_raw and theme_raw not in ("nan", "NULL", ""):
                theme_set.add(theme_raw.lower()[:120])
                for part in _split_tags(theme_raw):
                    theme_tokens.update(tokenize(part))

        # Categories
        for col in _CATEGORY_COLS:
            cat_raw = str(row.get(col, "") or "").strip()
            if cat_raw and cat_raw not in ("nan", "NULL", ""):
                for part in _split_tags(cat_raw):
                    category_tokens.update(tokenize(part))

        # Tags
        for col in _TAG_COLS:
            tag_raw = str(row.get(col, "") or "").strip()
            if tag_raw and tag_raw not in ("nan", "NULL", ""):
                for part in _split_tags(tag_raw):
                    tag_tokens.update(tokenize(part))

        # Build per-product corpus text (all fields concatenated, higher weight for name/brand)
        parts = []
        # Repeat product name / brand 3x for higher TF-IDF weight
        for col in _PRODUCT_NAME_COLS + _BRAND_COLS:
            val = str(row.get(col, "") or "").strip()
            if val and val not in ("nan", "NULL", ""):
                parts.extend([tokenize_to_str(val)] * 3)
        for col in _THEME_COLS:
            val = str(row.get(col, "") or "").strip()
            if val and val not in ("nan", "NULL", ""):
                parts.extend([tokenize_to_str(val)] * 2)
        for col in _TAG_COLS + _CATEGORY_COLS:
            val = str(row.get(col, "") or "").strip()
            if val and val not in ("nan", "NULL", ""):
                parts.append(tokenize_to_str(val))
        corpus_text = " ".join(parts).strip()
        if corpus_text:
            product_corpus.append(corpus_text)

    # Remove generic/empty tokens
    for s in (brand_tokens, product_name_tokens, theme_tokens, category_tokens, tag_tokens):
        s.discard("")
        s.discard("nan")
        s.discard("null")

    all_vocab = (
        brand_tokens | product_name_tokens | theme_tokens | category_tokens | tag_tokens
    )
    # Remove single-char tokens
    all_vocab = {t for t in all_vocab if len(t) >= 2}

    return {
        "brand_set": brand_set,
        "brand_tokens": brand_tokens,
        "product_name_tokens": product_name_tokens,
        "theme_set": theme_set,
        "theme_tokens": theme_tokens,
        "category_tokens": category_tokens,
        "tag_tokens": tag_tokens,
        "all_vocab": all_vocab,
        "product_corpus": product_corpus,
    }


# ── TF-IDF vectorizer (product-pool driven) ───────────────────────────────

def build_product_tfidf(product_corpus: list[str]) -> "TfidfVectorizer | None":
    """
    Build TF-IDF vectorizer trained on the product pool corpus.

    This replaces the previous HashingVectorizer-on-hardcoded-contexts approach.
    Now the embedding space is defined by the actual products, brands, and themes
    from the internal product files.
    """
    if not _HAS_SKLEARN:
        log.warning("sklearn not available; TF-IDF product vectorizer disabled")
        return None
    if not product_corpus:
        log.warning("Empty product corpus; TF-IDF vectorizer not built")
        return None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=8000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r"\b[a-z][a-z0-9\-]{1,}\b",
    )
    vectorizer.fit(product_corpus)
    log.info(
        "Product TF-IDF vocabulary size: %d (from %d corpus docs)",
        len(vectorizer.vocabulary_),
        len(product_corpus),
    )
    return vectorizer


# ── Per-record scoring ───────────────────────────────────────────────────────

def _product_relevance_score(
    text_tokens: list[str],
    entities: dict[str, Any],
) -> float:
    """
    Compute product relevance as overlap ratio between text tokens
    and product vocabulary (brand/product/theme/tag tokens).
    """
    if not text_tokens:
        return 0.0
    vocab = entities.get("all_vocab", set())
    if not vocab:
        return 0.0
    hits = sum(1 for t in text_tokens if t in vocab)
    # Normalize by sqrt of text length to reward density, not just length
    return round(min(1.0, hits / max(1, len(text_tokens) ** 0.5) * 0.5), 6)


def _brand_match(text_lower: str, entities: dict[str, Any]) -> str:
    """Return the first matched brand name found in text, or empty string."""
    for brand in entities.get("brand_set", set()):
        if brand and len(brand) > 3 and brand in text_lower:
            return brand
    return ""


def _theme_match(text_tokens_set: set[str], entities: dict[str, Any]) -> str:
    """Return the first matched theme whose tokens are all present in text."""
    for theme in entities.get("theme_set", set()):
        theme_toks = set(tokenize(theme))
        if len(theme_toks) >= 2 and theme_toks.issubset(text_tokens_set):
            return theme[:80]
    return ""


def _category_match(text_tokens_set: set[str], entities: dict[str, Any]) -> str:
    """Return category tokens that appear in text."""
    hits = text_tokens_set & entities.get("category_tokens", set())
    return ", ".join(sorted(hits)[:5]) if hits else ""


# ── Sentiment analysis ───────────────────────────────────────────────────────

def compute_sentiment(text: str) -> dict[str, Any]:
    """
    Compute sentiment using VADER if available, else return neutral defaults.

    Returns: {compound, positive, negative, neutral, label}
    """
    if _HAS_NLTK and _VADER is not None and text:
        try:
            scores = _VADER.polarity_scores(str(text))
            compound = round(scores["compound"], 4)
            label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
            return {
                "sentiment_compound": compound,
                "sentiment_positive": round(scores["pos"], 4),
                "sentiment_negative": round(scores["neg"], 4),
                "sentiment_neutral": round(scores["neu"], 4),
                "sentiment_label": label,
            }
        except Exception:
            pass
    return {
        "sentiment_compound": 0.0,
        "sentiment_positive": 0.0,
        "sentiment_negative": 0.0,
        "sentiment_neutral": 1.0,
        "sentiment_label": "neutral",
    }


# ── TF-IDF product relevance via cosine similarity ───────────────────────────

def score_texts_against_product_corpus(
    texts: list[str],
    vectorizer: "TfidfVectorizer",
    product_corpus: list[str],
    batch_size: int = 500,
) -> list[float]:
    """
    Score each text against the product corpus by max cosine similarity.

    Returns a list of float scores in [0, 1].
    """
    if not _HAS_SKLEARN or vectorizer is None or not product_corpus:
        return [0.0] * len(texts)

    # Pre-compute product corpus matrix
    corpus_matrix = vectorizer.transform(product_corpus)
    scores: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        text_matrix = vectorizer.transform(batch)
        # cosine_similarity: (batch_size, n_products) -> take max per row
        sim = cosine_similarity(text_matrix, corpus_matrix)
        for row in sim:
            scores.append(float(row.max()) if row.size > 0 else 0.0)

    return scores


# ── Main NLP enrichment function ─────────────────────────────────────────────

def add_nlp_features(
    df: pd.DataFrame,
    entities: dict[str, Any],
    vectorizer: "TfidfVectorizer | None" = None,
    product_corpus: list[str] | None = None,
    text_col: str = "full_text",
) -> pd.DataFrame:
    """
    Add NLP and sentiment analysis columns to a mentions DataFrame.

    New columns added:
      tokens              – JSON list of cleaned tokens
      token_count         – number of tokens
      token_str           – space-joined tokens (ML-ready)
      product_relevance_score – vocab-overlap score vs product pool
      tfidf_product_score – cosine similarity to product corpus (if sklearn available)
      brand_match         – first detected brand name from product pool
      theme_match         – first detected theme from product pool
      category_match      – detected category tokens from product pool
      sentiment_compound  – VADER compound score (-1 to +1)
      sentiment_positive  – VADER positive probability
      sentiment_negative  – VADER negative probability
      sentiment_neutral   – VADER neutral probability
      sentiment_label     – "positive" / "negative" / "neutral"
      has_product_mention – bool: any product vocab token present
    """
    out = df.copy()
    raw_texts = out.get(text_col, pd.Series([""] * len(out))).fillna("").astype(str).tolist()

    # Tokenize all texts
    all_tokens: list[list[str]] = [tokenize(t) for t in raw_texts]
    all_token_sets: list[set[str]] = [set(tl) for tl in all_tokens]

    out["tokens"] = [json.dumps(tl, ensure_ascii=False) for tl in all_tokens]
    out["token_count"] = [len(tl) for tl in all_tokens]
    out["token_str"] = [" ".join(tl) for tl in all_tokens]

    # Product vocab overlap score
    out["product_relevance_score"] = [
        _product_relevance_score(tl, entities) for tl in all_tokens
    ]

    # TF-IDF cosine product score
    if vectorizer is not None and product_corpus:
        token_strs = [" ".join(tl) for tl in all_tokens]
        tfidf_scores = score_texts_against_product_corpus(token_strs, vectorizer, product_corpus)
        out["tfidf_product_score"] = [round(s, 6) for s in tfidf_scores]
    else:
        out["tfidf_product_score"] = out["product_relevance_score"]

    # Entity matching
    out["brand_match"] = [_brand_match(t.lower(), entities) for t in raw_texts]
    out["theme_match"] = [_theme_match(ts, entities) for ts in all_token_sets]
    out["category_match"] = [_category_match(ts, entities) for ts in all_token_sets]
    out["has_product_mention"] = out["product_relevance_score"].gt(0.05)

    # Sentiment
    sentiments = [compute_sentiment(t) for t in raw_texts]
    for key in ["sentiment_compound", "sentiment_positive", "sentiment_negative",
                "sentiment_neutral", "sentiment_label"]:
        out[key] = [s[key] for s in sentiments]

    return out


# ── Public high-level API ─────────────────────────────────────────────────────

def build_product_nlp_pipeline(
    readme_csv: str | Path | None = None,
    key_product_csv: str | Path | None = None,
    live_xlsx: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load product pool files and build the NLP pipeline components.

    Returns a dict with keys: df, entities, vectorizer, product_corpus
    suitable for passing to add_nlp_features().
    """
    product_df = load_all_product_pools(readme_csv, key_product_csv, live_xlsx)
    if product_df.empty:
        log.warning("Product pool is empty — NLP features will be minimal.")
        entities: dict[str, Any] = {
            "brand_set": set(), "brand_tokens": set(), "product_name_tokens": set(),
            "theme_set": set(), "theme_tokens": set(), "category_tokens": set(),
            "tag_tokens": set(), "all_vocab": set(), "product_corpus": [],
        }
        return {"df": product_df, "entities": entities, "vectorizer": None, "product_corpus": []}

    entities = extract_product_entities(product_df)
    log.info(
        "Product vocab built: %d brands, %d product-name tokens, "
        "%d theme tokens, %d category tokens, %d tag tokens → %d total vocab",
        len(entities["brand_set"]),
        len(entities["product_name_tokens"]),
        len(entities["theme_tokens"]),
        len(entities["category_tokens"]),
        len(entities["tag_tokens"]),
        len(entities["all_vocab"]),
    )

    vectorizer = build_product_tfidf(entities["product_corpus"])
    return {
        "df": product_df,
        "entities": entities,
        "vectorizer": vectorizer,
        "product_corpus": entities["product_corpus"],
    }


def prepare_nlp_ready(
    mentions_df: pd.DataFrame,
    pipeline: dict[str, Any],
    text_col: str = "full_text",
) -> pd.DataFrame:
    """
    Apply the full NLP pipeline to a mentions DataFrame and return an
    NLP + sentiment-analysis-ready dataset.
    """
    return add_nlp_features(
        df=mentions_df,
        entities=pipeline["entities"],
        vectorizer=pipeline.get("vectorizer"),
        product_corpus=pipeline.get("product_corpus"),
        text_col=text_col,
    )
