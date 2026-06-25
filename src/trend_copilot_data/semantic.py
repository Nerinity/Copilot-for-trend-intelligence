from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .preprocess import clean_text


DEFAULT_BUSINESS_CONTEXTS = [
    {
        "label": "tiktok_creator_commerce",
        "text": (
            "TikTok Shop, TikTok made me buy it, creator affiliate, live shopping, "
            "viral product review, haul, unboxing, GRWM, UGC creator, commission, "
            "creator storefront, influencer product recommendation"
        ),
    },
    {
        "label": "consumer_product_discovery",
        "text": (
            "favorite product, holy grail, worth it, not worth it, honest review, "
            "dupe, Amazon finds, target finds, best purchase, product recommendation, "
            "restock, sold out, new launch"
        ),
    },
    {
        "label": "beauty_wellness_lifestyle",
        "text": (
            "skincare routine, beauty product, wellness supplement, protein coffee, "
            "magnesium, gut health, red light therapy, ice roller, hair care, "
            "self care, fitness recovery"
        ),
    },
    {
        "label": "home_food_family_pets",
        "text": (
            "home organization, cleaning hack, kitchen gadget, functional beverage, "
            "meal prep, baby product, pet product, dog enrichment, cat fountain, "
            "decor trend, cozy home"
        ),
    },
    {
        "label": "fashion_accessories_travel",
        "text": (
            "fashion trend, outfit, handbag, shoes, jewelry, quiet luxury, capsule "
            "wardrobe, travel gear, packing cubes, carry on, summer essentials"
        ),
    },
    {
        "label": "creator_business_signal",
        "text": (
            "brand deal, sponsorship, paid partnership, PR package, gifted product, "
            "affiliate conversion, product samples, content hook, social commerce, "
            "ecommerce, Shopify, Amazon seller"
        ),
    },
]

TIKTOK_PATTERNS = [
    r"\btiktok\b",
    r"\btik tok\b",
    r"\btiktokshop\b",
    r"\btiktok shop\b",
    r"\btiktok made me buy it\b",
    r"\btt shop\b",
    r"\bfyp\b",
    r"\bfor you page\b",
    r"\bcreator marketplace\b",
    r"\bcreator affiliate\b",
    r"\blive shopping\b",
    r"\blive selling\b",
    r"\bugc creator\b",
    r"\bgrwm\b",
    r"\bhauls?\b",
    r"\bunboxing\b",
]


@dataclass(frozen=True)
class SemanticConfig:
    min_relevance_score: float = 0.018
    keep_tiktok_score: float = 0.18
    tiktok_priority_boost: float = 0.25
    keep_all: bool = False


def load_semantic_config(config: dict[str, Any]) -> SemanticConfig:
    raw = config.get("semantic_filter", {})
    return SemanticConfig(
        min_relevance_score=float(raw.get("min_relevance_score", 0.018)),
        keep_tiktok_score=float(raw.get("keep_tiktok_score", 0.18)),
        tiktok_priority_boost=float(raw.get("tiktok_priority_boost", 0.25)),
        keep_all=bool(raw.get("keep_all", False)),
    )


def load_business_contexts(config: dict[str, Any]) -> list[dict[str, str]]:
    contexts = config.get("semantic_filter", {}).get("business_contexts") or DEFAULT_BUSINESS_CONTEXTS
    clean_contexts = []
    for item in contexts:
        label = str(item.get("label", "")).strip() or "business_context"
        text = str(item.get("text", "")).strip()
        if text:
            clean_contexts.append({"label": label, "text": text})
    return clean_contexts or DEFAULT_BUSINESS_CONTEXTS


def tiktok_relevance(text: str, source: str = "", sub_source: str = "", query: str = "") -> float:
    haystack = clean_text(" ".join([text or "", source or "", sub_source or "", query or ""]))
    if not haystack:
        return 0.0
    hits = sum(1 for pattern in TIKTOK_PATTERNS if re.search(pattern, haystack))
    return min(1.0, hits / 4)


def _safe_log_engagement(value: Any) -> float:
    try:
        number = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, math.log1p(number) / math.log1p(10000))


def score_mentions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        for column in [
            "semantic_relevance_score",
            "tiktok_relevance_score",
            "business_context_label",
            "collector_priority_score",
        ]:
            if column not in df.columns:
                df[column] = []
        return df

    semantic_cfg = load_semantic_config(config)
    contexts = load_business_contexts(config)
    vectorizer = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    )

    texts = (
        df.get("full_text", pd.Series([""] * len(df))).fillna("").astype(str).map(clean_text).tolist()
    )
    context_texts = [clean_text(item["text"]) for item in contexts]
    text_matrix = vectorizer.transform(texts)
    context_matrix = vectorizer.transform(context_texts)
    similarity = text_matrix @ context_matrix.T

    labels = []
    semantic_scores = []
    for row_idx in range(similarity.shape[0]):
        row = similarity.getrow(row_idx)
        if row.nnz == 0:
            labels.append("")
            semantic_scores.append(0.0)
            continue
        best_pos = int(row.indices[row.data.argmax()])
        labels.append(contexts[best_pos]["label"])
        semantic_scores.append(float(row.data.max()))

    out = df.copy()
    out["semantic_relevance_score"] = semantic_scores
    out["business_context_label"] = labels
    out["tiktok_relevance_score"] = [
        tiktok_relevance(text, source, sub_source, query)
        for text, source, sub_source, query in zip(
            texts,
            out.get("source", pd.Series([""] * len(out))).fillna("").astype(str),
            out.get("sub_source", pd.Series([""] * len(out))).fillna("").astype(str),
            out.get("query", pd.Series([""] * len(out))).fillna("").astype(str),
        )
    ]
    engagement_scores = [
        _safe_log_engagement(value)
        for value in out.get("engagement_score", pd.Series([0] * len(out))).tolist()
    ]
    out["collector_priority_score"] = [
        round((semantic * 0.62) + (tiktok * semantic_cfg.tiktok_priority_boost) + (engagement * 0.13), 6)
        for semantic, tiktok, engagement in zip(
            out["semantic_relevance_score"], out["tiktok_relevance_score"], engagement_scores
        )
    ]

    if not semantic_cfg.keep_all:
        keep_mask = (
            (out["semantic_relevance_score"] >= semantic_cfg.min_relevance_score)
            | (out["tiktok_relevance_score"] >= semantic_cfg.keep_tiktok_score)
            | (out["collector_priority_score"] >= semantic_cfg.min_relevance_score)
        )
        out = out.loc[keep_mask].copy()

    return out.sort_values("collector_priority_score", ascending=False)


def context_query_terms(config: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in load_business_contexts(config):
        raw_terms = re.split(r"[,;/]", item["text"])
        for term in raw_terms:
            cleaned = clean_text(term)
            if 3 <= len(cleaned) <= 80:
                terms.append(cleaned)

    extra = config.get("semantic_filter", {}).get("seed_queries", [])
    terms.extend(str(item).strip() for item in extra if str(item).strip())

    seen = set()
    out = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def contexts_to_json(config: dict[str, Any]) -> str:
    return json.dumps(load_business_contexts(config), ensure_ascii=False)
