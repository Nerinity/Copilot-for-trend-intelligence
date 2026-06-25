from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .preprocess import clean_text


CANONICAL_COLUMNS = [
    "first_category_name", "second_category_name", "third_category_name",
    "main_theme", "sub_theme_en", "scenarios_tags", "function_tags", "style_tags",
]


def load_product_taxonomy(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "first_category_name" not in raw.columns:
        header = raw.iloc[0].fillna("").tolist()
        raw = raw.iloc[1:].copy()
        raw.columns = header
    keep = [c for c in CANONICAL_COLUMNS if c in raw.columns]
    df = raw[keep].copy()
    for col in keep:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df[keep].apply(lambda row: any(bool(v) for v in row), axis=1)]
    return df.drop_duplicates().reset_index(drop=True)


def taxonomy_keywords(df: pd.DataFrame, limit: int = 500) -> list[str]:
    values: list[str] = []
    for col in [c for c in CANONICAL_COLUMNS if c in df.columns]:
        for value in df[col].dropna().astype(str):
            chunks = re.split(r"[,;/()|\[\]:]+", value)
            values.extend(clean_text(chunk).strip() for chunk in chunks)
    keywords = []
    seen = set()
    for value in values:
        if len(value) < 3 or value in seen:
            continue
        seen.add(value)
        keywords.append(value)
        if len(keywords) >= limit:
            break
    return keywords
