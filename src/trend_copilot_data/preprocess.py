from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional at runtime
    TfidfVectorizer = None


STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has", "had",
    "are", "was", "were", "you", "your", "they", "them", "their", "our", "about",
    "just", "like", "really", "very", "more", "most", "some", "any", "all", "can",
    "get", "got", "getting", "use", "used", "using", "want", "wanted", "need",
    "needs", "make", "made", "think", "thought", "people", "thing", "things",
    "reddit", "post", "comment", "thread", "update", "edit",
}


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"https?://\S+", " ", str(text))
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9#&/\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def stable_hash(*parts: object) -> str:
    joined = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def extract_hashtags(text: str | None) -> list[str]:
    return [m.group(1).lower() for m in re.finditer(r"#([A-Za-z0-9_]{2,60})", text or "")]


def extract_keywords(texts: Iterable[str], top_n: int = 50) -> list[dict[str, object]]:
    cleaned = [clean_text(t) for t in texts if clean_text(t)]
    if len(cleaned) < 3:
        return []
    if TfidfVectorizer is None:
        counts = Counter()
        for text in cleaned:
            counts.update(t for t in text.split() if len(t) > 3 and t not in STOP_WORDS)
        return [{"keyword": k, "score": float(v), "method": "frequency"} for k, v in counts.most_common(top_n)]

    vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        min_df=2,
        max_df=0.88,
        stop_words=list(STOP_WORDS),
        token_pattern=r"\b[a-z][a-z0-9&/\-]{2,}\b",
    )
    matrix = vec.fit_transform(cleaned)
    names = vec.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    ranked = scores.argsort()[-top_n:][::-1]
    return [
        {"keyword": names[i], "score": round(float(scores[i]), 4), "method": "tfidf"}
        for i in ranked
        if names[i] not in STOP_WORDS
    ]
