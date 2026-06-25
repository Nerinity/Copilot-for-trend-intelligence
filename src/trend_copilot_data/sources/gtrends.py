from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from ..storage import now_utc

log = logging.getLogger(__name__)


def _trend_req():
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.warning("pytrends is not installed. Run: python -m pip install -e .")
        return None
    return TrendReq(
        hl="en-US",
        tz=300,
        timeout=(10, 30),
        requests_args={"headers": {"User-Agent": "Mozilla/5.0"}},
    )


def _infer_category(keyword: str, category_keywords: dict[str, list[str]]) -> str:
    text = keyword.lower()
    best = "other"
    hits = 0
    for category, terms in category_keywords.items():
        score = sum(1 for term in terms if term.lower() in text)
        if score > hits:
            best = category
            hits = score
    return best


def _batched(values: list[str], size: int = 5):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def collect(
    config: dict[str, Any],
    run_id: str,
    start_date: str,
    end_date: str,
    seed_keywords: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    pytrends = _trend_req()
    if pytrends is None:
        return {
            "gtrends_timeseries": pd.DataFrame(),
            "gtrends_related_queries": pd.DataFrame(),
            "gtrends_category_summary": pd.DataFrame(),
        }

    cfg = config.get("google_trends", {})
    category_keywords = cfg.get("category_keywords", {})
    keywords = []
    for values in category_keywords.values():
        keywords.extend(values)
    keywords.extend(seed_keywords or [])
    keywords = list(dict.fromkeys([k.strip() for k in keywords if k and len(k.strip()) > 2]))
    keywords = keywords[: int(cfg.get("max_keywords", 180))]
    timeframe = f"{start_date} {end_date}"
    geo = cfg.get("geo", "US")

    timeseries_rows = []
    related_rows = []
    batch_size = int(cfg.get("batch_size", 1))
    for batch in _batched(keywords, batch_size):
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            interest = pytrends.interest_over_time()
            if not interest.empty:
                for kw in batch:
                    if kw not in interest.columns:
                        continue
                    series = interest[kw].dropna()
                    if series.empty:
                        continue
                    n = len(series)
                    early = float(series.iloc[: max(1, n // 3)].mean())
                    recent = float(series.iloc[max(0, n - max(2, n // 3)) :].mean())
                    growth = ((recent - early) / max(early, 1)) * 100
                    timeseries_rows.append(
                        {
                            "run_id": run_id,
                            "source": "google_trends",
                            "keyword": kw,
                            "category_hint": _infer_category(kw, category_keywords),
                            "timeframe": timeframe,
                            "geo": geo,
                            "early_score": round(early, 2),
                            "recent_score": round(recent, 2),
                            "peak_score": round(float(series.max()), 2),
                            "growth_pct": round(growth, 2),
                            "is_rising": growth >= float(cfg.get("rising_threshold_pct", 20)),
                            "collected_at": now_utc(),
                        }
                    )
            related = pytrends.related_queries()
            for seed, payload in related.items():
                if not payload:
                    continue
                for signal_type in ["rising", "top"]:
                    df = payload.get(signal_type)
                    if df is None or df.empty:
                        continue
                    for _, row in df.head(20).iterrows():
                        kw = str(row.get("query", "")).strip()
                        related_rows.append(
                            {
                                "run_id": run_id,
                                "source": "google_trends",
                                "seed_keyword": seed,
                                "keyword": kw,
                                "value": row.get("value", 0),
                                "signal_type": signal_type,
                                "is_breakout": str(row.get("value", "")).lower() == "breakout",
                                "category_hint": _infer_category(kw, category_keywords),
                                "collected_at": now_utc(),
                            }
                        )
            time.sleep(float(cfg.get("request_delay_seconds", 3)))
        except Exception as exc:
            log.warning("Google Trends failed for %s: %s", batch, exc)
            time.sleep(8)

    timeseries_df = pd.DataFrame(timeseries_rows)
    related_df = pd.DataFrame(related_rows)
    summary_df = _summarize(timeseries_df, related_df)
    return {
        "gtrends_timeseries": timeseries_df,
        "gtrends_related_queries": related_df,
        "gtrends_category_summary": summary_df,
    }


def _summarize(timeseries_df: pd.DataFrame, related_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = set()
    if not timeseries_df.empty:
        categories.update(timeseries_df["category_hint"].dropna().unique())
    if not related_df.empty:
        categories.update(related_df["category_hint"].dropna().unique())
    for category in categories:
        if category == "other":
            continue
        ts = timeseries_df[timeseries_df["category_hint"] == category] if not timeseries_df.empty else pd.DataFrame()
        rq = related_df[related_df["category_hint"] == category] if not related_df.empty else pd.DataFrame()
        rows.append(
            {
                "category_hint": category,
                "trend_score": round(
                    (ts["growth_pct"].clip(lower=0).sum() if not ts.empty else 0)
                    + (len(rq) * 5 if not rq.empty else 0)
                    + (rq["is_breakout"].sum() * 20 if not rq.empty else 0),
                    2,
                ),
                "keyword_count": int(len(ts) + len(rq)),
                "rising_keyword_count": int(ts["is_rising"].sum()) if not ts.empty else 0,
                "breakout_count": int(rq["is_breakout"].sum()) if not rq.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("trend_score", ascending=False) if rows else pd.DataFrame()
