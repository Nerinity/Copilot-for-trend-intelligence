from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

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
            best = category.replace("_", " ").title().replace("Ai", "AI")
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
            "gtrends_trending": pd.DataFrame(),
            "gtrends_timeseries": pd.DataFrame(),
            "gtrends_rising": pd.DataFrame(),
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
    trending_df = _fetch_trending_searches(pytrends, category_keywords)
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
                    current = float(series.iloc[max(0, n - 28) :].mean())
                    prev = float(series.iloc[max(0, n - 56) : max(1, n - 28)].mean())
                    early = float(series.iloc[: max(1, n // 4)].mean())
                    peak = float(series.max())
                    peak_week = str(series.idxmax().date()) if hasattr(series.idxmax(), "date") else ""
                    growth_recent = ((current - prev) / max(prev, 1)) * 100
                    growth_longterm = ((current - early) / max(early, 1)) * 100
                    if growth_recent > 20 and current > 30:
                        trend_shape = "accelerating"
                    elif growth_recent > 5:
                        trend_shape = "rising"
                    elif growth_recent > -5:
                        trend_shape = "stable"
                    elif current >= peak * 0.8:
                        trend_shape = "at_peak"
                    else:
                        trend_shape = "declining"
                    timeseries_rows.append(
                        {
                            "keyword": kw,
                            "category": _infer_category(kw, category_keywords),
                            "current_score": round(current, 2),
                            "prev_score": round(prev, 2),
                            "peak_score": round(peak, 2),
                            "peak_week": peak_week,
                            "growth_recent_pct": round(growth_recent, 2),
                            "growth_longterm_pct": round(growth_longterm, 2),
                            "trend_shape": trend_shape,
                            "is_rising": trend_shape in {"accelerating", "rising"},
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
                                "seed_keyword": seed,
                                "related_keyword": kw,
                                "rise_value": row.get("value", 0),
                                "is_breakout": str(row.get("value", "")).lower() == "breakout",
                                "category": _infer_category(kw, category_keywords),
                                "signal_type": f"related_{signal_type}",
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
        "gtrends_trending": trending_df,
        "gtrends_timeseries": timeseries_df,
        "gtrends_rising": related_df,
        "gtrends_category_summary": summary_df,
    }


def _fetch_trending_searches(pytrends: Any, category_keywords: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    try:
        df = pytrends.trending_searches(pn="united_states")
        for rank, row in df.iterrows():
            keyword = str(row.iloc[0]).strip()
            rows.append(
                {
                    "keyword": keyword,
                    "type": "realtime_trending",
                    "rank": rank + 1,
                    "category": _infer_category(keyword, category_keywords),
                    "fetched_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
                }
            )
    except Exception as exc:
        log.warning("Google Trends trending searches failed: %s", exc)
    try:
        today = pytrends.today_searches(pn="US")
        for keyword in today:
            keyword = str(keyword).strip()
            rows.append(
                {
                    "keyword": keyword,
                    "type": "today_search",
                    "rank": None,
                    "category": _infer_category(keyword, category_keywords),
                    "fetched_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
                }
            )
    except Exception as exc:
        log.warning("Google Trends today searches failed: %s", exc)
    return pd.DataFrame(rows).drop_duplicates("keyword") if rows else pd.DataFrame()


def _summarize(timeseries_df: pd.DataFrame, related_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categories = set()
    if not timeseries_df.empty:
        cat_col = "category" if "category" in timeseries_df.columns else "category_hint"
        categories.update(timeseries_df[cat_col].dropna().unique())
    if not related_df.empty:
        cat_col = "category" if "category" in related_df.columns else "category_hint"
        categories.update(related_df[cat_col].dropna().unique())
    for category in categories:
        if category == "other":
            continue
        ts_col = "category" if "category" in timeseries_df.columns else "category_hint"
        rq_col = "category" if "category" in related_df.columns else "category_hint"
        ts = timeseries_df[timeseries_df[ts_col] == category] if not timeseries_df.empty else pd.DataFrame()
        rq = related_df[related_df[rq_col] == category] if not related_df.empty else pd.DataFrame()
        growth_col = "growth_recent_pct" if "growth_recent_pct" in ts.columns else "growth_pct"
        rising_col = "is_rising"
        keyword_col = "keyword" if "keyword" in ts.columns else None
        related_col = "related_keyword" if "related_keyword" in rq.columns else "keyword"
        top_keywords = []
        if keyword_col and not ts.empty:
            top_keywords.extend(ts.sort_values(growth_col, ascending=False)[keyword_col].astype(str).head(3).tolist())
        if not rq.empty and related_col in rq.columns:
            top_keywords.extend(rq[related_col].astype(str).head(3).tolist())
        rows.append(
            {
                "category": category,
                "trend_score": round(
                    (ts[growth_col].clip(lower=0).sum() if not ts.empty else 0)
                    + (len(rq) * 5 if not rq.empty else 0)
                    + (rq["is_breakout"].sum() * 20 if not rq.empty else 0),
                    2,
                ),
                "keyword_count": int(len(ts) + len(rq)),
                "rising_keywords": int(ts[rising_col].sum()) if not ts.empty else 0,
                "top_keywords": ", ".join(list(dict.fromkeys(top_keywords))[:5]),
            }
        )
    return pd.DataFrame(rows).sort_values("trend_score", ascending=False) if rows else pd.DataFrame()
