from __future__ import annotations

import json
import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from ..preprocess import clean_text, extract_keywords, stable_hash
from ..settings import Settings
from ..storage import normalize_records, now_utc

log = logging.getLogger(__name__)


def _dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def collect_hackernews(config: dict[str, Any], settings: Settings, run_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    cfg = config.get("public_sources", {}).get("hackernews", {})
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    tags = cfg.get("queries", [])
    records = []
    start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp()) + 86399
    for query in tags:
        params = {
            "query": query,
            "tags": "story,comment",
            "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
            "hitsPerPage": int(cfg.get("hits_per_query", 50)),
        }
        try:
            r = requests.get("https://hn.algolia.com/api/v1/search_by_date", params=params, timeout=settings.request_timeout_seconds)
            if r.status_code != 200:
                log.warning("HN HTTP %s for %s", r.status_code, query)
                continue
            for hit in r.json().get("hits", []):
                text = hit.get("story_text") or hit.get("comment_text") or hit.get("title") or ""
                title = hit.get("title") or hit.get("story_title") or ""
                object_id = hit.get("objectID", stable_hash(title, text))
                metrics = {"points": hit.get("points", 0), "num_comments": hit.get("num_comments", 0)}
                records.append(
                    {
                        "run_id": run_id,
                        "source": "hackernews",
                        "source_type": hit.get("_tags", ["story"])[0],
                        "source_id": f"hn_{object_id}",
                        "source_url": hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
                        "category_hint": "tech_startup_consumer_ai",
                        "community": "hackernews",
                        "query": query,
                        "title": title,
                        "text": text,
                        "clean_text": clean_text(f"{title} {text}"),
                        "author": hit.get("author", ""),
                        "published_at": hit.get("created_at", ""),
                        "collected_at": now_utc(),
                        "engagement_score": (hit.get("points") or 0) + (hit.get("num_comments") or 0) * 3,
                        "metrics_json": json.dumps(metrics),
                        "content_hash": stable_hash("hn", object_id, title, text),
                        "raw_json": json.dumps(hit, ensure_ascii=False)[:20000],
                    }
                )
            time.sleep(settings.min_request_delay_seconds)
        except Exception as exc:
            log.warning("HN failed for %s: %s", query, exc)
    return normalize_records(records)


def collect_rss(config: dict[str, Any], settings: Settings, run_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    cfg = config.get("public_sources", {}).get("rss", {})
    if not cfg.get("enabled", True):
        return pd.DataFrame()
    records = []
    queries = cfg.get("google_news_queries", [])
    for query in queries:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = requests.get(url, timeout=settings.request_timeout_seconds)
            if r.status_code != 200:
                log.warning("RSS HTTP %s for %s", r.status_code, query)
                continue
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[: int(cfg.get("items_per_query", 30))]:
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub = item.findtext("pubDate") or ""
                source = item.findtext("source") or ""
                text = item.findtext("description") or ""
                records.append(
                    {
                        "run_id": run_id,
                        "source": "google_news_rss",
                        "source_type": "news_result",
                        "source_id": f"rss_{stable_hash(link or title)}",
                        "source_url": link,
                        "category_hint": "market_news",
                        "community": source,
                        "query": query,
                        "title": title,
                        "text": text,
                        "clean_text": clean_text(f"{title} {text}"),
                        "author": source,
                        "published_at": pub,
                        "collected_at": now_utc(),
                        "engagement_score": 0,
                        "metrics_json": "{}",
                        "content_hash": stable_hash("rss", link, title),
                        "raw_json": "",
                    }
                )
            time.sleep(settings.min_request_delay_seconds)
        except Exception as exc:
            log.warning("RSS failed for %s: %s", query, exc)
    return normalize_records(records)


def collect(
    config: dict[str, Any],
    settings: Settings,
    run_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    hn_df = collect_hackernews(config, settings, run_id, start_date, end_date)
    rss_df = collect_rss(config, settings, run_id, start_date, end_date)
    texts = []
    if not hn_df.empty:
        texts.extend(hn_df["clean_text"].tolist())
    if not rss_df.empty:
        texts.extend(rss_df["clean_text"].tolist())
    keywords_df = pd.DataFrame(extract_keywords(texts, top_n=100))
    if not keywords_df.empty:
        keywords_df.insert(0, "run_id", run_id)
        keywords_df["source"] = "public_sources"
        keywords_df["collected_at"] = now_utc()
    return {"hackernews_posts": hn_df, "rss_market_news": rss_df, "public_keywords": keywords_df}
