from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from ..preprocess import clean_text, stable_hash
from ..settings import Settings
from ..storage import now_utc

log = logging.getLogger(__name__)

MENTION_COLUMNS = [
    "mention_id",
    "source",
    "platform",
    "source_type",
    "keyword",
    "query",
    "category",
    "title",
    "text",
    "full_text",
    "author",
    "community",
    "published_at",
    "collected_at",
    "url",
    "engagement_score",
    "metrics_json",
]


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _date_to_gdelt(value: str, end: bool = False) -> str:
    return value.replace("-", "") + ("235959" if end else "000000")


def build_queries(config: dict[str, Any], taxonomy_terms: list[str] | None = None) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    for category, keywords in config.get("google_trends", {}).get("category_keywords", {}).items():
        for keyword in keywords:
            queries.append({"keyword": keyword, "query": keyword, "category": category})
    for category, keywords in config.get("twitter", {}).get("category_keywords", {}).items():
        for keyword in keywords:
            queries.append({"keyword": keyword, "query": keyword, "category": category})
    for query in config.get("twitter", {}).get("seed_queries", []) + config.get("reddit", {}).get("search_queries", []):
        cleaned = query.replace('"', "").replace("OR", " ").strip()
        if cleaned:
            queries.append({"keyword": cleaned[:80], "query": cleaned, "category": "seed_signal"})
    for term in (taxonomy_terms or [])[: int(config.get("dashboard_mentions", {}).get("taxonomy_query_limit", 80))]:
        if 2 <= len(term) <= 80:
            queries.append({"keyword": term, "query": term, "category": "product_taxonomy"})

    seen = set()
    out = []
    for item in queries:
        key = clean_text(item["query"])
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[: int(config.get("dashboard_mentions", {}).get("max_queries", 120))]


def _record(
    *,
    source: str,
    platform: str,
    source_type: str,
    keyword: str,
    query: str,
    category: str,
    title: str,
    text: str,
    author: str,
    community: str,
    published_at: str,
    url: str,
    engagement_score: float = 0,
    metrics: dict | None = None,
) -> dict[str, Any]:
    title = _strip_html(title)
    text = _strip_html(text)
    mention_id = stable_hash(source, platform, keyword, title, text, url)
    return {
        "mention_id": mention_id,
        "source": source,
        "platform": platform,
        "source_type": source_type,
        "keyword": keyword,
        "query": query,
        "category": category,
        "title": title,
        "text": text,
        "full_text": clean_text(f"{title} {text}"),
        "author": author,
        "community": community,
        "published_at": published_at,
        "collected_at": now_utc(),
        "url": url,
        "engagement_score": engagement_score,
        "metrics_json": json.dumps(metrics or {}, ensure_ascii=False),
    }


def collect_google_news_rss(
    query_items: list[dict[str, str]],
    settings: Settings,
    start_date: str,
    end_date: str,
    per_query: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in query_items:
        search = f'("{item["query"]}") (trend OR viral OR review OR product OR TikTok OR Amazon) after:{start_date} before:{end_date}'
        url = "https://news.google.com/rss/search?q=" + urllib.parse.quote_plus(search) + "&hl=en-US&gl=US&ceid=US:en"
        try:
            response = requests.get(url, timeout=settings.request_timeout_seconds)
            if response.status_code != 200:
                log.warning("Google News RSS HTTP %s for %s", response.status_code, item["query"])
                continue
            root = ET.fromstring(response.content)
            for rss_item in root.findall(".//item")[:per_query]:
                records.append(
                    _record(
                        source="google_news_rss",
                        platform="news",
                        source_type="article_snippet",
                        keyword=item["keyword"],
                        query=item["query"],
                        category=item["category"],
                        title=rss_item.findtext("title") or "",
                        text=rss_item.findtext("description") or "",
                        author=rss_item.findtext("source") or "",
                        community=rss_item.findtext("source") or "",
                        published_at=rss_item.findtext("pubDate") or "",
                        url=rss_item.findtext("link") or "",
                    )
                )
            time.sleep(settings.min_request_delay_seconds)
        except Exception as exc:
            log.warning("Google News RSS failed for %s: %s", item["query"], exc)
    return records


def collect_gdelt(
    query_items: list[dict[str, str]],
    settings: Settings,
    start_date: str,
    end_date: str,
    per_query: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in query_items:
        params = {
            "query": f'"{item["query"]}" (trend OR viral OR review OR product OR shopping OR TikTok OR Amazon)',
            "mode": "ArtList",
            "format": "json",
            "maxrecords": min(per_query, 250),
            "sort": "HybridRel",
            "startdatetime": _date_to_gdelt(start_date),
            "enddatetime": _date_to_gdelt(end_date, end=True),
        }
        try:
            response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=settings.request_timeout_seconds)
            if response.status_code != 200:
                log.warning("GDELT HTTP %s for %s", response.status_code, item["query"])
                continue
            for article in response.json().get("articles", [])[:per_query]:
                records.append(
                    _record(
                        source="gdelt_doc",
                        platform="news_web",
                        source_type="article",
                        keyword=item["keyword"],
                        query=item["query"],
                        category=item["category"],
                        title=article.get("title", ""),
                        text=article.get("seendate", ""),
                        author=article.get("domain", ""),
                        community=article.get("sourceCountry", ""),
                        published_at=article.get("seendate", ""),
                        url=article.get("url", ""),
                        metrics={"language": article.get("language", ""), "domain": article.get("domain", "")},
                    )
                )
            time.sleep(settings.min_request_delay_seconds)
        except Exception as exc:
            log.warning("GDELT failed for %s: %s", item["query"], exc)
    return records


def collect_hackernews_mentions(
    query_items: list[dict[str, str]],
    settings: Settings,
    start_date: str,
    end_date: str,
    per_query: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp()) + 86399
    for item in query_items:
        params = {
            "query": item["query"],
            "tags": "story,comment",
            "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
            "hitsPerPage": per_query,
        }
        try:
            response = requests.get("https://hn.algolia.com/api/v1/search_by_date", params=params, timeout=settings.request_timeout_seconds)
            if response.status_code != 200:
                log.warning("HN HTTP %s for %s", response.status_code, item["query"])
                continue
            for hit in response.json().get("hits", [])[:per_query]:
                object_id = hit.get("objectID", "")
                records.append(
                    _record(
                        source="hackernews_algolia",
                        platform="forum",
                        source_type="comment" if "comment" in hit.get("_tags", []) else "story",
                        keyword=item["keyword"],
                        query=item["query"],
                        category=item["category"],
                        title=hit.get("title") or hit.get("story_title") or "",
                        text=hit.get("comment_text") or hit.get("story_text") or "",
                        author=hit.get("author", ""),
                        community="hackernews",
                        published_at=hit.get("created_at", ""),
                        url=hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
                        engagement_score=(hit.get("points") or 0) + (hit.get("num_comments") or 0) * 3,
                        metrics={"points": hit.get("points", 0), "num_comments": hit.get("num_comments", 0)},
                    )
                )
            time.sleep(settings.min_request_delay_seconds)
        except Exception as exc:
            log.warning("HN failed for %s: %s", item["query"], exc)
    return records


def collect(
    config: dict[str, Any],
    settings: Settings,
    start_date: str,
    end_date: str,
    taxonomy_terms: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config.get("dashboard_mentions", {})
    queries = build_queries(config, taxonomy_terms)
    log.info("Dashboard mentions: collecting %s queries", len(queries))
    records: list[dict[str, Any]] = []
    if cfg.get("google_news_enabled", True):
        records.extend(collect_google_news_rss(queries, settings, start_date, end_date, int(cfg.get("google_news_per_query", 40))))
    if cfg.get("gdelt_enabled", True):
        records.extend(collect_gdelt(queries, settings, start_date, end_date, int(cfg.get("gdelt_per_query", 40))))
    if cfg.get("hackernews_enabled", True):
        records.extend(collect_hackernews_mentions(queries, settings, start_date, end_date, int(cfg.get("hackernews_per_query", 30))))
    df = pd.DataFrame(records).reindex(columns=MENTION_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates("mention_id")
    return {"trend_mentions_raw": df}
