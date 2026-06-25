from __future__ import annotations

import html
import json
import logging
import re
import shutil
import subprocess
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
    "sub_source",
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


def _date_to_unix(value: str, end: bool = False) -> int:
    suffix = "T23:59:59+00:00" if end else "T00:00:00+00:00"
    return int(datetime.fromisoformat(value + suffix).timestamp())


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
    sub_source: str = "",
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
        "sub_source": sub_source,
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
                        sub_source=rss_item.findtext("source") or "",
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
                        sub_source=article.get("domain", ""),
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
                        sub_source="hackernews",
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


def collect_producthunt_rss(
    query_items: list[dict[str, str]],
    settings: Settings,
    per_query: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        response = requests.get("https://www.producthunt.com/feed", timeout=settings.request_timeout_seconds)
        if response.status_code != 200:
            log.warning("Product Hunt RSS HTTP %s", response.status_code)
            return records
        root = ET.fromstring(response.content)
        items = root.findall(".//item")
    except Exception as exc:
        log.warning("Product Hunt RSS failed: %s", exc)
        return records

    for item in query_items:
        hits = 0
        needle = clean_text(item["query"])
        for rss_item in items:
            title = rss_item.findtext("title") or ""
            desc = rss_item.findtext("description") or ""
            full = clean_text(f"{title} {desc}")
            if needle not in full:
                continue
            records.append(
                _record(
                    source="producthunt_rss",
                    platform="product_launch",
                    sub_source="producthunt",
                    source_type="product_launch",
                    keyword=item["keyword"],
                    query=item["query"],
                    category=item["category"],
                    title=title,
                    text=desc,
                    author="Product Hunt",
                    community="producthunt",
                    published_at=rss_item.findtext("pubDate") or "",
                    url=rss_item.findtext("link") or "",
                )
            )
            hits += 1
            if hits >= per_query:
                break
    return records


def collect_reddit_pullpush(
    query_items: list[dict[str, str]],
    settings: Settings,
    start_date: str,
    end_date: str,
    per_query: int,
    include_comments: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    endpoints = [("submission", "https://api.pullpush.io/reddit/search/submission/")]
    if include_comments:
        endpoints.append(("comment", "https://api.pullpush.io/reddit/search/comment/"))
    start_ts = _date_to_unix(start_date)
    end_ts = _date_to_unix(end_date, end=True)
    headers = {"User-Agent": settings.reddit_user_agent}

    for item in query_items:
        for source_type, endpoint in endpoints:
            before = end_ts
            collected = 0
            pages = 0
            while collected < per_query and pages < 10:
                params = {
                    "q": item["query"],
                    "after": start_ts,
                    "before": before,
                    "size": min(100, per_query - collected),
                    "sort": "desc",
                    "sort_type": "created_utc",
                }
                try:
                    response = requests.get(endpoint, params=params, headers=headers, timeout=60)
                    if response.status_code == 429:
                        time.sleep(15)
                        continue
                    if response.status_code != 200:
                        log.warning("PullPush %s HTTP %s for %s", source_type, response.status_code, item["query"])
                        break
                    rows = response.json().get("data", [])
                except Exception as exc:
                    log.warning("PullPush %s failed for %s: %s", source_type, item["query"], exc)
                    break
                if not rows:
                    break
                for row in rows:
                    created = row.get("created_utc") or 0
                    published = datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else ""
                    subreddit = row.get("subreddit", "")
                    if source_type == "submission":
                        title = row.get("title", "")
                        text = row.get("selftext", "")
                        url = f"https://reddit.com{row.get('permalink', '')}"
                        engagement = (row.get("score") or 0) + (row.get("num_comments") or 0) * 3
                        metrics = {
                            "score": row.get("score", 0),
                            "num_comments": row.get("num_comments", 0),
                            "upvote_ratio": row.get("upvote_ratio", 0),
                        }
                    else:
                        title = row.get("link_title", "")
                        text = row.get("body", "")
                        url = f"https://reddit.com{row.get('permalink', '')}"
                        engagement = row.get("score") or 0
                        metrics = {"score": row.get("score", 0), "link_id": row.get("link_id", "")}
                    records.append(
                        _record(
                            source="reddit_pullpush",
                            platform="reddit",
                            sub_source=f"r/{subreddit}" if subreddit else "reddit",
                            source_type=source_type,
                            keyword=item["keyword"],
                            query=item["query"],
                            category=item["category"],
                            title=title,
                            text=text,
                            author=row.get("author", "[deleted]"),
                            community=subreddit,
                            published_at=published,
                            url=url,
                            engagement_score=engagement,
                            metrics=metrics,
                        )
                    )
                collected += len(rows)
                pages += 1
                before = min((r.get("created_utc") or before for r in rows), default=before)
                if len(rows) < params["size"]:
                    break
                time.sleep(settings.min_request_delay_seconds)
    return records


def collect_reddit_rss(
    config: dict[str, Any],
    query_items: list[dict[str, str]],
    settings: Settings,
    per_subreddit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    reddit_cfg = config.get("reddit", {})
    subreddits: list[str] = []
    for subs in reddit_cfg.get("subreddit_categories", {}).values():
        subreddits.extend(subs)
    subreddits = list(dict.fromkeys(subreddits))[: int(config.get("dashboard_mentions", {}).get("reddit_rss_max_subreddits", 160))]
    sorts = config.get("dashboard_mentions", {}).get("reddit_rss_sorts", ["hot", "new", "top?t=month", "top?t=year", "rising"])
    needles = [(item, clean_text(item["query"])) for item in query_items]

    for subreddit in subreddits:
        saved_for_sub = 0
        for sort in sorts:
            if "?" in sort:
                url = f"https://www.reddit.com/r/{subreddit}/{sort.replace('?', '.rss?')}&limit=100"
            else:
                url = f"https://www.reddit.com/r/{subreddit}/{sort}.rss?limit=100"
            try:
                response = requests.get(url, headers={"User-Agent": settings.reddit_user_agent}, timeout=settings.request_timeout_seconds)
                if response.status_code != 200:
                    continue
                root = ET.fromstring(response.text)
            except Exception:
                continue
            for entry in root.findall("atom:entry", atom_ns):
                title = entry.findtext("atom:title", "", atom_ns)
                body = _strip_html(entry.findtext("atom:content", "", atom_ns))
                full = clean_text(f"{title} {body}")
                matches = [item for item, needle in needles if needle and needle in full]
                if not matches:
                    continue
                link_el = entry.find("atom:link", atom_ns)
                url_out = link_el.get("href", "") if link_el is not None else ""
                author_el = entry.find("atom:author", atom_ns)
                author = author_el.findtext("atom:name", "[deleted]", atom_ns) if author_el is not None else "[deleted]"
                published = entry.findtext("atom:published", "", atom_ns) or entry.findtext("atom:updated", "", atom_ns)
                for item in matches[:3]:
                    records.append(
                        _record(
                            source="reddit_rss",
                            platform="reddit",
                            sub_source=f"r/{subreddit}",
                            source_type="submission_rss",
                            keyword=item["keyword"],
                            query=item["query"],
                            category=item["category"],
                            title=title,
                            text=body,
                            author=author or "[deleted]",
                            community=subreddit,
                            published_at=published,
                            url=url_out,
                        )
                    )
                    saved_for_sub += 1
                    if saved_for_sub >= per_subreddit:
                        break
                if saved_for_sub >= per_subreddit:
                    break
            if saved_for_sub >= per_subreddit:
                break
            time.sleep(settings.min_request_delay_seconds)
    return records


def collect_youtube_ytdlp(
    query_items: list[dict[str, str]],
    settings: Settings,
    max_queries: int,
    videos_per_query: int,
    comments_per_video: int,
) -> list[dict[str, Any]]:
    if not shutil.which("yt-dlp"):
        log.warning("yt-dlp not installed; skipping YouTube no-API collector")
        return []
    records: list[dict[str, Any]] = []
    for item in query_items[:max_queries]:
        search_query = f"{item['query']} review trend TikTok product"
        cmd = ["yt-dlp", "--dump-json", "--flat-playlist", "--no-download", f"ytsearch{videos_per_query}:{search_query}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if result.returncode != 0:
                continue
            videos = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        except Exception as exc:
            log.warning("yt-dlp search failed for %s: %s", item["query"], exc)
            continue
        for video in videos:
            video_id = video.get("id")
            if not video_id:
                continue
            detail_cmd = [
                "yt-dlp",
                "--dump-json",
                "--skip-download",
                "--write-comments",
                "--extractor-args",
                f"youtube:comment_sort=top;max_comments={comments_per_video}",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            try:
                detail_result = subprocess.run(detail_cmd, capture_output=True, text=True, timeout=90)
                if detail_result.returncode != 0:
                    continue
                detail = json.loads(detail_result.stdout)
            except Exception:
                continue
            upload = detail.get("upload_date", "")
            published = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if upload and len(upload) >= 8 else ""
            title = detail.get("title", "")
            description = detail.get("description", "")
            records.append(
                _record(
                    source="youtube_ytdlp",
                    platform="youtube",
                    sub_source=detail.get("channel", ""),
                    source_type="video",
                    keyword=item["keyword"],
                    query=item["query"],
                    category=item["category"],
                    title=title,
                    text=description,
                    author=detail.get("channel", ""),
                    community=detail.get("channel", ""),
                    published_at=published,
                    url=f"https://youtube.com/watch?v={video_id}",
                    engagement_score=(detail.get("like_count") or 0) + (detail.get("comment_count") or 0) * 3,
                    metrics={"views": detail.get("view_count", 0), "likes": detail.get("like_count", 0), "comments": detail.get("comment_count", 0)},
                )
            )
            for comment in detail.get("comments", [])[:comments_per_video]:
                text = comment.get("text", "")
                if not text:
                    continue
                records.append(
                    _record(
                        source="youtube_ytdlp",
                        platform="youtube",
                        sub_source=detail.get("channel", ""),
                        source_type="comment",
                        keyword=item["keyword"],
                        query=item["query"],
                        category=item["category"],
                        title=title,
                        text=text,
                        author=comment.get("author", ""),
                        community=detail.get("channel", ""),
                        published_at=comment.get("timestamp") or published,
                        url=f"https://youtube.com/watch?v={video_id}&lc={comment.get('id', '')}",
                        engagement_score=comment.get("like_count") or 0,
                        metrics={"video_id": video_id, "likes": comment.get("like_count", 0)},
                    )
                )
            time.sleep(settings.min_request_delay_seconds)
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
    if cfg.get("producthunt_enabled", True):
        records.extend(collect_producthunt_rss(queries, settings, int(cfg.get("producthunt_per_query", 20))))
    if cfg.get("reddit_pullpush_enabled", True):
        records.extend(
            collect_reddit_pullpush(
                queries,
                settings,
                start_date,
                end_date,
                int(cfg.get("reddit_pullpush_per_query", 120)),
                include_comments=bool(cfg.get("reddit_pullpush_comments_enabled", True)),
            )
        )
    if cfg.get("reddit_rss_enabled", True):
        records.extend(collect_reddit_rss(config, queries, settings, int(cfg.get("reddit_rss_per_subreddit", 80))))
    if cfg.get("youtube_ytdlp_enabled", False):
        records.extend(
            collect_youtube_ytdlp(
                queries,
                settings,
                int(cfg.get("youtube_max_queries", 20)),
                int(cfg.get("youtube_videos_per_query", 8)),
                int(cfg.get("youtube_comments_per_video", 30)),
            )
        )
    df = pd.DataFrame(records).reindex(columns=MENTION_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates("mention_id")
    return {"trend_mentions_raw": df}
