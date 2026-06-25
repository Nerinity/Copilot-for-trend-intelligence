from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from ..preprocess import clean_text, extract_keywords, stable_hash
from ..settings import Settings
from ..storage import normalize_records, now_utc

log = logging.getLogger(__name__)
_REDDIT_ACCESS_TOKEN: str | None = None


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _created_at(post: dict[str, Any]) -> datetime:
    return datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)


def _reddit_access_token(settings: Settings) -> str | None:
    global _REDDIT_ACCESS_TOKEN
    if _REDDIT_ACCESS_TOKEN:
        return _REDDIT_ACCESS_TOKEN
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        return None
    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=HTTPBasicAuth(settings.reddit_client_id, settings.reddit_client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": settings.reddit_user_agent},
            timeout=settings.request_timeout_seconds,
        )
        if response.status_code != 200:
            log.warning("Reddit OAuth token HTTP %s: %s", response.status_code, response.text[:120])
            return None
        _REDDIT_ACCESS_TOKEN = response.json().get("access_token")
        return _REDDIT_ACCESS_TOKEN
    except Exception as exc:
        log.warning("Reddit OAuth token failed: %s", exc)
        return None


def _headers(settings: Settings) -> dict[str, str]:
    headers = {
        "User-Agent": settings.reddit_user_agent,
        "Accept": "application/json",
    }
    token = _reddit_access_token(settings)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _base_url(settings: Settings) -> str:
    return "https://oauth.reddit.com" if _reddit_access_token(settings) else "https://www.reddit.com"


def _request_json(url: str, settings: Settings, params: dict | None = None) -> dict | None:
    headers = _headers(settings)
    for attempt in range(settings.max_request_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=settings.request_timeout_seconds)
            if r.status_code in {429, 500, 502, 503, 504}:
                wait = min(60, (attempt + 1) * 5)
                log.warning("Reddit HTTP %s for %s; retrying in %ss", r.status_code, url, wait)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                log.warning("Reddit HTTP %s for %s", r.status_code, url)
                return None
            return r.json()
        except Exception as exc:
            log.warning("Reddit request failed for %s: %s", url, exc)
            time.sleep((attempt + 1) * 2)
    return None


def fetch_subreddit_posts(
    subreddit: str,
    settings: Settings,
    sort: str = "hot",
    limit: int = 100,
    time_filter: str = "month",
) -> list[dict[str, Any]]:
    base = _base_url(settings)
    suffix = f"/r/{subreddit}/{sort}"
    url = f"{base}{suffix if 'oauth' in base else suffix + '.json'}"
    params = {"limit": min(limit, 100)}
    if sort in {"top", "controversial"}:
        params["t"] = time_filter
    data = _request_json(url, settings, params=params)
    if not data:
        return []
    return [child.get("data", {}) for child in data.get("data", {}).get("children", [])]


def fetch_comments(post: dict[str, Any], settings: Settings, limit: int = 20) -> list[dict[str, Any]]:
    permalink = post.get("permalink")
    if not permalink:
        return []
    base = _base_url(settings)
    url = f"{base}{permalink if 'oauth' in base else permalink + '.json'}"
    data = _request_json(url, settings, params={"limit": limit, "sort": "top"})
    if not isinstance(data, list) or len(data) < 2:
        return []
    comments = []
    for child in data[1].get("data", {}).get("children", [])[:limit]:
        c = child.get("data", {})
        if c.get("body"):
            comments.append(c)
    return comments


def search_reddit(query: str, settings: Settings, limit: int = 50, sort: str = "relevance") -> list[dict[str, Any]]:
    data = _request_json(
        f"{_base_url(settings)}/search{'' if 'oauth' in _base_url(settings) else '.json'}",
        settings,
        params={"q": query, "limit": min(limit, 100), "sort": sort, "t": "month", "type": "link"},
    )
    if not data:
        return []
    return [child.get("data", {}) for child in data.get("data", {}).get("children", [])]


def _post_record(
    post: dict[str, Any],
    category: str,
    settings: Settings,
    run_id: str,
    start_dt: datetime,
    end_dt: datetime,
    query: str = "",
) -> dict | None:
    created = _created_at(post)
    if created < start_dt or created > end_dt:
        return None
    title = post.get("title", "")
    body = post.get("selftext", "")
    text = f"{title} {body}".strip()
    metrics = {
        "score": post.get("score", 0),
        "upvote_ratio": post.get("upvote_ratio", 0),
        "num_comments": post.get("num_comments", 0),
        "subreddit_subscribers": post.get("subreddit_subscribers", 0),
    }
    source_id = f"reddit_post_{post.get('id', stable_hash(text))}"
    return {
        "run_id": run_id,
        "source": "reddit",
        "source_type": "post",
        "source_id": source_id,
        "source_url": f"https://reddit.com{post.get('permalink', '')}",
        "category_hint": category,
        "community": post.get("subreddit", ""),
        "query": query,
        "title": title[:500],
        "text": body[:5000],
        "clean_text": clean_text(text),
        "author": post.get("author", "[deleted]"),
        "published_at": created.isoformat(),
        "collected_at": now_utc(),
        "engagement_score": metrics["score"] + metrics["num_comments"] * 3,
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
        "content_hash": stable_hash("reddit", source_id, title, body),
        "raw_json": json.dumps(post, ensure_ascii=False)[:20000],
    }


def _comment_record(
    comment: dict[str, Any],
    post: dict[str, Any],
    category: str,
    run_id: str,
) -> dict | None:
    created = datetime.fromtimestamp(comment.get("created_utc", 0), tz=timezone.utc)
    body = comment.get("body", "")
    if not body:
        return None
    metrics = {"score": comment.get("score", 0), "parent_post_id": post.get("id", "")}
    source_id = f"reddit_comment_{comment.get('id', stable_hash(body))}"
    return {
        "run_id": run_id,
        "source": "reddit",
        "source_type": "comment",
        "source_id": source_id,
        "source_url": f"https://reddit.com{comment.get('permalink', post.get('permalink', ''))}",
        "category_hint": category,
        "community": post.get("subreddit", ""),
        "query": "",
        "title": post.get("title", "")[:500],
        "text": body[:5000],
        "clean_text": clean_text(body),
        "author": comment.get("author", "[deleted]"),
        "published_at": created.isoformat(),
        "collected_at": now_utc(),
        "engagement_score": metrics["score"],
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
        "content_hash": stable_hash("reddit_comment", source_id, body),
        "raw_json": json.dumps(comment, ensure_ascii=False)[:20000],
    }


def collect(
    config: dict[str, Any],
    settings: Settings,
    run_id: str,
    start_date: str,
    end_date: str,
    mode: str = "incremental",
) -> dict[str, pd.DataFrame]:
    reddit_cfg = config.get("reddit", {})
    start_dt = _parse_dt(start_date)
    end_dt = _parse_dt(end_date) if len(end_date) > 10 else _parse_dt(f"{end_date}T23:59:59")
    categories = reddit_cfg.get("subreddit_categories", {})
    sorts = reddit_cfg.get("sorts", ["hot", "new", "top"])
    posts_per_subreddit = int(reddit_cfg.get("posts_per_subreddit", 60))
    comment_limit = int(reddit_cfg.get("comments_per_post", 8))
    search_limit = int(reddit_cfg.get("search_results_per_query", 40))
    max_subreddits_per_category = reddit_cfg.get("max_subreddits_per_category")

    post_records: list[dict] = []
    comment_records: list[dict] = []
    texts: list[str] = []

    for category, subreddits in categories.items():
        selected = subreddits[:max_subreddits_per_category] if max_subreddits_per_category else subreddits
        log.info("Reddit category %s: %s communities", category, len(selected))
        for subreddit in selected:
            seen_posts: dict[str, dict] = {}
            for sort in sorts:
                for post in fetch_subreddit_posts(subreddit, settings, sort=sort, limit=posts_per_subreddit):
                    if post.get("id"):
                        seen_posts[post["id"]] = post
                time.sleep(settings.min_request_delay_seconds)

            for post in seen_posts.values():
                rec = _post_record(post, category, settings, run_id, start_dt, end_dt)
                if not rec:
                    continue
                post_records.append(rec)
                texts.append(rec["clean_text"])
                if comment_limit > 0:
                    for comment in fetch_comments(post, settings, limit=comment_limit):
                        crecord = _comment_record(comment, post, category, run_id)
                        if crecord:
                            comment_records.append(crecord)
                            texts.append(crecord["clean_text"])
                    time.sleep(settings.min_request_delay_seconds)

    seed_queries = reddit_cfg.get("search_queries", [])
    for query in seed_queries:
        for post in search_reddit(query, settings, limit=search_limit):
            category = "search_expansion"
            rec = _post_record(post, category, settings, run_id, start_dt, end_dt, query=query)
            if rec:
                post_records.append(rec)
                texts.append(rec["clean_text"])
        time.sleep(settings.min_request_delay_seconds)

    posts_df = normalize_records(post_records)
    comments_df = normalize_records(comment_records)
    keywords_df = pd.DataFrame(extract_keywords(texts, top_n=100))
    if not keywords_df.empty:
        keywords_df.insert(0, "run_id", run_id)
        keywords_df["source"] = "reddit"
        keywords_df["collected_at"] = now_utc()
    log.info("Reddit collected %s posts, %s comments", len(posts_df), len(comments_df))
    return {"reddit_posts": posts_df, "reddit_comments": comments_df, "reddit_keywords": keywords_df}
