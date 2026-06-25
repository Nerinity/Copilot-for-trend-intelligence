from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
import requests

from ..preprocess import clean_text, extract_hashtags, extract_keywords, stable_hash
from ..settings import Settings
from ..storage import normalize_records, now_utc

log = logging.getLogger(__name__)

CULTURE_SIGNALS = [
    "aesthetic", "vibe", "era", "core", "lifestyle", "trend", "culture",
    "routine", "wellness", "quiet luxury", "underconsumption", "dopamine",
]

PRODUCT_SIGNALS = [
    "review", "haul", "unboxing", "worth it", "buy", "product", "shop",
    "sale", "deal", "brand", "launch", "tiktok shop", "amazon", "dupe",
    "skincare", "makeup", "supplement", "gadget",
]


def _headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.twitter_bearer_token}"}


def classify_topic(text: str) -> str:
    t = text.lower()
    culture_hits = sum(1 for signal in CULTURE_SIGNALS if signal in t)
    product_hits = sum(1 for signal in PRODUCT_SIGNALS if signal in t)
    if product_hits > culture_hits:
        return "product_trend"
    if culture_hits > product_hits:
        return "culture_trend"
    if product_hits or culture_hits:
        return "mixed"
    return "other"


def fetch_trending_topics(settings: Settings, woeid: int = 23424977) -> pd.DataFrame:
    if not settings.twitter_bearer_token:
        return pd.DataFrame()
    url = f"https://api.twitter.com/1.1/trends/place.json?id={woeid}"
    try:
        response = requests.get(url, headers=_headers(settings), timeout=settings.request_timeout_seconds)
        if response.status_code != 200:
            log.warning("Twitter trends HTTP %s: %s", response.status_code, response.text[:160])
            return pd.DataFrame()
        rows = []
        for rank, trend in enumerate(response.json()[0].get("trends", []), 1):
            topic = trend.get("name", "")
            rows.append(
                {
                    "topic": topic,
                    "rank": rank,
                    "tweet_volume": trend.get("tweet_volume") or 0,
                    "topic_type": classify_topic(topic),
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        log.warning("Twitter trends failed: %s", exc)
        return pd.DataFrame()


def build_queries(config: dict[str, Any], taxonomy_terms: list[str] | None = None) -> list[str]:
    tw_cfg = config.get("twitter", {})
    queries = []
    queries.extend(tw_cfg.get("seed_queries", []))
    for category, values in tw_cfg.get("category_keywords", {}).items():
        for value in values:
            queries.append(f'("{value}") ({category.replace("_", " ")} OR review OR viral OR "tiktok shop")')
    for term in (taxonomy_terms or [])[: int(tw_cfg.get("taxonomy_query_limit", 60))]:
        if len(term.split()) <= 5:
            queries.append(f'"{term}" (review OR viral OR haul OR "worth it" OR dupe)')
    seen = set()
    out = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[: int(tw_cfg.get("max_queries", 120))]


def search_tweets(query: str, settings: Settings, start_time: str, max_results: int = 100) -> list[dict]:
    if not settings.twitter_bearer_token:
        log.warning("TWITTER_BEARER_TOKEN is not set; skipping Twitter/X.")
        return []
    endpoint = (
        "https://api.twitter.com/2/tweets/search/all"
        if settings.twitter_search_mode == "all"
        else "https://api.twitter.com/2/tweets/search/recent"
    )
    params = {
        "query": f"{query} lang:en -is:retweet",
        "max_results": min(100, max_results),
        "start_time": start_time,
        "tweet.fields": "created_at,author_id,public_metrics,entities,context_annotations,lang",
        "expansions": "author_id",
        "user.fields": "username,public_metrics,description,verified",
    }
    tweets = []
    next_token = None
    while len(tweets) < max_results:
        if next_token:
            params["next_token"] = next_token
        r = requests.get(endpoint, headers=_headers(settings), params=params, timeout=settings.request_timeout_seconds)
        if r.status_code == 429:
            reset = int(r.headers.get("x-rate-limit-reset", time.time() + 900))
            wait = max(60, reset - int(time.time()))
            log.warning("Twitter rate limited; waiting %ss", wait)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            log.warning("Twitter search HTTP %s for %s: %s", r.status_code, query, r.text[:200])
            break
        data = r.json()
        users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
        for tweet in data.get("data", []):
            user = users.get(tweet.get("author_id", ""), {})
            tweets.append({"tweet": tweet, "user": user})
        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            break
        time.sleep(settings.min_request_delay_seconds)
    return tweets


def collect(
    config: dict[str, Any],
    settings: Settings,
    run_id: str,
    start_date: str,
    end_date: str,
    taxonomy_terms: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if not settings.twitter_bearer_token:
        return {"twitter_posts": pd.DataFrame(), "twitter_keywords": pd.DataFrame()}
    if settings.twitter_search_mode != "all":
        start_dt = datetime.now(timezone.utc) - timedelta(days=6, hours=20)
        log.warning("Twitter recent search can only access recent data; using %s", start_dt.isoformat())
    else:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    start_time = start_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    max_results = int(config.get("twitter", {}).get("max_results_per_query", 80))

    records = []
    culture_rows = []
    product_rows = []
    texts = []
    hashtag_counts: dict[str, int] = {}
    for query in build_queries(config, taxonomy_terms):
        for item in search_tweets(query, settings, start_time=start_time, max_results=max_results):
            tweet = item["tweet"]
            user = item.get("user", {})
            metrics = tweet.get("public_metrics", {})
            text = tweet.get("text", "")
            for tag in extract_hashtags(text):
                hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
            contexts = [
                f"{ann.get('domain', {}).get('name', '')}:{ann.get('entity', {}).get('name', '')}"
                for ann in tweet.get("context_annotations", [])
            ]
            engagement = (
                metrics.get("like_count", 0)
                + metrics.get("retweet_count", 0) * 2
                + metrics.get("reply_count", 0) * 3
                + metrics.get("quote_count", 0) * 2
            )
            layer = classify_topic(query)
            if layer == "mixed":
                layer = "product_trend" if any(s in query.lower() for s in PRODUCT_SIGNALS) else "culture_trend"
            raw_row = {
                "tweet_id": tweet.get("id", ""),
                "text": text,
                "created_at": tweet.get("created_at", ""),
                "author_followers": user.get("public_metrics", {}).get("followers_count", 0),
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "engagement": engagement,
                "twitter_context": "|".join(contexts[:3]),
                "layer": layer,
                "query": query,
                "url": f"https://twitter.com/i/web/status/{tweet.get('id')}",
            }
            if layer == "product_trend":
                product_rows.append(raw_row)
            else:
                culture_rows.append(raw_row)
            record = {
                "run_id": run_id,
                "source": "twitter_x",
                "source_type": "post",
                "source_id": f"tweet_{tweet.get('id')}",
                "source_url": f"https://twitter.com/i/web/status/{tweet.get('id')}",
                "category_hint": "social_discussion",
                "community": "twitter_x",
                "query": query,
                "title": "",
                "text": text,
                "clean_text": clean_text(text),
                "author": user.get("username", tweet.get("author_id", "")),
                "published_at": tweet.get("created_at", ""),
                "collected_at": now_utc(),
                "engagement_score": engagement,
                "metrics_json": json.dumps({"tweet": metrics, "user": user.get("public_metrics", {}), "contexts": contexts}, ensure_ascii=False),
                "content_hash": stable_hash("twitter", tweet.get("id"), text),
                "raw_json": json.dumps(item, ensure_ascii=False)[:20000],
            }
            records.append(record)
            texts.append(record["clean_text"])
        time.sleep(settings.min_request_delay_seconds)

    keywords = extract_keywords(texts, top_n=100)
    keyword_df = pd.DataFrame(keywords)
    if not keyword_df.empty:
        keyword_df.insert(0, "run_id", run_id)
        keyword_df["source"] = "twitter_x"
        keyword_df["collected_at"] = now_utc()
    hashtag_df = pd.DataFrame(
        [{"run_id": run_id, "source": "twitter_x", "keyword": f"#{k}", "score": v, "method": "hashtag"} for k, v in sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:100]]
    )
    if not hashtag_df.empty:
        keyword_df = pd.concat([keyword_df, hashtag_df], ignore_index=True)
    trending_df = fetch_trending_topics(settings)
    culture_df = pd.DataFrame(culture_rows).drop_duplicates("tweet_id") if culture_rows else pd.DataFrame()
    product_df = pd.DataFrame(product_rows).drop_duplicates("tweet_id") if product_rows else pd.DataFrame()
    return {
        "twitter_trending": trending_df,
        "twitter_culture_posts": culture_df,
        "twitter_product_posts": product_df,
        "_twitter_posts_normalized": normalize_records(records),
        "_twitter_keywords_normalized": keyword_df,
    }
