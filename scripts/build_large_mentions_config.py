#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a large one-year dashboard mention crawl config.")
    parser.add_argument("--base", default="configs/sources.json")
    parser.add_argument("--output", default="work/mentions_large_1y.json")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--taxonomy-query-limit", type=int, default=250)
    args = parser.parse_args()

    with Path(args.base).open("r", encoding="utf-8") as f:
        config = json.load(f)

    mentions = config.setdefault("dashboard_mentions", {})
    mentions.update(
        {
            "max_queries": args.max_queries,
            "taxonomy_query_limit": args.taxonomy_query_limit,
            "google_news_enabled": True,
            "google_news_per_query": 100,
            "gdelt_enabled": False,
            "hackernews_enabled": True,
            "hackernews_per_query": 60,
            "producthunt_enabled": True,
            "producthunt_per_query": 50,
            "reddit_pullpush_enabled": True,
            "reddit_pullpush_per_query": 250,
            "reddit_pullpush_comments_enabled": True,
            "reddit_rss_enabled": True,
            "reddit_rss_per_subreddit": 140,
            "reddit_rss_max_subreddits": 220,
            "youtube_ytdlp_enabled": False,
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(out)


if __name__ == "__main__":
    main()
