from datetime import datetime, timezone

import pandas as pd

from trend_copilot_data.sources.gtrends import _summarize
from trend_copilot_data.sources.reddit import _raw_post_row


def test_reddit_raw_post_schema():
    row = _raw_post_row(
        {
            "id": "abc123",
            "subreddit": "SkincareAddiction",
            "title": "Routine help",
            "selftext": "Dry skin question",
            "score": 10,
            "upvote_ratio": 0.9,
            "num_comments": 3,
            "author": "user",
            "created_utc": datetime(2026, 4, 15, tzinfo=timezone.utc).timestamp(),
            "permalink": "/r/SkincareAddiction/comments/abc123/routine_help/",
            "link_flair_text": "Routine Help",
        },
        "culture_trend",
        datetime(2026, 4, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 25, tzinfo=timezone.utc),
    )
    assert list(row.keys()) == [
        "post_id",
        "subreddit",
        "layer",
        "title",
        "body",
        "full_text",
        "score",
        "upvote_ratio",
        "num_comments",
        "author",
        "created_date",
        "url",
        "flair",
    ]


def test_gtrends_category_summary_schema():
    summary = _summarize(
        pd.DataFrame([{"keyword": "matcha", "category": "Food Beverage", "growth_recent_pct": 42, "is_rising": True}]),
        pd.DataFrame([{"related_keyword": "matcha powder", "category": "Food Beverage", "is_breakout": False}]),
    )
    assert list(summary.columns) == [
        "category",
        "trend_score",
        "keyword_count",
        "rising_keywords",
        "top_keywords",
    ]
