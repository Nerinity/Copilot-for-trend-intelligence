import pandas as pd

from trend_copilot_data.semantic import score_mentions
from trend_copilot_data.sources.mentions import MENTION_COLUMNS, _record


def test_mention_record_schema():
    row = _record(
        source="google_news_rss",
        platform="news",
        sub_source="Example",
        source_type="article_snippet",
        keyword="matcha",
        query="matcha",
        category="food_beverage",
        title="Matcha trend",
        text="<b>Matcha</b> is trending",
        author="Example",
        community="Example",
        published_at="2026-06-25",
        url="https://example.com",
    )
    assert list(row.keys()) == MENTION_COLUMNS
    assert row["full_text"] == "matcha trend matcha is trending"


def test_semantic_scoring_keeps_tiktok_product_mentions():
    row = _record(
        source="reddit_pullpush_subreddit",
        platform="reddit",
        sub_source="r/TikTokShop",
        source_type="submission",
        keyword="r/TikTokShop",
        query="r/TikTokShop broad recent",
        category="creator_commerce_social",
        title="TikTok Shop product review",
        text="This viral Amazon finds dupe is worth it and everyone is asking for the link.",
        author="creator123",
        community="TikTokShop",
        published_at="2026-06-25",
        url="https://reddit.com/r/TikTokShop/example",
    )
    df = score_mentions(pd.DataFrame([row]), {})
    assert len(df) == 1
    assert df.iloc[0]["collector_priority_score"] > 0
    assert df.iloc[0]["tiktok_relevance_score"] > 0
