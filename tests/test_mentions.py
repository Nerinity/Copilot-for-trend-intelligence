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
