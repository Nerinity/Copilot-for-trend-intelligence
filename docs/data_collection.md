# Stage 1 Data Collection

## Historical Initialization

Use April 1, 2026 through June 25, 2026:

```bash
python -m trend_copilot_data.pipeline \
  --mode init \
  --sources reddit,google_trends,public_sources,twitter \
  --start-date 2026-04-01 \
  --end-date 2026-06-25
```

## Incremental Collection

Daily incremental runs should use the default dates from `.env` or today:

```bash
python -m trend_copilot_data.pipeline --mode incremental --sources all
```

The state file is stored at:

```text
data/state/collection_state.json
```

## Reddit Strategy

Reddit is organized into category-specific communities in `configs/sources.json`.

The collector pulls:

- `hot`
- `new`
- `top`
- selected top comments
- search expansion queries

This gives both velocity and depth: new posts show early movement, top posts show engagement quality, and comments reveal consumer pain points.

Default raw outputs:

```text
reddit_culture_trends.csv:
post_id, subreddit, layer, title, body, full_text, score, upvote_ratio,
num_comments, author, created_date, url, flair

reddit_product_trends.csv:
post_id, subreddit, layer, title, body, full_text, score, upvote_ratio,
num_comments, author, created_date, url, flair

reddit_tfidf_keywords.csv:
keyword, tfidf_score, layer
```

## Twitter/X Strategy

The collector uses:

- seed queries
- category keywords
- taxonomy-driven product terms
- hashtag extraction

For April-June historical backfill, use `TWITTER_SEARCH_MODE=all` only if your X API account has full-archive access.

## Google Trends Strategy

Google Trends runs category-level keywords plus discovered seed keywords from Reddit/public sources.

Outputs:

- time-series interest
- growth percentage
- related top/rising queries
- category summary

Default raw outputs:

```text
gtrends_trending.csv:
keyword, type, rank, category, fetched_at

gtrends_timeseries.csv:
keyword, category, current_score, prev_score, peak_score, peak_week,
growth_recent_pct, growth_longterm_pct, trend_shape, is_rising

gtrends_rising.csv:
seed_keyword, related_keyword, rise_value, is_breakout, category, signal_type

gtrends_category_summary.csv:
category, trend_score, keyword_count, rising_keywords, top_keywords
```

## Twitter/X Strategy

Default raw outputs:

```text
twitter_trending.csv:
topic, rank, tweet_volume, topic_type, fetched_at

twitter_culture_posts.csv and twitter_product_posts.csv:
tweet_id, text, created_at, author_followers, likes, retweets, replies,
engagement, twitter_context, layer, query, url
```

## Additional Public Sources

Currently implemented:

- Hacker News public Algolia API
- Google News RSS search

Recommended next integrations:

- YouTube Data API comments and video metadata
- TikTok Research API or approved partner/social listening provider
- Pinterest Trends manual export or approved API partner
- Product Hunt API/RSS
- Amazon Best Sellers through compliant API/export workflows
- Review platforms with permitted scraping/API access
- Brand/community forums with RSS or public APIs
