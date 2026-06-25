# Raw Data Schema

Stage 1 prioritizes source-native raw CSV files for NLP and trend detection.

## Reddit

`data/raw/reddit_culture_trends.csv` and `data/raw/reddit_product_trends.csv`

```text
post_id, subreddit, layer, title, body, full_text, score, upvote_ratio,
num_comments, author, created_date, url, flair
```

`data/raw/reddit_tfidf_keywords.csv`

```text
keyword, tfidf_score, layer
```

## Google Trends

`data/raw/gtrends_trending.csv`

```text
keyword, type, rank, category, fetched_at
```

`data/raw/gtrends_timeseries.csv`

```text
keyword, category, current_score, prev_score, peak_score, peak_week,
growth_recent_pct, growth_longterm_pct, trend_shape, is_rising
```

`data/raw/gtrends_rising.csv`

```text
seed_keyword, related_keyword, rise_value, is_breakout, category, signal_type
```

`data/raw/gtrends_category_summary.csv`

```text
category, trend_score, keyword_count, rising_keywords, top_keywords
```

## Twitter/X

`data/raw/twitter_trending.csv`

```text
topic, rank, tweet_volume, topic_type, fetched_at
```

`data/raw/twitter_culture_posts.csv` and `data/raw/twitter_product_posts.csv`

```text
tweet_id, text, created_at, author_followers, likes, retweets, replies,
engagement, twitter_context, layer, query, url
```

## Direct NLP Inputs

- Trend discovery: `gtrends_timeseries.trend_shape` and `gtrends_timeseries.growth_recent_pct`
- Culture signal: `reddit_tfidf_keywords.keyword` and `reddit_tfidf_keywords.tfidf_score`
- Early warning: `gtrends_rising.is_breakout = True`
- Sentiment: Reddit `full_text` and Twitter `text`
- Cross-platform validation: overlap between Reddit keywords, Google rising queries, and Twitter queries/hashtags
