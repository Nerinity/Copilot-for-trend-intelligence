# AI Trend Detection Copilot: Data Foundation

This repository is the Stage 1 data collection layer for an AI Trend Detection Copilot for creator commerce.

The current focus is **data collection only**:

- collect recent market signals from public sources
- normalize all sources into one event schema
- deduplicate and snapshot raw signals
- prepare a maintainable foundation for future dashboard, product dictionary, and recommendation copilot layers

## Product Vision

Layer 1: Trend Dashboard  
Continuously monitor emerging products, topics, keywords, cross-platform velocity, category distribution, and time-series trend changes.

Layer 2: Product Intelligence Dictionary  
Given a product, generate a structured opportunity report covering growth, volume, commercial value, competition, consumer interests, pain points, sentiment, risks, and related products.

Layer 3: Trend Recommendation Copilot  
Recommend creator personas, creators, products, matching scores, and explanations for commercializing emerging trends.

## Current Stage

Stage 1 collects signals from:

- Reddit posts and comments across high-value communities
- Twitter/X search, if API credentials are available
- Google Trends category keywords and related rising queries
- Hacker News public API for tech/startup/consumer AI signals
- Google News RSS for public market/news trend signals

Historical initialization target:

```text
2026-04-01 through 2026-06-25
```

After initialization, run the same pipeline in incremental mode daily.

## Repository Structure

```text
.
├── configs/
│   ├── pipeline.toml
│   ├── sources.json
│   └── taxonomy/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── snapshots/
│   └── state/
├── docs/
├── scripts/
│   ├── collect.py
│   └── prepare_taxonomy.py
├── src/trend_copilot_data/
│   ├── pipeline.py
│   ├── storage.py
│   ├── preprocess.py
│   ├── taxonomy.py
│   └── sources/
├── tests/
├── run_trend_scrapers.py
├── reddit_smart.py
├── twitter_smart.py
└── gtrends_smart.py
```

The root `*_smart.py` files are compatibility wrappers. New development should happen under `src/trend_copilot_data`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Fill optional credentials:

```text
TWITTER_BEARER_TOKEN=...
```

Twitter/X historical collection before the recent-search window requires full-archive access. Without it, the pipeline still runs Reddit, Google Trends, Hacker News, and RSS.

## Run Historical Initialization

```bash
python -m trend_copilot_data.pipeline \
  --mode init \
  --sources reddit,google_trends,public_sources,twitter \
  --start-date 2026-04-01 \
  --end-date 2026-06-25
```

Fast public-source smoke run:

```bash
python -m trend_copilot_data.pipeline \
  --mode init \
  --sources public_sources \
  --start-date 2026-04-01 \
  --end-date 2026-06-25
```

## Run Incremental Updates

```bash
python -m trend_copilot_data.pipeline --mode incremental --sources all
```

Outputs:

- `data/raw/*.csv`: append-only deduplicated raw source tables
- `data/snapshots/YYYY-MM-DD/*.csv`: per-run snapshots
- `data/state/collection_state.json`: run history and source watermarks
- `logs/*.log`: execution logs

## Data Contract

All public discussion sources are normalized into this schema:

```text
run_id, source, source_type, source_id, source_url, category_hint,
community, query, title, text, clean_text, author, published_at,
collected_at, engagement_score, metrics_json, content_hash, raw_json
```

This gives future AI/LLM layers a stable contract regardless of source.

## GitHub Upload

This folder is designed to be a GitHub repository. To upload:

```bash
git init
git branch -M main
git add .
git commit -m "Build trend data collection foundation"
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

If using Codex GitHub integration, provide the target repository as `owner/repo`.

## Notes on Legal and API Limits

- Reddit public JSON is useful for early validation but can miss deep historical coverage. Use the official Reddit API for production.
- Twitter/X recent search is limited unless the account has full-archive access.
- TikTok, Pinterest, Amazon, Product Hunt, YouTube, and review sites should be integrated through official APIs, RSS, exports, or compliant public endpoints.
- Respect platform terms, rate limits, robots policies, and privacy constraints.
