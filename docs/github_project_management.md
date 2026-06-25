# GitHub Project Management

## Suggested Milestones

1. Stage 1A: Data collection foundation
2. Stage 1B: Source expansion and quality monitoring
3. Stage 1C: Trend scoring tables
4. Stage 2: Product intelligence dictionary
5. Stage 3: Recommendation copilot

## Suggested Labels

- `stage:data-collection`
- `source:reddit`
- `source:twitter`
- `source:gtrends`
- `source:public`
- `pipeline`
- `data-quality`
- `documentation`
- `future-dashboard`
- `future-llm`

## Initial Issues

### Expand YouTube Data API collector

Collect video metadata and public comments for category/product trend queries.

### Add official Reddit API client

Replace public JSON fallback with authenticated API and stronger rate-limit handling.

### Build processed trend scoring table

Aggregate raw records by keyword, source, category, day, velocity, sentiment, and engagement.

### Add dashboard prototype

Build the Layer 1 dashboard on top of processed trend tables.

### Add product dictionary endpoint

Given a product name, retrieve discussion growth, volume, category fit, sentiment, pain points, risk alerts, and opportunity score.
