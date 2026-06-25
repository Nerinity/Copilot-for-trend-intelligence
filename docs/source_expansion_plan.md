# Source Expansion Plan

## High-Value Sources

| Source | Value | Preferred Access |
| --- | --- | --- |
| Reddit posts/comments | Early consumer language, pain points, niche communities | Official Reddit API |
| Twitter/X | Hashtags, creator discourse, fast-moving conversation | X API |
| Google Trends | Search validation and category time-series | pytrends or official/approved provider |
| YouTube | Creator/video/comment signals | YouTube Data API |
| TikTok | Short-video and social commerce signals | Research API or approved provider |
| Pinterest Trends | Visual/lifestyle intent | Pinterest Trends export/API partner |
| Product Hunt | Emerging tech/product launches | Product Hunt API/RSS |
| Hacker News | Tech/startup/AI signals | Public Algolia API |
| Amazon Best Sellers | Commercial demand proxy | Compliant API/export workflow |
| Review sites | Consumer pain points and product gaps | API, RSS, or permitted scraping |

## Sustainable Workflow

1. Seed from taxonomy and curated source config.
2. Collect raw records daily.
3. Normalize into the standard signal schema.
4. Deduplicate by source ID and content hash.
5. Extract keywords, products, pain points, and entities.
6. Aggregate by day, source, keyword, category, and product.
7. Compute velocity, volume, sentiment, competition, and cross-platform confirmation.
8. Serve dashboard and LLM layers from processed aggregates.
