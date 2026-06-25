# Operations

## Scheduling

Recommended production schedule:

- Daily incremental collection at 03:17 Pacific Time
- Weekly backfill for the last 14 days
- Monthly taxonomy refresh

GitHub Actions example is included in `.github/workflows/daily_collection.yml`.

## Failure Recovery

Each run writes:

- raw append tables
- per-run snapshots
- logs
- state file

If a run fails, rerun the same command. Deduplication uses `content_hash`, so duplicate rows are removed automatically.

## Monitoring

Minimum alerts to add:

- source returns zero rows
- API auth failure
- rate limit spike
- row count drops more than 50% day over day
- schema changes
- snapshot write failure

## Versioning

Keep code in git. Keep large raw datasets out of git unless they are small samples.

For production-scale data, use:

- object storage for raw snapshots
- DuckDB/Postgres/BigQuery for queryable history
- dbt or scheduled SQL for scoring tables

## API Management

Store credentials in environment variables:

- `TWITTER_BEARER_TOKEN`
- future Reddit API credentials
- future YouTube API key
- future Product Hunt token

Never commit `.env`.
