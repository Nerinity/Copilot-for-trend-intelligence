from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import load_config
from .logging_utils import configure_logging
from .settings import get_settings
from .storage import append_dedup_csv, ensure_data_dirs, record_run, write_snapshot
from .taxonomy import load_product_taxonomy, taxonomy_keywords

log = logging.getLogger(__name__)


def _save_frames(frames: dict[str, pd.DataFrame], data_dir: Path, run_id: str) -> None:
    for name, df in frames.items():
        if df is None or df.empty:
            log.info("%s produced no rows", name)
            continue
        raw_path = data_dir / "raw" / f"{name}.csv"
        if "content_hash" in df.columns:
            append_dedup_csv(df, raw_path, subset=["content_hash"])
        elif "keyword" in df.columns and "source" in df.columns:
            append_dedup_csv(df, raw_path, subset=[c for c in ["source", "keyword", "method"] if c in df.columns])
        else:
            append_dedup_csv(df, raw_path)
        snapshot = write_snapshot(df, data_dir, name, run_id)
        log.info("Snapshot written: %s", snapshot)


def run_collection(
    sources: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    mode: str = "incremental",
    config_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    settings = get_settings()
    configure_logging()
    ensure_data_dirs(settings.data_dir)
    start_date = start_date or settings.start_date
    end_date = end_date or settings.end_date
    config = load_config(config_path or settings.config_path)
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    taxonomy_terms: list[str] = []
    taxonomy_path = config.get("taxonomy", {}).get("product_category_theme_csv")
    if taxonomy_path and Path(taxonomy_path).exists():
        taxonomy_df = load_product_taxonomy(taxonomy_path)
        taxonomy_terms = taxonomy_keywords(taxonomy_df, limit=500)
        log.info("Loaded %s taxonomy terms from %s", len(taxonomy_terms), taxonomy_path)

    selected = set(sources)
    if "all" in selected:
        selected = {"reddit", "twitter", "google_trends", "public_sources"}

    frames: dict[str, pd.DataFrame] = {}

    if "reddit" in selected:
        from .sources import reddit

        frames.update(reddit.collect(config, settings, run_id, start_date, end_date, mode=mode))

    if "public_sources" in selected:
        from .sources import public_sources

        frames.update(public_sources.collect(config, settings, run_id, start_date, end_date))

    seed_keywords = []
    for key in ["reddit_keywords", "public_keywords", "twitter_keywords"]:
        df = frames.get(key)
        if df is not None and not df.empty and "keyword" in df.columns:
            seed_keywords.extend(df["keyword"].dropna().astype(str).head(80).tolist())
    seed_keywords.extend(taxonomy_terms[:80])

    if "google_trends" in selected:
        from .sources import gtrends

        frames.update(gtrends.collect(config, run_id, start_date, end_date, seed_keywords=seed_keywords))

    if "twitter" in selected:
        from .sources import twitter

        frames.update(twitter.collect(config, settings, run_id, start_date, end_date, taxonomy_terms=taxonomy_terms))

    _save_frames(frames, settings.data_dir, run_id)
    record_run(settings.data_dir, run_id, sorted(selected), start_date, end_date)
    log.info("Collection run complete: %s", run_id)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 trend data collection.")
    parser.add_argument("--sources", default="all", help="Comma list: reddit,twitter,google_trends,public_sources,all")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD. Default from .env/settings.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD. Default from .env/settings.")
    parser.add_argument("--mode", choices=["init", "incremental"], default="incremental")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    run_collection(sources, args.start_date, args.end_date, mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    main()
