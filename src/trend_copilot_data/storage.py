from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .preprocess import stable_hash

log = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "run_id", "source", "source_type", "source_id", "source_url", "category_hint",
    "community", "query", "title", "text", "clean_text", "author", "published_at",
    "collected_at", "engagement_score", "metrics_json", "content_hash", "raw_json",
]


def ensure_data_dirs(data_dir: Path) -> None:
    for sub in ["raw", "processed", "snapshots", "state"]:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_records(records: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {col: r.get(col, "") for col in STANDARD_COLUMNS}
        if not row["content_hash"]:
            row["content_hash"] = stable_hash(
                row.get("source"), row.get("source_id"), row.get("title"), row.get("text")
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=STANDARD_COLUMNS)


def append_dedup_csv(df: pd.DataFrame, path: Path, subset: list[str] | None = None) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = subset or ["content_hash"]
    subset = [col for col in subset if col in df.columns]
    if not subset:
        subset = list(df.columns)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()
    before = len(combined)
    combined = combined.drop_duplicates(subset=subset, keep="last")
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("Saved %s rows to %s (%s duplicates removed)", len(combined), path, before - len(combined))
    return combined


def write_snapshot(df: pd.DataFrame, data_dir: Path, name: str, run_id: str) -> Path:
    snapshot_dir = data_dir / "snapshots" / run_id[:10]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{name}_{run_id}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_state(data_dir: Path) -> dict:
    path = data_dir / "state" / "collection_state.json"
    if not path.exists():
        return {"watermarks": {}, "runs": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(data_dir: Path, state: dict) -> None:
    path = data_dir / "state" / "collection_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def record_run(data_dir: Path, run_id: str, sources: list[str], start_date: str, end_date: str) -> None:
    state = load_state(data_dir)
    state.setdefault("runs", []).append(
        {
            "run_id": run_id,
            "sources": sources,
            "start_date": start_date,
            "end_date": end_date,
            "finished_at": now_utc(),
        }
    )
    for source in sources:
        state.setdefault("watermarks", {})[source] = end_date
    save_state(data_dir, state)
