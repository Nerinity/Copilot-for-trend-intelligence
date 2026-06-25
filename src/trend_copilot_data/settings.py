from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("TREND_DATA_DIR", "data"))
    config_path: Path = Path(os.getenv("TREND_COLLECTION_CONFIG", "configs/sources.json"))
    start_date: str = os.getenv("TREND_START_DATE", "2026-04-01")
    end_date: str = os.getenv("TREND_END_DATE", date.today().isoformat())
    geo: str = os.getenv("TREND_GEO", "US")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    twitter_search_mode: str = os.getenv("TWITTER_SEARCH_MODE", "recent")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent: str = os.getenv(
        "REDDIT_USER_AGENT",
        "trend-detection-copilot/0.1 contact:your_email@example.com",
    )
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
    min_request_delay_seconds: float = float(os.getenv("MIN_REQUEST_DELAY_SECONDS", "1.2"))
    max_request_retries: int = int(os.getenv("MAX_REQUEST_RETRIES", "3"))


def get_settings() -> Settings:
    return Settings()
