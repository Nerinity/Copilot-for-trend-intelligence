from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
