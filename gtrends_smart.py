#!/usr/bin/env python3
"""Compatibility wrapper for the new Google Trends collector."""

from trend_copilot_data.config import load_config
from trend_copilot_data.settings import get_settings
from trend_copilot_data.sources.gtrends import collect


def scrape(seed_keywords=None):
    settings = get_settings()
    config = load_config(settings.config_path)
    return collect(config, "manual-gtrends", settings.start_date, settings.end_date, seed_keywords)


if __name__ == "__main__":
    scrape()
