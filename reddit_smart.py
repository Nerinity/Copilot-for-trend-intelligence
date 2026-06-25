#!/usr/bin/env python3
"""Compatibility wrapper for the new Reddit collector."""

from trend_copilot_data.config import load_config
from trend_copilot_data.settings import get_settings
from trend_copilot_data.sources.reddit import collect


def scrape():
    settings = get_settings()
    config = load_config(settings.config_path)
    return collect(config, settings, "manual-reddit", settings.start_date, settings.end_date)


if __name__ == "__main__":
    scrape()
