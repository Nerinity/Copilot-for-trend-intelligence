from trend_copilot_data.config import load_config


def test_sources_config_loads():
    cfg = load_config("configs/sources.json")
    assert "reddit" in cfg
    assert "subreddit_categories" in cfg["reddit"]
