from trend_copilot_data.preprocess import clean_text, stable_hash


def test_clean_text_removes_urls_and_normalizes_space():
    assert clean_text("Hello   https://example.com  Ice Roller!") == "hello ice roller"


def test_stable_hash_is_stable():
    assert stable_hash("a", "b") == stable_hash("a", "b")
