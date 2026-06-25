import pandas as pd

from trend_copilot_data.storage import append_dedup_csv


def test_append_dedup_csv_handles_tables_without_content_hash(tmp_path):
    path = tmp_path / "summary.csv"
    df = pd.DataFrame([{"category_hint": "beauty", "trend_score": 10}])
    saved = append_dedup_csv(df, path)
    assert len(saved) == 1
