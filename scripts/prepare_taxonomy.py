#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from trend_copilot_data.taxonomy import load_product_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="configs/taxonomy/product_taxonomy_clean.csv")
    args = parser.parse_args()
    df = load_product_taxonomy(args.input_csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
