from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trend_copilot_data.preprocess import clean_text, stable_hash


VIDEO_POOL_DEFAULT = (
    "/Users/bytedance/Downloads/"
    "[for Video SPM]#1.2 Key Product Pool for Top Creator_Internal - Week69(0601-0607).csv"
)
AFFILIATE_RULES_DEFAULT = (
    "/Users/bytedance/Downloads/#1.1 Product Pool for Affiliate Creator_Internal - Read Me.csv"
)
LIVE_POOL_DEFAULT = "/Users/bytedance/Desktop/直播货盘格式0527-2026-06-25 15-36-38.xlsx"


VIBE_LEXICON = {
    "cozy_comfort": [
        "cozy",
        "comfort",
        "cushion",
        "soft",
        "lounge",
        "home",
        "recovery",
        "memory foam",
    ],
    "clean_minimal": ["minimal", "minimalist", "clean", "simple", "neutral", "quiet luxury"],
    "summer_outdoor": ["summer", "cooling", "outdoor", "vacation", "travel", "beach", "water"],
    "beauty_self_care": [
        "beauty",
        "skincare",
        "hair",
        "mask",
        "serum",
        "routine",
        "self care",
        "glow",
    ],
    "wellness_functional": [
        "wellness",
        "supplement",
        "fitness",
        "protein",
        "hydration",
        "sleep",
        "recovery",
    ],
    "pet_parent_lifestyle": ["pet", "dog", "cat", "puppy", "kitten", "enrichment"],
    "family_parenting": ["baby", "kids", "mom", "parent", "toddler", "family"],
    "creator_affiliate_ready": [
        "affiliate",
        "commission",
        "free sample",
        "brand deal",
        "creator affiliate",
        "open commission",
    ],
    "live_commerce_ready": ["live commerce", "live trendy", "livestream", "直播", "达播"],
    "viral_review_angle": [
        "viral",
        "trendy",
        "tiktok",
        "review",
        "dupe",
        "worth it",
        "finds",
    ],
}

TREND_LEXICON = {
    "tiktok_shop_affiliate": ["tiktok shop", "creator affiliate", "affiliate", "commission"],
    "strong_video_product": ["video gpm", "video opm", "short video", "达人短视频"],
    "live_selling_product": ["live commerce", "livestream", "直播", "达播", "live trendy"],
    "summer_seasonal": ["summer", "cooling", "vacation", "outdoor", "july", "independence day"],
    "event_seasonal": ["fifa", "world cup", "wedding", "graduation", "father", "juneteenth"],
    "value_deal": ["deal", "discount", "bundle", "gift", "low price", "high roas"],
    "hero_or_key_product": ["hero", "key product", "core", "strong trend", "trendy products"],
    "evergreen_assortment": ["evergreen", "长青", "daily", "routine", "essential"],
}

BOOLEAN_TREND_COLUMNS = [
    "Trendy Products",
    "FIFA Strong related products\n (official authorized)",
    "FIFA World Cup 6/11-7/19",
    "Euphoria 亢奋风",
    "Wedding Season\n婚礼季 (5.1-6.30)",
    "Outdoor Activity",
    "NBA & MLB",
    "Graduation Season 毕业季",
    "Father's Day 6/21",
    "Juneteenth 6/19",
    "National Pet Month 全国宠物月",
    "Summer Vibes",
    "Summer Cooling 夏季降温",
    "Summer Vacation 暑期旅游",
    "Swimwear + Water Sports",
    "Moving Season (May-Aug)",
    "Independence Day (July 4th)  独立日",
    "Live Trendy",
    "六月年中促核心品",
    "是否六月促强趋势货盘",
    "是否六月促长青类目货盘",
    "周促核心品",
    "is_discovery_product",
    "是否站外爆品",
    "TTS-Exclusive Products",
    "长青类目",
]


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    source_name: str
    source_type: str


def _read_csv_with_promoted_header(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if not df.empty and str(df.columns[0]).startswith("Unnamed") and "product_id" in df.iloc[0].astype(str).str.lower().tolist():
        new_cols = [str(x).strip() if str(x).strip() and str(x) != "nan" else f"unnamed_{i}" for i, x in enumerate(df.iloc[0])]
        df = df.iloc[1:].copy()
        df.columns = new_cols
    return df.reset_index(drop=True)


def _read_table(spec: SourceSpec) -> pd.DataFrame:
    if spec.path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(spec.path)
    else:
        df = _read_csv_with_promoted_header(spec.path)
    df["source_file"] = spec.path.name
    df["source_name"] = spec.source_name
    df["source_type"] = spec.source_type
    return df


def _first_present(row: pd.Series, candidates: list[str], default: str = "") -> Any:
    for col in candidates:
        if col in row.index:
            value = row.get(col)
            if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}:
                return value
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    text = str(value).replace(",", "").replace("%", "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _as_flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y", "是", "x"}


def _split_tags(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    pieces = re.split(r"[,;/|，、\n]+", text)
    return [p.strip() for p in pieces if p.strip()]


def _contains_term(normalized: str, term: str) -> bool:
    needle = clean_text(term)
    if not needle:
        return False
    if " " in needle:
        return needle in normalized
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", normalized) is not None


def _match_lexicon(text: str, lexicon: dict[str, list[str]]) -> list[str]:
    normalized = clean_text(text)
    labels = []
    for label, terms in lexicon.items():
        if any(_contains_term(normalized, term) for term in terms):
            labels.append(label)
    return labels


def _boolean_trend_tags(row: pd.Series) -> list[str]:
    tags = []
    for col in BOOLEAN_TREND_COLUMNS:
        if col in row.index and _as_flag(row.get(col)):
            tags.append(col.replace("\n", " ").strip())
    return tags


def _score_product(row: pd.Series, text: str) -> dict[str, float]:
    gmv = _to_float(
        _first_present(
            row,
            [
                "L7D Video GMV",
                "pay_usd_amt_7d",
                "affiliate_live_gmv_30d",
                "Overall GMV",
                "L30D商播GMV",
            ],
        )
    )
    gpm = _to_float(_first_present(row, ["L7D Video GPM", "达播GPM_30d", "Live 30d GPM", "video 30d GPM"]))
    commission = _to_float(_first_present(row, ["open_commission_value", "open_commission_rate"]))
    stock = _to_float(_first_present(row, ["stock_days", "available_quantity"]))
    sample_ready = any(_as_flag(_first_present(row, [col])) for col in ["is_open_freesample", "is_open_free_sample_apply_adjusted", "is_refundable_sample"])
    trendy_flags = len(_boolean_trend_tags(row))

    commercial = min(1.0, math.log1p(max(gmv, 0)) / math.log1p(50000))
    performance = min(1.0, max(gpm, 0) / 50)
    creator_fit = min(1.0, (commission / 5 if commission <= 20 else commission / 100) + (0.15 if sample_ready else 0))
    availability = min(1.0, stock / 30 if stock < 1000 else 1.0)
    trend_signal = min(1.0, trendy_flags / 5 + (0.15 if "tiktok" in clean_text(text) else 0))
    return {
        "commercial_score": round(commercial, 4),
        "performance_score": round(performance, 4),
        "creator_fit_score": round(creator_fit, 4),
        "availability_score": round(availability, 4),
        "trend_signal_score": round(trend_signal, 4),
        "opportunity_score": round(
            commercial * 0.25
            + performance * 0.2
            + creator_fit * 0.2
            + availability * 0.15
            + trend_signal * 0.2,
            4,
        ),
    }


def _build_context(row: pd.Series) -> str:
    fields = [
        "product_name",
        "shop_name",
        "main_theme",
        "sub_theme_en",
        "sub_theme_cn",
        "scenarios_tags",
        "function_tags",
        "style_tags",
        "first_category_name",
        "second_category_name",
        "third_category_name",
        "Vertical Creator Category",
        "Secondary Vertical Creator Category",
        "Vertical Category",
        "Secondary Vertical Category",
        "Team Industry",
    ]
    values = []
    for field in fields:
        if field in row.index and pd.notna(row.get(field)):
            values.append(str(row.get(field)))
    values.extend(_boolean_trend_tags(row))
    return " | ".join(v for v in values if v.strip())


def normalize_product_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in frames:
        for _, row in df.iterrows():
            product_id = str(_first_present(row, ["product_id", "product_id.1"])).strip()
            product_name = str(_first_present(row, ["product_name"])).strip()
            if not product_id or product_id.lower() == "nan" or not product_name:
                continue
            context = _build_context(row)
            scenario_tags = _split_tags(_first_present(row, ["scenarios_tags"]))
            function_tags = _split_tags(_first_present(row, ["function_tags"]))
            style_tags = _split_tags(_first_present(row, ["style_tags"]))
            explicit_trends = _boolean_trend_tags(row)
            vibe_labels = sorted(set(_match_lexicon(context, VIBE_LEXICON) + style_tags[:3]))
            trend_labels = sorted(set(_match_lexicon(context, TREND_LEXICON) + explicit_trends[:8]))
            if row.get("source_type") == "video_product_pool":
                trend_labels.append("video_creator_pool")
            if row.get("source_type") == "live_product_pool":
                vibe_labels.append("live_commerce_ready")
                trend_labels.append("live_selling_product")
            vibe_labels = sorted(set(vibe_labels))
            trend_labels = sorted(set(trend_labels))
            scores = _score_product(row, context)
            category = str(
                _first_present(
                    row,
                    ["first_category_name", "lvl1_product_industry", "Vertical Creator Category", "Vertical Category"],
                )
            )
            subcategory = str(
                _first_present(
                    row,
                    ["second_category_name", "lvl2_product_industry", "Secondary Vertical Creator Category"],
                )
            )
            qa_context = (
                f"Product: {product_name}. Category: {category} / {subcategory}. "
                f"Vibes: {', '.join(vibe_labels[:8]) or 'general commerce product'}. "
                f"Trends: {', '.join(trend_labels[:8]) or 'no explicit trend tag'}. "
                f"Scenarios: {', '.join(scenario_tags[:6])}. "
                f"Functions: {', '.join(function_tags[:6])}. "
                f"Opportunity score: {scores['opportunity_score']}."
            )
            rows.append(
                {
                    "profile_id": stable_hash(product_id, product_name, row.get("source_name", "")),
                    "product_id": product_id,
                    "product_name": product_name,
                    "shop_name": _first_present(row, ["shop_name"]),
                    "source_name": row.get("source_name", ""),
                    "source_type": row.get("source_type", ""),
                    "category": category,
                    "subcategory": subcategory,
                    "third_category": _first_present(row, ["third_category_name"]),
                    "main_theme": _first_present(row, ["main_theme"]),
                    "sub_theme_en": _first_present(row, ["sub_theme_en"]),
                    "sub_theme_cn": _first_present(row, ["sub_theme_cn"]),
                    "scenario_tags": ", ".join(scenario_tags),
                    "function_tags": ", ".join(function_tags),
                    "style_tags": ", ".join(style_tags),
                    "vibe_labels": ", ".join(vibe_labels),
                    "trend_labels": ", ".join(trend_labels),
                    "explicit_trend_tags": ", ".join(explicit_trends),
                    "sale_price": _first_present(row, ["sale_price_1d", "hot_sku_after_voucher_mall_old_user_price_usd"]),
                    "commission_rate": _first_present(row, ["open_commission_rate"]),
                    "commission_value": _first_present(row, ["open_commission_value"]),
                    "gmv_signal": _first_present(
                        row, ["L7D Video GMV", "pay_usd_amt_7d", "affiliate_live_gmv_30d", "Overall GMV"]
                    ),
                    "gpm_signal": _first_present(row, ["L7D Video GPM", "达播GPM_30d", "Live 30d GPM"]),
                    "stock_days": _first_present(row, ["stock_days"]),
                    "detail_url": _first_present(row, ["detail_url"]),
                    "first_image": _first_present(row, ["first_image"]),
                    "business_context_text": context,
                    "qa_context": qa_context,
                    **scores,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("opportunity_score", ascending=False).drop_duplicates(
        ["product_id", "source_name"], keep="first"
    )


def build_taxonomy(profiles: pd.DataFrame) -> pd.DataFrame:
    records = []
    for field, label_type in [
        ("vibe_labels", "vibe"),
        ("trend_labels", "trend"),
        ("category", "category"),
        ("subcategory", "subcategory"),
    ]:
        exploded = profiles[["product_id", field, "opportunity_score"]].copy()
        exploded[field] = exploded[field].fillna("").astype(str).str.split(", ")
        exploded = exploded.explode(field)
        exploded[field] = exploded[field].fillna("").astype(str).str.strip()
        exploded = exploded[exploded[field] != ""]
        for value, group in exploded.groupby(field):
            records.append(
                {
                    "label_type": label_type,
                    "label": value,
                    "product_count": group["product_id"].nunique(),
                    "avg_opportunity_score": round(group["opportunity_score"].mean(), 4),
                    "top_product_ids": ", ".join(group.sort_values("opportunity_score", ascending=False)["product_id"].astype(str).head(8)),
                }
            )
    return pd.DataFrame(records).sort_values(["label_type", "product_count"], ascending=[True, False])


def _assistant_answer(profile: pd.Series, mode: str) -> str:
    vibe = profile.get("vibe_labels", "") or "general shopping-friendly"
    trends = profile.get("trend_labels", "") or "no explicit trend tag"
    product = profile.get("product_name", "")
    category = profile.get("category", "")
    score = profile.get("opportunity_score", "")
    if mode == "vibe":
        return (
            f"{product} fits a {vibe} vibe. It sits in {category} and is best positioned around "
            f"{profile.get('scenario_tags', '') or 'everyday usage'} with functional hooks like "
            f"{profile.get('function_tags', '') or 'clear product benefit'}."
        )
    if mode == "trend":
        return (
            f"The main trend signals are {trends}. The product can be framed through creator-commerce "
            f"content such as demos, honest reviews, comparison angles, and use-case storytelling. "
            f"Opportunity score: {score}."
        )
    return (
        f"Recommended creator angle: match creators whose audience already cares about {category}, "
        f"{vibe}, and practical product discovery. Prioritize short demo videos, before/after use, "
        f"and value proof. Current opportunity score: {score}."
    )


def build_qa_examples(profiles: pd.DataFrame, rules_df: pd.DataFrame, max_products: int = 2500) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    system = (
        "You are a creator-commerce trend intelligence assistant. Answer using product pool facts, "
        "vibe/trend labels, commercial signals, creator fit, and risk-aware reasoning."
    )
    top = profiles.sort_values("opportunity_score", ascending=False).head(max_products)
    for _, profile in top.iterrows():
        product = profile.get("product_name", "")
        metadata = {"product_id": str(profile.get("product_id", "")), "source": profile.get("source_name", "")}
        prompts = [
            (f"What is the vibe of {product}?", _assistant_answer(profile, "vibe")),
            (f"What trend does {product} map to and why?", _assistant_answer(profile, "trend")),
            (f"What creator content angle should we use for {product}?", _assistant_answer(profile, "creator")),
        ]
        for user, assistant in prompts:
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                    "metadata": metadata,
                }
            )

    for _, row in rules_df.iterrows():
        text = " ".join(str(v) for v in row.dropna().tolist())
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 80:
            continue
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "What are the selection rules for affiliate creator product pools?"},
                    {"role": "assistant", "content": text[:3000]},
                ],
                "metadata": {"source": "affiliate_creator_pool_rules"},
            }
        )
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    video_pool: str = VIDEO_POOL_DEFAULT,
    affiliate_rules: str = AFFILIATE_RULES_DEFAULT,
    live_pool: str = LIVE_POOL_DEFAULT,
    output_dir: str = "outputs/vibe_trend_training",
    max_qa_products: int = 2500,
) -> dict[str, Path]:
    specs = [
        SourceSpec(Path(video_pool), "video_spm_key_product_pool_week69", "video_product_pool"),
        SourceSpec(Path(live_pool), "live_product_pool_0527", "live_product_pool"),
    ]
    frames = [_read_table(spec) for spec in specs if spec.path.exists()]
    rules_df = _read_csv_with_promoted_header(Path(affiliate_rules)) if Path(affiliate_rules).exists() else pd.DataFrame()
    profiles = normalize_product_rows(frames)
    taxonomy = build_taxonomy(profiles) if not profiles.empty else pd.DataFrame()
    qa = build_qa_examples(profiles, rules_df, max_products=max_qa_products)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / "product_vibe_trend_profiles.csv"
    taxonomy_path = out / "vibe_trend_taxonomy.csv"
    qa_path = out / "vibe_trend_qa_finetune.jsonl"
    sample_path = out / "vibe_trend_qa_sample.jsonl"
    profiles.to_csv(profile_path, index=False)
    taxonomy.to_csv(taxonomy_path, index=False)
    write_jsonl(qa_path, qa)
    write_jsonl(sample_path, qa[:100])
    return {
        "profiles": profile_path,
        "taxonomy": taxonomy_path,
        "qa_finetune": qa_path,
        "qa_sample": sample_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build product vibe/trend profiles and Q&A JSONL.")
    parser.add_argument("--video-pool", default=VIDEO_POOL_DEFAULT)
    parser.add_argument("--affiliate-rules", default=AFFILIATE_RULES_DEFAULT)
    parser.add_argument("--live-pool", default=LIVE_POOL_DEFAULT)
    parser.add_argument("--output-dir", default="outputs/vibe_trend_training")
    parser.add_argument("--max-qa-products", type=int, default=2500)
    args = parser.parse_args()
    outputs = run(
        video_pool=args.video_pool,
        affiliate_rules=args.affiliate_rules,
        live_pool=args.live_pool,
        output_dir=args.output_dir,
        max_qa_products=args.max_qa_products,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
