# Vibe & Trend Training Data Builder

This module prepares product-pool knowledge for later Q&A fine-tuning and retrieval.

It does not call an external LLM. It uses deterministic NLP/rule extraction first so the output is repeatable and auditable.

## Inputs

- Video SPM key product pool CSV
- Affiliate creator product pool rule/readme CSV
- Live product pool XLSX

Default local paths are configured in:

`src/trend_copilot_data/training/vibe_trend_builder.py`

## Workflow

```text
Product pool / live pool / selection rules
        ↓
Normalize product rows
        ↓
Combine product name, category, theme, scenario, function, style, tags
        ↓
Infer vibe labels
        ↓
Infer trend labels
        ↓
Compute opportunity score
        ↓
Write product profiles + taxonomy + Q&A JSONL
```

## NLP / Labeling Logic

The builder uses:

- lexicon-based vibe matching
- lexicon-based trend matching
- explicit campaign/trend flag extraction
- product performance and creator-readiness scoring
- deterministic Q&A generation

Current vibe labels include:

- `cozy_comfort`
- `clean_minimal`
- `summer_outdoor`
- `beauty_self_care`
- `wellness_functional`
- `pet_parent_lifestyle`
- `family_parenting`
- `creator_affiliate_ready`
- `live_commerce_ready`
- `viral_review_angle`

Current trend labels include:

- `tiktok_shop_affiliate`
- `strong_video_product`
- `live_selling_product`
- `summer_seasonal`
- `event_seasonal`
- `value_deal`
- `hero_or_key_product`
- `evergreen_assortment`

## Outputs

Default output directory:

`outputs/vibe_trend_training/`

Files:

- `product_vibe_trend_profiles.csv`
- `vibe_trend_taxonomy.csv`
- `vibe_trend_qa_finetune.jsonl`
- `vibe_trend_qa_sample.jsonl`

## Run

```bash
PYTHONPATH=src python3 scripts/build_vibe_trend_training.py
```

If the system Python does not have `openpyxl`, install project dependencies first:

```bash
python3 -m pip install -e .
```

