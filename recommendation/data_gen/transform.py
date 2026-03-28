from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

from recommendation.data_gen.schemas import (
    CERTIFICATES,
    INDUSTRY_MAPPING,
    SUB_INDUSTRY_MAP,
    TRADE_REGIONS,
    Business,
    Category,
    Industry,
    Role,
    TradeType,
)
from recommendation.data_gen.mappings import INDUSTRY_TAGS, country_to_regions
from recommendation.data_gen.llm_gen import generate_texts


# ---------------------------------------------------------------------------
# Sampling with industry diversity
# ---------------------------------------------------------------------------
def diverse_sample(df: pd.DataFrame, count: int, min_per_industry: int = 5) -> pd.DataFrame:
    groups = {ind: grp for ind, grp in df.groupby("mapped_industry")}

    # Guarantee min_per_industry for each group that has enough rows
    guaranteed: list[pd.DataFrame] = []
    for ind, grp in groups.items():
        n = min(min_per_industry, len(grp))
        guaranteed.append(grp.sample(n=n, random_state=42))

    guaranteed_df = pd.concat(guaranteed)
    remaining_count = count - len(guaranteed_df)

    if remaining_count <= 0:
        return guaranteed_df.sample(n=count, random_state=42)

    # Fill the rest from rows not already selected
    already_idx = set(guaranteed_df.index)
    pool = df[~df.index.isin(already_idx)]
    if len(pool) >= remaining_count:
        extra = pool.sample(n=remaining_count, random_state=42)
    else:
        extra = pool
    return pd.concat([guaranteed_df, extra]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Row → Business dict
# ---------------------------------------------------------------------------
def transform_row(row: pd.Series) -> dict:
    industry: Industry = INDUSTRY_MAPPING[row["industry"].strip().lower()]

    sub_industry = random.choice(SUB_INDUSTRY_MAP[industry])
    city = str(row["locality"]).split(",")[0].strip().title()
    location = f"{city}, {str(row['country']).title()}"
    base_regions = country_to_regions(str(row["country"]))
    extra_regions = random.sample(
        [r for r in TRADE_REGIONS if r not in base_regions],
        k=random.randint(0, 2),
    )
    trade_regions = list(dict.fromkeys(base_regions + extra_regions))

    _all_roles = [Role.BUYER, Role.SELLER, Role.IMPORTER, Role.EXPORTER, Role.AGENT, Role.RESELLER]
    roles = random.sample(_all_roles, k=random.randint(1, 3))
    certificates = random.sample(CERTIFICATES, k=random.randint(0, 3))

    all_tags = INDUSTRY_TAGS[industry].copy()
    sub_tag = sub_industry.lower().replace(" ", "-").replace("&", "and")
    if sub_tag not in all_tags:
        all_tags.append(sub_tag)
    tags = random.sample(all_tags, k=min(random.randint(3, 6), len(all_tags)))

    capacity = str(row.get("size range", "11-50")) if pd.notna(row.get("size range")) else "11-50"

    partial: dict = {
        "id": str(uuid4()),
        "owner_id": str(uuid4()),
        "verified": random.random() < 0.70,
        "reputation": round(random.uniform(2.5, 5.0), 1),
        "name": str(row["name"]).title(),
        "industry": industry,
        "sub_industry": sub_industry,
        "roles": roles,
        "category": random.choice([
            Category.MANUFACTURER, Category.TRADER, Category.DISTRIBUTOR,
            Category.SERVICE_PROVIDER, Category.SUPPLIER, Category.BROKER, Category.WHOLESALER,
        ]),
        "trade_type": random.choice([TradeType.DOMESTIC, TradeType.INTERNATIONAL, TradeType.BOTH]),
        "location": location,
        "trade_regions": trade_regions,
        "capacity": capacity,
        "certificates": certificates,
        "tags": tags,
    }

    desc, goals = generate_texts(partial)
    partial["description"] = desc
    partial["partner_goals"] = goals
    return partial


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(csv_path: str, count: int, output: str) -> None:
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize industry column
    df["industry"] = df["industry"].astype(str).str.strip().str.lower()

    # Drop rows missing required fields
    df = df.dropna(subset=["name", "locality", "country"])
    df = df[df["industry"] != "nan"]

    # Filter to mapped industries
    df["mapped_industry"] = df["industry"].map(INDUSTRY_MAPPING)
    df = df[df["mapped_industry"].notna()].copy()
    print(f"Rows after industry filter: {len(df)}")

    if len(df) == 0:
        print("No rows matched INDUSTRY_MAPPING. Exiting.")
        sys.exit(1)

    sample_df = diverse_sample(df, count)
    print(f"Sampled {len(sample_df)} rows across {sample_df['mapped_industry'].nunique()} industries")

    profiles: list[dict] = []
    for i, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"  [{i}/{len(sample_df)}] {str(row['name'])[:40]:<40}  {row['industry']}")
        try:
            record = transform_row(row)
            # Validate with Pydantic
            biz = Business(**record)
            profiles.append(biz.to_generation_dict())
        except Exception as exc:
            print(f"    SKIPPED: {exc}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(profiles)} profiles → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform PDL companies CSV into Business profiles.")
    parser.add_argument("--csv", default="data/companies_sorted.csv", help="Path to input CSV")
    parser.add_argument("--count", type=int, default=200, help="Number of profiles to generate")
    parser.add_argument("--output", default="data/profiles.json", help="Output JSON path")
    args = parser.parse_args()

    random.seed(42)
    run(args.csv, args.count, args.output)
