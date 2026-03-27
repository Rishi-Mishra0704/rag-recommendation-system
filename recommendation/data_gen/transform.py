from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests

# Allow running from repo root or from this file's directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

# ---------------------------------------------------------------------------
# Tag map: deterministic tags per industry
# ---------------------------------------------------------------------------
INDUSTRY_TAGS: dict[Industry, list[str]] = {
    Industry.AGRICULTURE: ["agriculture", "farming", "crop", "agri-trade", "harvest", "organic"],
    Industry.AUTOMOTIVE: ["automotive", "spare-parts", "vehicles", "oem", "aftermarket", "fleet"],
    Industry.CHEMICALS: ["chemicals", "industrial", "specialty-chemicals", "b2b", "raw-materials", "processing"],
    Industry.CONSTRUCTION: ["construction", "building", "infrastructure", "materials", "contractor", "civil"],
    Industry.ELECTRONICS: ["electronics", "telecom", "components", "hardware", "pcb", "semiconductors"],
    Industry.ENERGY: ["energy", "oil-gas", "renewables", "power", "solar", "utilities"],
    Industry.FOOD_BEVERAGE: ["food", "beverages", "fmcg", "packaged-goods", "supply-chain", "ingredients"],
    Industry.HEALTHCARE: ["healthcare", "medical", "devices", "diagnostics", "hospital", "wellness"],
    Industry.LOGISTICS: ["logistics", "freight", "supply-chain", "shipping", "warehousing", "trade"],
    Industry.MANUFACTURING: ["manufacturing", "factory", "production", "industrial", "oem", "b2b"],
    Industry.METALS_MINING: ["metals", "mining", "steel", "aluminum", "raw-materials", "commodities"],
    Industry.PHARMACEUTICALS: ["pharma", "drugs", "biologics", "api", "nutraceuticals", "biotech"],
    Industry.RETAIL: ["retail", "wholesale", "distribution", "fmcg", "consumer-goods", "e-commerce"],
    Industry.TEXTILE_APPAREL: ["textiles", "apparel", "fabric", "garments", "fashion", "yarn"],
    Industry.TECHNOLOGY: ["technology", "software", "saas", "it-services", "digital", "cloud"],
    Industry.WOOD_FURNITURE: ["wood", "furniture", "timber", "panels", "flooring", "interior"],
}

# ---------------------------------------------------------------------------
# Country → trade regions mapping
# ---------------------------------------------------------------------------
COUNTRY_REGION_MAP: dict[str, list[str]] = {
    "united states": ["North America"],
    "canada": ["North America"],
    "mexico": ["North America", "South America"],
    "brazil": ["South America"],
    "argentina": ["South America"],
    "colombia": ["South America"],
    "chile": ["South America"],
    "united kingdom": ["Western Europe"],
    "germany": ["Western Europe"],
    "france": ["Western Europe"],
    "italy": ["Western Europe"],
    "spain": ["Western Europe"],
    "netherlands": ["Western Europe"],
    "sweden": ["Western Europe"],
    "norway": ["Western Europe"],
    "denmark": ["Western Europe"],
    "switzerland": ["Western Europe"],
    "austria": ["Western Europe"],
    "belgium": ["Western Europe"],
    "portugal": ["Western Europe"],
    "poland": ["Eastern Europe"],
    "ukraine": ["Eastern Europe"],
    "czech republic": ["Eastern Europe"],
    "romania": ["Eastern Europe"],
    "hungary": ["Eastern Europe"],
    "russia": ["Eastern Europe", "Central Asia"],
    "turkey": ["Middle East", "Western Europe"],
    "saudi arabia": ["Middle East"],
    "united arab emirates": ["Middle East"],
    "israel": ["Middle East"],
    "iran": ["Middle East"],
    "iraq": ["Middle East"],
    "egypt": ["Middle East", "Africa"],
    "south africa": ["Africa"],
    "nigeria": ["Africa"],
    "kenya": ["Africa"],
    "ghana": ["Africa"],
    "ethiopia": ["Africa"],
    "india": ["South Asia", "India"],
    "pakistan": ["South Asia"],
    "bangladesh": ["South Asia"],
    "sri lanka": ["South Asia"],
    "indonesia": ["Southeast Asia"],
    "malaysia": ["Southeast Asia"],
    "thailand": ["Southeast Asia"],
    "vietnam": ["Southeast Asia"],
    "philippines": ["Southeast Asia"],
    "singapore": ["Southeast Asia"],
    "china": ["East Asia"],
    "japan": ["East Asia"],
    "south korea": ["East Asia"],
    "taiwan": ["East Asia"],
    "hong kong": ["East Asia"],
    "australia": ["Oceania"],
    "new zealand": ["Oceania"],
    "kazakhstan": ["Central Asia"],
    "uzbekistan": ["Central Asia"],
}


def country_to_regions(country: str) -> list[str]:
    key = country.strip().lower()
    return COUNTRY_REGION_MAP.get(key, ["Global"])


# ---------------------------------------------------------------------------
# Ollama LLM generation
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


def _build_prompt(biz: dict) -> str:
    return (
        f"You are writing a B2B company profile for a trade platform.\n\n"
        f"Company: {biz['name']}\n"
        f"Industry: {biz['industry']} > {biz['sub_industry']}\n"
        f"Category: {biz['category']}\n"
        f"Location: {biz['location']}\n"
        f"Roles: {', '.join(biz['roles'])}\n"
        f"Tags: {', '.join(biz['tags'])}\n\n"
        f"Write exactly two sections:\n"
        f"DESCRIPTION: A 2-3 sentence factual description of the company's business and products.\n"
        f"PARTNER_GOALS: A 2-3 sentence statement of what kind of B2B partners this company is seeking.\n\n"
        f"Use plain text, no markdown, no bullet points."
    )


def _call_ollama(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def _parse_llm_response(text: str) -> tuple[str, str]:
    description, partner_goals = "", ""
    for line in text.splitlines():
        if line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PARTNER_GOALS:"):
            partner_goals = line.split(":", 1)[1].strip()
    # Fallback: split on the keyword if single-line response
    if not description and "PARTNER_GOALS:" in text.upper():
        idx = text.upper().index("PARTNER_GOALS:")
        description = text[:idx].replace("DESCRIPTION:", "").strip()
        partner_goals = text[idx:].split(":", 1)[1].strip()
    return description, partner_goals


def _fallback_texts(biz: dict) -> tuple[str, str]:
    description = (
        f"{biz['name']} is a {biz['category']} operating in the {biz['sub_industry']} "
        f"segment of the {biz['industry']} industry, based in {biz['location']}. "
        f"The company focuses on delivering quality products and services to B2B clients worldwide."
    )
    partner_goals = (
        f"{biz['name']} is looking for reliable {', '.join(biz['roles'])} partners "
        f"in the {biz['industry']} space. "
        f"We seek long-term trade relationships with companies that share our commitment to quality and efficiency."
    )
    return description, partner_goals


def generate_texts(biz: dict, retries: int = 3) -> tuple[str, str]:
    prompt = _build_prompt(biz)
    for attempt in range(retries):
        raw = _call_ollama(prompt)
        if raw:
            desc, goals = _parse_llm_response(raw)
            if desc and goals:
                return desc, goals
        if attempt < retries - 1:
            time.sleep(1)
    return _fallback_texts(biz)


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
