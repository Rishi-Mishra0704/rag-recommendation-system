"""validate.py - Validate profiles.json against the Business Pydantic model.

Usage:
    python validate.py
    python validate.py --input data/profiles.json --output data/profiles_validated.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from recommendation.data_gen.schemas import (
    CERTIFICATES,
    SUB_INDUSTRY_MAP,
    TRADE_REGIONS,
    Business,
    Category,
    Industry,
    Role,
    TradeType,
)

# Valid string sets derived from the (possibly plain) class attributes
VALID_INDUSTRIES = {v for v in vars(Industry).values() if isinstance(v, str) and not v.startswith("_")}
VALID_CATEGORIES = {v for v in vars(Category).values() if isinstance(v, str) and not v.startswith("_")}
VALID_TRADE_TYPES = {v for v in vars(TradeType).values() if isinstance(v, str) and not v.startswith("_")}
VALID_ROLES = {v for v in vars(Role).values() if isinstance(v, str) and not v.startswith("_")}
VALID_TRADE_REGIONS = set(TRADE_REGIONS)
VALID_CERTIFICATES = set(CERTIFICATES)

# SUB_INDUSTRY_MAP keys may be enum members or plain strings; normalise to strings
SUB_INDUSTRY_MAP_STR: dict[str, list[str]] = {
    (k.value if hasattr(k, "value") else k): v for k, v in SUB_INDUSTRY_MAP.items()
}


def _check(profile: dict) -> list[str]:
    """Return a list of error strings. Empty list means the profile is valid."""
    errors: list[str] = []

    # --- Pydantic structural validation ---
    try:
        biz = Business(**profile)
    except ValidationError as exc:
        return [f"pydantic: {e['loc']} - {e['msg']}" for e in exc.errors()]

    # --- Enum field values ---
    if biz.industry not in VALID_INDUSTRIES:
        errors.append(f"industry '{biz.industry}' not in valid set")

    if biz.category not in VALID_CATEGORIES:
        errors.append(f"category '{biz.category}' not in valid set")

    if biz.trade_type not in VALID_TRADE_TYPES:
        errors.append(f"trade_type '{biz.trade_type}' not in valid set")

    invalid_roles = [r for r in biz.roles if r not in VALID_ROLES]
    if invalid_roles:
        errors.append(f"invalid roles: {invalid_roles}")

    # --- sub_industry must belong to its industry's map ---
    valid_subs = SUB_INDUSTRY_MAP_STR.get(biz.industry, [])
    if biz.sub_industry not in valid_subs:
        errors.append(
            f"sub_industry '{biz.sub_industry}' not valid for industry '{biz.industry}' "
            f"(expected one of {valid_subs})"
        )

    # --- trade_regions ---
    invalid_regions = [r for r in biz.trade_regions if r not in VALID_TRADE_REGIONS]
    if invalid_regions:
        errors.append(f"invalid trade_regions: {invalid_regions}")

    # --- certificates ---
    invalid_certs = [c for c in biz.certificates if c not in VALID_CERTIFICATES]
    if invalid_certs:
        errors.append(f"invalid certificates: {invalid_certs}")

    # --- reputation ---
    if not (0.0 <= biz.reputation <= 5.0):
        errors.append(f"reputation {biz.reputation} out of range [0.0, 5.0]")

    # --- description / partner_goals ---
    if not biz.description or len(biz.description.strip()) < 20:
        errors.append("description missing or too short (<20 chars)")

    if not biz.partner_goals or len(biz.partner_goals.strip()) < 20:
        errors.append("partner_goals missing or too short (<20 chars)")

    # --- tags ---
    if not biz.tags:
        errors.append("tags list is empty")

    return errors


def run(input_path: str, output_path: str) -> None:
    src = Path(input_path)
    if not src.exists():
        print(f"ERROR: input file not found: {src}")
        sys.exit(1)

    with open(src, encoding="utf-8") as f:
        profiles: list[dict] = json.load(f)

    total = len(profiles)
    valid_profiles: list[dict] = []
    invalid_count = 0

    for i, profile in enumerate(profiles):
        name = profile.get("name", f"<profile #{i}>")
        errors = _check(profile)
        if errors:
            invalid_count += 1
            print(f"  INVALID [{i}] {name}")
            for err in errors:
                print(f"    - {err}")
        else:
            valid_profiles.append(profile)

    valid_count = len(valid_profiles)

    print()
    print("-" * 50)
    print(f"Total profiles  : {total}")
    print(f"Valid           : {valid_count}")
    print(f"Invalid         : {invalid_count}")
    print("-" * 50)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(valid_profiles, f, indent=2, ensure_ascii=False)

    print(f"Saved {valid_count} valid profiles -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate profiles.json against the Business model.")
    parser.add_argument("--input", default="data/profiles.json", help="Input JSON path")
    parser.add_argument("--output", default="data/profiles_validated.json", help="Output JSON path")
    args = parser.parse_args()
    run(args.input, args.output)
