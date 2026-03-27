"""
embed_service.py — Generate embeddings for every business profile and insert
them into the businesses table.

Usage:
    python embed_service.py
"""

import json
import os
import sys
from pathlib import Path
import time
import psycopg2
import requests
from pgvector.psycopg2 import register_vector
from constants.constants import OLLAMA_EMBED_URL, EMBED_MODEL, DB_CONFIG, INSERT_SQL
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILES_PATH = Path(__file__).resolve().parents[2] / "data" / "profiles.json"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_profiles(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_block(profile: dict) -> str:
    roles         = ", ".join(profile.get("roles", []))
    tags          = ", ".join(profile.get("tags", []))
    trade_regions = ", ".join(profile.get("trade_regions", []))

    parts = [
        profile.get("description", ""),
        f"Industry: {profile.get('industry', '')}",
        f"Sub-industry: {profile.get('sub_industry', '')}",
        f"Roles: {roles}",
        f"Tags: {tags}",
        f"Trade regions: {trade_regions}",
        f"Partner goals: {profile.get('partner_goals', '')}",
    ]
    return "\n".join(p for p in parts if p)


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    # Ollama /api/embed returns {"embeddings": [[...]], ...}
    return data["embeddings"][0]


def connect_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def insert_business(cursor, profile: dict, embedding: list[float]) -> None:
    cursor.execute(
        INSERT_SQL,
        (
            profile["id"],
            profile["owner_id"],
            profile["verified"],
            profile["reputation"],
            profile["name"],
            profile["industry"],
            profile["sub_industry"],
            profile.get("roles", []),
            profile["category"],
            profile["trade_type"],
            profile["location"],
            profile.get("trade_regions", []),
            profile.get("capacity"),
            profile.get("certificates", []),
            profile.get("tags", []),
            profile["description"],
            profile["partner_goals"],
            embedding,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    profiles = load_profiles(PROFILES_PATH)
    total = len(profiles)
    print(f"Loaded {total} profiles from {PROFILES_PATH}")

    conn = connect_db()
    cursor = conn.cursor()

    start = time.time()
    try:
        for i, profile in enumerate(profiles, start=1):
            text_block = build_text_block(profile)
            embedding  = get_embedding(text_block)
            insert_business(cursor, profile, embedding)
            conn.commit()
            elapsed = time.time() - start
            avg = elapsed / i
            remaining = avg * (total - i)
            print(f"  [{i}/{total}] inserted: {profile['name']} ({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")
    except Exception as exc:
        conn.rollback()
        print(f"\nERROR at profile {i}: {exc}", file=sys.stderr)
        raise
    finally:
        cursor.close()
        conn.close()

    total_time = time.time() - start
    print(f"\nDone — {total} businesses embedded and inserted in {total_time:.1f}s.")



if __name__ == "__main__":
    main()
