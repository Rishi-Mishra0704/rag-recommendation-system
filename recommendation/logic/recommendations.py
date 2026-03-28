"""
recommendation.py — B2B partnership recommendation engine.

Flow:
  1. Fetch source business from DB
  2. LLM reasons about ideal partner characteristics
  3. Embed LLM output → query vector
  4. pgvector cosine search + metadata filters
  5. LLM explains why each match is a good partner

Usage:
    python recommendation.py <business_id>
"""

import json
import sys

import psycopg2
import requests
from pgvector.psycopg2 import register_vector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_recommendation",
    "user": "raguser",
    "password": "password",
}

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "qwen3-embedding"
LLM_MODEL = "llama3.1:8b"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def connect_db():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def get_business_by_id(cursor, business_id: str) -> dict:
    cursor.execute(
        """
        SELECT id, owner_id, verified, reputation, name, industry, sub_industry,
               roles, category, trade_type, location, trade_regions, capacity,
               certificates, tags, description, partner_goals
        FROM businesses
        WHERE id = %s
        """,
        (business_id,),
    )
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Business {business_id} not found")

    columns = [
        "id", "owner_id", "verified", "reputation", "name", "industry",
        "sub_industry", "roles", "category", "trade_type", "location",
        "trade_regions", "capacity", "certificates", "tags", "description",
        "partner_goals",
    ]
    return dict(zip(columns, row))


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def get_llm_response(prompt: str) -> str:
    """Call llama3.1:8b via Ollama generate endpoint."""
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def build_partner_prompt(business: dict) -> str:
    """Ask LLM to describe the ideal partner for this business."""
    return f"""You are a B2B partnership advisor. Given the following business profile, describe what an ideal business partner would look like. Focus on complementary capabilities, industry alignment, trade compatibility, and geographic fit.

Business Profile:
- Name: {business['name']}
- Industry: {business['industry']} ({business['sub_industry']})
- Category: {business['category']}
- Roles: {', '.join(business['roles'])}
- Location: {business['location']}
- Trade Type: {business['trade_type']}
- Trade Regions: {', '.join(business['trade_regions'])}
- Tags: {', '.join(business['tags'])}
- Description: {business['description']}
- Partner Goals: {business['partner_goals']}

Write a 3-4 sentence description of the ideal partner for this business. Focus on what industry they should be in, what role they should play (buyer, seller, supplier, etc.), what trade regions they should operate in, and what capabilities they should have. Be specific and practical."""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def vector_search(
    cursor,
    query_embedding: list[float],
    source_id: str,
    top_k: int = 5,
    trade_type: str = None,
    category: str = None,
    roles: list = None,
) -> list[dict]:
    """
    Cosine similarity search with optional metadata filters.
    Passing None for a filter skips it.
    """
    cursor.execute(
        """
        SELECT id, name, industry, sub_industry, category, trade_type,
               roles, location, trade_regions, tags, description, partner_goals,
               1 - (profile_embedding <=> %s::vector) AS similarity
        FROM businesses
        WHERE id != %s
          AND (%s IS NULL OR trade_type = %s)
          AND (%s IS NULL OR category = %s)
          AND (%s IS NULL OR roles && %s)
        ORDER BY profile_embedding <=> %s::vector
        LIMIT %s
        """,
        (
            query_embedding,
            source_id,
            trade_type, trade_type,
            category, category,
            roles, roles,
            query_embedding,
            top_k,
        ),
    )

    columns = [
        "id", "name", "industry", "sub_industry", "category", "trade_type",
        "roles", "location", "trade_regions", "tags", "description",
        "partner_goals", "similarity",
    ]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Explain matches
# ---------------------------------------------------------------------------

def explain_matches(business: dict, matches: list[dict]) -> list[dict]:
    """Ask LLM to explain why each match is a good partner."""
    for match in matches:
        prompt = f"""You are a B2B partnership advisor. Explain in 2-3 sentences why these two businesses would be good partners.

Business A:
- Name: {business['name']}
- Industry: {business['industry']} ({business['sub_industry']})
- Category: {business['category']}
- Roles: {', '.join(business['roles'])}
- Description: {business['description']}
- Partner Goals: {business['partner_goals']}

Business B:
- Name: {match['name']}
- Industry: {match['industry']} ({match['sub_industry']})
- Category: {match['category']}
- Roles: {', '.join(match['roles'])}
- Description: {match['description']}
- Partner Goals: {match['partner_goals']}

Why are they a good match?"""

        try:
            match["reasoning"] = get_llm_response(prompt)
        except Exception as e:
            match["reasoning"] = f"Could not generate reasoning: {e}"

    return matches


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def recommend(
    business_id: str,
    top_k: int = 5,
    trade_type: str = None,
    category: str = None,
    roles: list = None,
    explain: bool = True,
) -> dict:
    conn = connect_db()
    cursor = conn.cursor()

    try:
        # 1. Fetch source business
        print(f"Fetching business {business_id}...")
        business = get_business_by_id(cursor, business_id)
        print(f"Source: {business['name']} ({business['industry']})")

        # 2. LLM reasons about ideal partner
        print("Generating ideal partner description...")
        partner_prompt = build_partner_prompt(business)
        ideal_partner_text = get_llm_response(partner_prompt)
        print(f"Ideal partner: {ideal_partner_text[:200]}...")

        # 3. Embed the ideal partner description
        print("Embedding ideal partner description...")
        query_embedding = get_embedding(ideal_partner_text)

        # 4. Vector search with optional filters
        print(f"Searching for top {top_k} matches...")
        matches = vector_search(
            cursor, query_embedding, business_id,
            top_k=top_k, trade_type=trade_type,
            category=category, roles=roles,
        )
        print(f"Found {len(matches)} matches.")

        # 5. Explain matches
        if explain and matches:
            print("Generating explanations...")
            matches = explain_matches(business, matches)

        return {
            "source": business,
            "ideal_partner_description": ideal_partner_text,
            "recommendations": matches,
        }

    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Grab a random ID from profiles.json for testing
        import json as j
        from pathlib import Path
        profiles = j.load(open(Path(__file__).resolve().parents[1] / "data" / "profiles.json"))
        bid = profiles[0]["id"]
        print(f"No ID provided, using first profile: {bid}")
    else:
        bid = sys.argv[1]

    results = recommend(bid, top_k=5)

    print("\n" + "=" * 60)
    print(f"RECOMMENDATIONS FOR: {results['source']['name']}")
    print("=" * 60)

    for i, r in enumerate(results["recommendations"], 1):
        print(f"\n--- Match #{i} ---")
        print(f"Name:       {r['name']}")
        print(f"Industry:   {r['industry']} ({r['sub_industry']})")
        print(f"Category:   {r['category']}")
        print(f"Location:   {r['location']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"Reasoning:  {r.get('reasoning', 'N/A')}")