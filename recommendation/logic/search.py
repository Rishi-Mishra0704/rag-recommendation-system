from sentence_transformers import CrossEncoder



# Load once at module level
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



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
# BM25 search
# ---------------------------------------------------------------------------
def bm25_search(
    cursor,
    query_text: str,
    source_id: str,
    top_k: int = 5,
    trade_type: str = None,
    category: str = None,
    roles: list = None,
) -> list[dict]:
    """
    Full-text search using Postgres tsvector (BM25-like ranking via ts_rank).
    Passing None for a filter skips it.
    """
    cursor.execute(
        """
        SELECT id, name, industry, sub_industry, category, trade_type,
               roles, location, trade_regions, tags, description, partner_goals,
               ts_rank(search_text, plainto_tsquery('english', %s)) AS rank_score
        FROM businesses
        WHERE id != %s
          AND search_text @@ plainto_tsquery('english', %s)
          AND (%s IS NULL OR trade_type = %s)
          AND (%s IS NULL OR category = %s)
          AND (%s IS NULL OR roles && %s)
        ORDER BY ts_rank(search_text, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (
            query_text,
            source_id,
            query_text,
            trade_type, trade_type,
            category, category,
            roles, roles,
            query_text,
            top_k,
        ),
    )

    columns = [
        "id", "name", "industry", "sub_industry", "category", "trade_type",
        "roles", "location", "trade_regions", "tags", "description",
        "partner_goals", "rank_score",
    ]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# RRF merging
# ---------------------------------------------------------------------------

def rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5,
    k: int = 60,  # RRF constant
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    Score = sum(1 / (k + rank)) across lists.
    Lower rank (closer to 1) = higher contribution.

    Assumes each result has a unique 'id'.
    """

    scores = {}
    result_map = {}

    # Process vector results
    for rank, item in enumerate(vector_results, start=1):
        doc_id = item["id"]
        score = 1 / (k + rank)

        scores[doc_id] = scores.get(doc_id, 0) + score
        result_map[doc_id] = item  # keep latest (or first, doesn't matter much)

    # Process BM25 results
    for rank, item in enumerate(bm25_results, start=1):
        doc_id = item["id"]
        score = 1 / (k + rank)

        scores[doc_id] = scores.get(doc_id, 0) + score

        # Prefer richer object if not already present
        if doc_id not in result_map:
            result_map[doc_id] = item

    # Sort by combined score
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Build final results
    merged_results = []
    for doc_id, score in ranked[:top_k]:
        item = result_map[doc_id].copy()
        item["rrf_score"] = score
        merged_results.append(item)

    return merged_results




# ---------------------------------------------------------------------------
# Reranking with cross-encoder
# ---------------------------------------------------------------------------

def _build_candidate_text(candidate: dict) -> str:
    """Combine candidate fields into a single text block for reranking."""
    roles = ", ".join(candidate.get("roles", []))
    tags = ", ".join(candidate.get("tags", []))
    trade_regions = ", ".join(candidate.get("trade_regions", []))

    return (
        f"{candidate.get('description', '')} "
        f"Industry: {candidate.get('industry', '')} "
        f"Sub-industry: {candidate.get('sub_industry', '')} "
        f"Roles: {roles} "
        f"Tags: {tags} "
        f"Trade regions: {trade_regions} "
        f"Partner goals: {candidate.get('partner_goals', '')}"
    )


def rerank(query_text: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates using a cross-encoder.
    Takes query + candidate pairs, scores them together, returns top_k.
    """
    if not candidates:
        return []

    pairs = [(query_text, _build_candidate_text(c)) for c in candidates]
    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]