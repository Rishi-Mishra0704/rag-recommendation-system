from dataclasses import dataclass

# (query_trade_type, result_trade_type) → compatible
TRADE_TYPE_COMPAT = {
    ("domestic",      "domestic"):      True,
    ("domestic",      "international"): False,
    ("domestic",      "both"):          True,
    ("international", "domestic"):      False,
    ("international", "international"): True,
    ("international", "both"):          True,
    ("both",          "domestic"):      True,
    ("both",          "international"): True,
    ("both",          "both"):          True,
}

# Complementary roles: role → set of roles in a match that would be a good fit
ROLE_COMPAT = {
    "buyer":    {"seller", "exporter", "agent"},
    "seller":   {"buyer", "importer", "agent", "reseller"},
    "importer": {"exporter", "agent"},
    "exporter": {"importer", "buyer", "agent"},
    "agent":    {"buyer", "seller", "importer", "exporter", "reseller"},
    "reseller": {"seller", "agent"},
}


@dataclass
class HeuristicScores:
    industry_match = 0.0
    trade_region_overlap = 0.0
    trade_type_compat = 0.0
    role_compat = 0.0
    category_match = 0.0

    def __init__(self, industry_match=0.0, trade_region_overlap=0.0,
                 trade_type_compat=0.0, role_compat=0.0, category_match=0.0):
        self.industry_match = industry_match
        self.trade_region_overlap = trade_region_overlap
        self.trade_type_compat = trade_type_compat
        self.role_compat = role_compat
        self.category_match = category_match


# ---------------------------------------------------------------------------
# Heuristic scoring helpers
# ---------------------------------------------------------------------------

def score_industry_match(source, matches):
    if not matches:
        return 0.0
    hits = sum(
        1 for m in matches
        if m["industry"] == source["industry"] or m["sub_industry"] == source["sub_industry"]
    )
    return hits / len(matches)


def score_trade_region_overlap(source, matches):
    if not matches:
        return 0.0
    src_regions = set(source.get("trade_regions") or [])
    if not src_regions:
        return 0.0
    hits = sum(
        1 for m in matches
        if src_regions & set(m.get("trade_regions") or [])
    )
    return hits / len(matches)


def score_trade_type_compat(source, matches):
    if not matches:
        return 0.0
    src_tt = (source.get("trade_type") or "").lower()
    hits = sum(
        1 for m in matches
        if TRADE_TYPE_COMPAT.get((src_tt, (m.get("trade_type") or "").lower()), True)
    )
    return hits / len(matches)


def score_role_compat(source, matches):
    if not matches:
        return 0.0
    src_roles = {r.lower() for r in (source.get("roles") or [])}
    hits = 0
    for m in matches:
        m_roles = {r.lower() for r in (m.get("roles") or [])}
        if any(m_roles & ROLE_COMPAT.get(sr, set()) for sr in src_roles):
            hits += 1
    return hits / len(matches)


def score_category_match(source, matches):
    if not matches:
        return 0.0
    src_cat = (source.get("category") or "").lower()
    hits = sum(
        1 for m in matches
        if (m.get("category") or "").lower() == src_cat
    )
    return hits / len(matches)


def compute_heuristics(source, matches):
    return HeuristicScores(
        industry_match=score_industry_match(source, matches),
        trade_region_overlap=score_trade_region_overlap(source, matches),
        trade_type_compat=score_trade_type_compat(source, matches),
        role_compat=score_role_compat(source, matches),
        category_match=score_category_match(source, matches),
    )


# ---------------------------------------------------------------------------
# Ground-truth metrics
# ---------------------------------------------------------------------------

def precision_at_k(top_k_ids, expected):
    if not top_k_ids or not expected:
        return 0.0
    expected_set = set(expected)
    return sum(1 for i in top_k_ids if i in expected_set) / len(top_k_ids)


def recall_at_k(top_k_ids, expected):
    if not expected:
        return 0.0
    expected_set = set(expected)
    return sum(1 for i in top_k_ids if i in expected_set) / len(expected_set)
