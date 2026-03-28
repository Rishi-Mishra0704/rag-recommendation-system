# eval.py - Evaluation metrics and pipeline for recommendation quality

import contextlib
import io
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.recommendations import connect_db, recommend

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K = 5
TARGET_EVAL_COUNT = 20
EVAL_DIR = Path(__file__).resolve().parent
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"

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

# Heuristic metric thresholds
# direction "high"    → avg >= threshold is a pass
# direction "low"     → avg <= threshold is a pass
# direction "neutral" → no pass/fail
METRIC_THRESHOLDS = {
    "industry_match":       {"direction": "high",    "threshold": 0.7},
    "trade_region_overlap": {"direction": "neutral", "threshold": None},
    "trade_type_compat":    {"direction": "high",    "threshold": 0.7},
    "role_compat":          {"direction": "high",    "threshold": 0.8},
    "category_match":       {"direction": "low",     "threshold": 0.3},
}

# Filter correctness test definitions
FILTER_TESTS = [
    {"name": "trade_type=domestic",      "filter_kwargs": {"trade_type": "domestic"},      "field": "trade_type", "value": "domestic"},
    {"name": "trade_type=international", "filter_kwargs": {"trade_type": "international"}, "field": "trade_type", "value": "international"},
    {"name": "category=manufacturer",    "filter_kwargs": {"category": "manufacturer"},    "field": "category",   "value": "manufacturer"},
    {"name": "category=distributor",     "filter_kwargs": {"category": "distributor"},     "field": "category",   "value": "distributor"},
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


@dataclass
class QueryEvalResult:
    business_id = ""
    business_name = ""
    industry = ""
    category = ""
    trade_type = ""
    latency_s = 0.0
    heuristic = None
    precision_at_k = None   # None when no expected_results
    recall_at_k = None
    top_k_ids = None
    top_k_reasoning = None

    def __init__(self, business_id, business_name, industry, category,
                 trade_type, latency_s, heuristic, precision_at_k, recall_at_k,
                 top_k_ids, top_k_reasoning):
        self.business_id = business_id
        self.business_name = business_name
        self.industry = industry
        self.category = category
        self.trade_type = trade_type
        self.latency_s = latency_s
        self.heuristic = heuristic
        self.precision_at_k = precision_at_k
        self.recall_at_k = recall_at_k
        self.top_k_ids = top_k_ids or []
        self.top_k_reasoning = top_k_reasoning or []


@dataclass
class FilterTestResult:
    name = ""
    business_id = ""
    passed = False
    violations = 0
    total = 0

    def __init__(self, name, business_id, passed, violations, total):
        self.name = name
        self.business_id = business_id
        self.passed = passed
        self.violations = violations
        self.total = total


# ---------------------------------------------------------------------------
# Database helpers (sampling only — recommend() manages its own connection)
# ---------------------------------------------------------------------------

def sample_diverse_businesses(conn, n):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT industry, category, trade_type
        FROM businesses
        ORDER BY industry, category, trade_type
    """)
    combos = cursor.fetchall()
    random.shuffle(combos)

    selected = []
    seen_ids = []
    cols = ["id", "name", "industry", "sub_industry", "category",
            "trade_type", "roles", "location", "trade_regions"]

    for industry, category, trade_type in combos:
        if len(selected) >= n:
            break
        cursor.execute("""
            SELECT id, name, industry, sub_industry, category, trade_type,
                   roles, location, trade_regions
            FROM businesses
            WHERE industry = %s AND category = %s AND trade_type = %s
            ORDER BY RANDOM()
            LIMIT 1
        """, (industry, category, trade_type))
        row = cursor.fetchone()
        if row and str(row[0]) not in seen_ids:
            seen_ids.append(str(row[0]))
            selected.append(dict(zip(cols, row)))

    # Fill any remaining slots with random businesses
    if len(selected) < n:
        remaining = n - len(selected)
        excl = seen_ids if seen_ids else ["00000000-0000-0000-0000-000000000000"]
        cursor.execute("""
            SELECT id, name, industry, sub_industry, category, trade_type,
                   roles, location, trade_regions
            FROM businesses
            WHERE NOT (id = ANY(%s::uuid[]))
            ORDER BY RANDOM()
            LIMIT %s
        """, (excl, remaining))
        for row in cursor.fetchall():
            selected.append(dict(zip(cols, row)))

    cursor.close()
    return selected[:n]


# ---------------------------------------------------------------------------
# Eval input construction
# ---------------------------------------------------------------------------

def is_valid_uuid(s):
    try:
        uuid.UUID(str(s))
        return True
    except (ValueError, AttributeError):
        return False


def load_test_queries():
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_inputs(conn):
    raw = load_test_queries()
    valid = [
        {"business_id": q["query"], "expected_results": q.get("expected_results", [])}
        for q in raw
        if is_valid_uuid(q.get("query", ""))
    ]

    if len(valid) < 15:
        needed = TARGET_EVAL_COUNT - len(valid)
        existing_ids = {v["business_id"] for v in valid}
        samples = sample_diverse_businesses(conn, needed + 5)  # oversample then trim
        for s in samples:
            sid = str(s["id"])
            if sid not in existing_ids and len(valid) < TARGET_EVAL_COUNT:
                valid.append({"business_id": sid, "expected_results": []})
                existing_ids.add(sid)

    return valid


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


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(eval_inputs):
    results = []
    total = len(eval_inputs)

    for i, entry in enumerate(eval_inputs, 1):
        bid = entry["business_id"]
        expected = entry.get("expected_results") or []
        print(f"  [{i:>2}/{total}] {bid} ...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = recommend(bid, top_k=TOP_K, explain=True)
        except Exception as exc:
            print(f"ERROR — {exc}")
            continue
        latency = time.perf_counter() - t0

        source = result["source"]
        matches = result["recommendations"]
        top_k_ids = [str(m["id"]) for m in matches]
        top_k_reasoning = [m.get("reasoning", "") for m in matches]

        heuristic = compute_heuristics(source, matches)
        prec = precision_at_k(top_k_ids, expected) if expected else None
        rec = recall_at_k(top_k_ids, expected) if expected else None

        results.append(QueryEvalResult(
            business_id=bid,
            business_name=source["name"],
            industry=source["industry"],
            category=source["category"],
            trade_type=source["trade_type"],
            latency_s=round(latency, 3),
            heuristic=heuristic,
            precision_at_k=prec,
            recall_at_k=rec,
            top_k_ids=top_k_ids,
            top_k_reasoning=top_k_reasoning,
        ))
        print(f"ok  ({latency:.2f}s)")

    return results


# ---------------------------------------------------------------------------
# Filter correctness tests
# ---------------------------------------------------------------------------

def run_filter_tests(conn):
    results = []
    cursor = conn.cursor()

    for test in FILTER_TESTS:
        cursor.execute("SELECT id FROM businesses ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        if not row:
            continue
        business_id = str(row[0])

        print(f"  {test['name']} (query={business_id[:8]}...) ...", end=" ", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = recommend(business_id, top_k=TOP_K, explain=False, **test["filter_kwargs"])
            recs = result["recommendations"]
            field_name = test["field"]
            expected_val = test["value"].lower()
            violations = sum(
                1 for r in recs
                if (r.get(field_name) or "").lower() != expected_val
            )
            passed = violations == 0
            print("PASS" if passed else f"FAIL ({violations}/{len(recs)} violations)")
            results.append(FilterTestResult(
                name=test["name"],
                business_id=business_id,
                passed=passed,
                violations=violations,
                total=len(recs),
            ))
        except Exception as exc:
            print(f"ERROR — {exc}")
            results.append(FilterTestResult(
                name=test["name"],
                business_id=business_id,
                passed=False,
                violations=-1,
                total=0,
            ))

    cursor.close()
    return results


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _percentile(values, p):
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(int(len(sv) * p / 100), len(sv) - 1)
    return sv[idx]


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(query_results, filter_results):
    W = 60
    print("\n" + "=" * W)
    print("EVALUATION SUMMARY")
    print("=" * W)

    if not query_results:
        print("No query results available.")
        return

    print(f"\n── Heuristic Scores  (avg, n={len(query_results)}) ──")
    heuristic_fields = [
        ("industry_match",       "Industry match rate"),
        ("trade_region_overlap", "Trade region overlap"),
        ("trade_type_compat",    "Trade type compatibility"),
        ("role_compat",          "Role compatibility"),
        ("category_match",       "Category match rate"),
    ]
    for attr, label in heuristic_fields:
        vals = [getattr(r.heuristic, attr) for r in query_results]
        score = _avg(vals)
        cfg = METRIC_THRESHOLDS[attr]
        if cfg["direction"] == "high":
            verdict = "PASS" if score >= cfg["threshold"] else "FAIL"
            print(f"  {label:<28} {score:.3f}  [{verdict} >= {cfg['threshold']}]")
        elif cfg["direction"] == "low":
            verdict = "PASS" if score <= cfg["threshold"] else "FAIL"
            print(f"  {label:<28} {score:.3f}  [{verdict} <= {cfg['threshold']}]")
        else:
            print(f"  {label:<28} {score:.3f}")

    gt = [r for r in query_results if r.precision_at_k is not None]
    if gt:
        print(f"\n── Ground Truth Metrics  (n={len(gt)}) ──")
        print(f"  precision@{TOP_K}                    {_avg([r.precision_at_k for r in gt]):.3f}")
        print(f"  recall@{TOP_K}                       {_avg([r.recall_at_k for r in gt]):.3f}")
    else:
        print(f"\n── Ground Truth Metrics ──")
        print("  (no queries with expected_results — skipped)")

    latencies = [r.latency_s for r in query_results]
    print(f"\n── Latency  (n={len(latencies)}) ──")
    print(f"  min    {min(latencies):.3f}s")
    print(f"  avg    {_avg(latencies):.3f}s")
    print(f"  p50    {_percentile(latencies, 50):.3f}s")
    print(f"  p95    {_percentile(latencies, 95):.3f}s")
    print(f"  max    {max(latencies):.3f}s")

    if filter_results:
        n_pass = sum(1 for r in filter_results if r.passed)
        print(f"\n── Filter Correctness  ({n_pass}/{len(filter_results)} passed) ──")
        for r in filter_results:
            if r.passed:
                status = "PASS"
            elif r.violations == -1:
                status = "ERROR"
            else:
                status = f"FAIL  ({r.violations}/{r.total} violations)"
            print(f"  {r.name:<30} {status}")

    print("=" * W)


# ---------------------------------------------------------------------------
# JSON dump
# ---------------------------------------------------------------------------

def dump_results(query_results, filter_results):
    def serialise_query(r):
        return {
            "business_id": r.business_id,
            "business_name": r.business_name,
            "industry": r.industry,
            "category": r.category,
            "trade_type": r.trade_type,
            "latency_s": r.latency_s,
            "heuristic": {
                "industry_match": r.heuristic.industry_match,
                "trade_region_overlap": r.heuristic.trade_region_overlap,
                "trade_type_compat": r.heuristic.trade_type_compat,
                "role_compat": r.heuristic.role_compat,
                "category_match": r.heuristic.category_match,
            },
            "precision_at_k": r.precision_at_k,
            "recall_at_k": r.recall_at_k,
            "top_k_ids": r.top_k_ids,
            "top_k_reasoning": r.top_k_reasoning,
        }

    def serialise_filter(r):
        return {
            "name": r.name,
            "business_id": r.business_id,
            "passed": r.passed,
            "violations": r.violations,
            "total": r.total,
        }

    out = {
        "config": {"top_k": TOP_K, "n_queries": len(query_results), "thresholds": METRIC_THRESHOLDS},
        "query_results": [serialise_query(r) for r in query_results],
        "filter_results": [serialise_filter(r) for r in filter_results],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nFull results → {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Connecting to database...")
    conn = connect_db()

    print("Building eval inputs...")
    eval_inputs = build_eval_inputs(conn)
    print(f"  {len(eval_inputs)} eval queries prepared.")

    print(f"\nRunning recommendation eval (top_k={TOP_K}, explain=False)...")
    query_results = run_eval(eval_inputs)

    print("\nRunning filter correctness tests...")
    filter_results = run_filter_tests(conn)

    conn.close()

    print_summary(query_results, filter_results)
    dump_results(query_results, filter_results)
