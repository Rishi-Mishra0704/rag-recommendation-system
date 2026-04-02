# eval.py - Evaluation metrics and pipeline for recommendation quality

import contextlib
import io
import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from recommendation.common import connect_db
from recommendation.logic.recommendations import recommend
from recommendation.evaluation.eval_scoring import (
    compute_heuristics,
    precision_at_k,
    recall_at_k,
)
from recommendation.evaluation.eval_report import (
    TOP_K,
    print_summary,
    dump_results,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_EVAL_COUNT = 20
EVAL_DIR = Path(__file__).resolve().parent
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"

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
                 top_k_ids, top_k_reasoning, top_k=None):
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
        self.top_k = top_k or []


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
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(eval_inputs):
    results = []

    for entry in eval_inputs:
        bid = entry["business_id"]
        expected = entry.get("expected_results") or []

        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = recommend(bid, top_k=TOP_K, explain=True)
        except Exception:
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
            top_k=matches[:TOP_K],
        ))

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
            results.append(FilterTestResult(
                name=test["name"],
                business_id=business_id,
                passed=passed,
                violations=violations,
                total=len(recs),
            ))
        except Exception:
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
