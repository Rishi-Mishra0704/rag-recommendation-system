import json
from pathlib import Path

TOP_K = 5
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "eval_results.json"

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
