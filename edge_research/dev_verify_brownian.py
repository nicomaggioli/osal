#!/usr/bin/env python3
"""Development-only adversarial verifier for the frozen-candidate precursor.

No post-development capture is opened. The script attacks latency, depth,
fees, adverse price movement, missing fills, strongest periods, concentration,
and position scaling before any final holdout is exposed.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.stats import t as student_t

import explore_brownian_average as b
import explore_brownian_v2 as v2

edge = b.edge

MODEL_WINDOW = 120
MODEL_MULTIPLIER = 0.75
BASE_PARAMS = {
    "min_seconds_to_close": 45,
    "max_seconds_to_close": 600,
    "p_min": 0.75,
    "edge_min": 0.02,
    "max_ask": 0.90,
}


def summarize(records):
    records = sorted(records, key=lambda r: (r["open_ms"], r["ticker"]))
    pnls = np.array([float(r["pnl"]) for r in records], dtype=float)
    n = len(records)
    mean_market = float(pnls.mean()) if n else -1e9
    cap = defaultdict(list)
    day = defaultdict(list)
    for r in records:
        cap[r["capture"]].append(float(r["pnl"]))
        d = datetime.fromtimestamp(r["open_ms"] / 1000.0, tz=timezone.utc).date().isoformat()
        day[d].append(float(r["pnl"]))
    cap_means = np.array([np.mean(v) for _, v in sorted(cap.items())], dtype=float)
    if len(cap_means) >= 2:
        se = cap_means.std(ddof=1) / math.sqrt(len(cap_means))
        lcb = (cap_means.mean() - student_t.ppf(0.95, df=len(cap_means) - 1) * se) * 96.0 * 365.0
    else:
        lcb = -1e9
    thirds = [float(x.mean()) for x in np.array_split(pnls, 3) if len(x)]
    ordered = sorted((float(x) for x in pnls), reverse=True)
    cut = max(1, math.ceil(0.10 * n)) if n else 0
    after_top = sum(ordered[cut:]) / n if n else -1e9
    cap_totals = {k: sum(v) for k, v in cap.items()}
    day_totals = {k: sum(v) for k, v in day.items()}
    sc = max(cap_totals, key=cap_totals.get) if cap_totals else None
    sd = max(day_totals, key=day_totals.get) if day_totals else None
    after_cap = sum(v for k, v in cap_totals.items() if k != sc) / max(1, n - len(cap[sc])) if sc else -1e9
    after_day = sum(v for k, v in day_totals.items() if k != sd) / max(1, n - len(day[sd])) if sd else -1e9
    positives = sorted((float(r["pnl"]) for r in records if r["pnl"] > 0), reverse=True)
    total_positive = sum(positives)
    top1_share = positives[0] / total_positive if positives and total_positive else 0.0
    top5_share = sum(positives[:5]) / total_positive if positives and total_positive else 0.0
    return {
        "markets": n, "trades": sum(r.get("qty", 0) > 0 for r in records),
        "wins": sum(r.get("win") is True for r in records),
        "losses": sum(r.get("win") is False for r in records),
        "failed_after_signal": sum(r.get("qty", 0) == 0 and "side" in r for r in records),
        "total_pnl": float(pnls.sum()), "mean_per_market": mean_market,
        "annualized": mean_market * 96.0 * 365.0, "annual_lcb95": float(lcb),
        "capture_means": cap_means.tolist(), "third_means": thirds,
        "after_top10_mean": float(after_top),
        "after_strongest_capture_mean": float(after_cap),
        "after_strongest_day_mean": float(after_day),
        "top1_positive_share": top1_share, "top5_positive_share": top5_share,
    }


def run_scenario(markets, *, latency, depth, fee_rate, adverse_per_contract=0.0, quantity_scale=1.0):
    old_depth, old_fee = edge.DEPTH_HAIRCUT, edge.FEE_RATE
    old_max = b.MAX_QTY
    try:
        edge.DEPTH_HAIRCUT = depth
        edge.FEE_RATE = fee_rate
        b.MAX_QTY = 100.0 * quantity_scale
        opps = b.build_opps(markets, MODEL_WINDOW, MODEL_MULTIPLIER, [latency])
        prepared = v2.prepare(opps)
        params = (latency, BASE_PARAMS["min_seconds_to_close"], BASE_PARAMS["max_seconds_to_close"],
                  BASE_PARAMS["p_min"], BASE_PARAMS["edge_min"], BASE_PARAMS["max_ask"])
        result = v2.eval_prepared(params, prepared, markets, keep_records=True)
        records = [dict(r) for r in result["records"]]
        if adverse_per_contract:
            for r in records:
                if r.get("qty", 0) > 0:
                    r["pnl"] -= r["qty"] * adverse_per_contract
        return summarize(records), records
    finally:
        edge.DEPTH_HAIRCUT = old_depth
        edge.FEE_RATE = old_fee
        b.MAX_QTY = old_max


def remove_best_fills(records, fraction):
    out = [dict(r) for r in records]
    candidates = [(float(r["pnl"]), i) for i, r in enumerate(out) if r.get("qty", 0) > 0 and r["pnl"] > 0]
    candidates.sort(reverse=True)
    k = math.ceil(fraction * max(1, sum(r.get("qty", 0) > 0 for r in out)))
    for _, i in candidates[:k]:
        out[i]["pnl"] = 0.0
        out[i]["qty"] = 0.0
        out[i]["win"] = None
        out[i]["removed_as_missing_fill"] = True
    return summarize(out)


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = []
    for k, p in dev_pairs:
        groups.append(edge.read_kalshi(k, edge.read_perp(p)))
    groups = b.dedupe(groups)
    markets = [m for group in groups for m in group]

    scenarios = {}
    baseline, base_records = run_scenario(markets, latency=2000, depth=0.50, fee_rate=0.07)
    scenarios["baseline"] = baseline
    for latency in [1000, 1500, 3000, 5000]:
        scenarios[f"latency_{latency}ms"] = run_scenario(markets, latency=latency, depth=0.50, fee_rate=0.07)[0]
    for depth in [0.25, 0.10]:
        scenarios[f"depth_haircut_{int(depth*100)}pct"] = run_scenario(markets, latency=2000, depth=depth, fee_rate=0.07)[0]
    scenarios["double_fees"] = run_scenario(markets, latency=2000, depth=0.50, fee_rate=0.14)[0]
    scenarios["plus_1cent_adverse"] = run_scenario(markets, latency=2000, depth=0.50, fee_rate=0.07, adverse_per_contract=0.01)[0]
    combined, combined_records = run_scenario(markets, latency=3000, depth=0.25, fee_rate=0.14, adverse_per_contract=0.01)
    scenarios["combined_3s_25pct_doublefee_plus1c"] = combined
    scenarios["quantity_50"] = run_scenario(markets, latency=2000, depth=0.50, fee_rate=0.07, quantity_scale=0.50)[0]
    scenarios["quantity_25"] = run_scenario(markets, latency=2000, depth=0.50, fee_rate=0.07, quantity_scale=0.25)[0]
    scenarios["missing_best_10pct"] = remove_best_fills(base_records, 0.10)
    scenarios["missing_best_20pct"] = remove_best_fills(base_records, 0.20)
    scenarios["combined_missing_best_10pct"] = remove_best_fills(combined_records, 0.10)

    report = {
        "candidate": {"model_window": MODEL_WINDOW, "model_multiplier": MODEL_MULTIPLIER,
                      "latency_ms": 2000, "depth_haircut": 0.50, "fee_rate": 0.07,
                      **BASE_PARAMS},
        "opened_captures": [k.name for k, _ in dev_pairs],
        "untouched_after_development": [k.name for k, _ in pairs[edge.DEV_CAPTURE_COUNT:]],
        "market_count": len(markets), "scenarios": scenarios,
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "brownian_dev_verifier.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(scenarios, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
