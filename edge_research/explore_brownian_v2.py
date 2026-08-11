#!/usr/bin/env python3
"""EDGE round 3b: broaden and stress the settlement-average variance rule.

This development-only search expands time windows, latency, volatility scaling,
and ask thresholds, then requires profitability on the early and late
chronological development blocks separately. All captures after development
remain unopened.
"""
from __future__ import annotations

import json

import explore_brownian_average as b

edge = b.edge


def compact(r):
    return {k: v for k, v in r.items() if k != "records"}


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = []
    for k, p in dev_pairs:
        groups.append(edge.read_kalshi(k, edge.read_perp(p)))
    groups = b.dedupe(groups)
    early = [m for g in groups[: edge.TRAIN_CAPTURE_COUNT] for m in g]
    late = [m for g in groups[edge.TRAIN_CAPTURE_COUNT:] for m in g]
    all_dev = early + late

    models = []
    for window in [30, 60, 120, 240]:
        for mult in [0.60, 0.75, 0.90, 1.00, 1.25]:
            models.append((window, mult, b.build_opps(all_dev, window, mult, [500, 1000, 1500, 2000, 3000])))

    stage1 = []
    for window, mult, opps in models:
        for latency in [500, 1000, 1500, 2000, 3000]:
            for min_t in [45, 60, 90, 120]:
                for max_t in [300, 450, 600, 780]:
                    if min_t > max_t:
                        continue
                    for pmin in [0.70, 0.75, 0.80, 0.85, 0.90]:
                        for edge_min in [0.01, 0.02, 0.03, 0.05, 0.08]:
                            for max_ask in [0.85, 0.90, 0.95, 0.99]:
                                r = b.evaluate((latency, min_t, max_t, pmin, edge_min, max_ask), opps, all_dev)
                                r["model"] = {"sigma_window": window, "sigma_multiplier": mult}
                                stage1.append(r)

    ranked = sorted(stage1, key=lambda r: (r["annual_lcb95"], r["after_top10_mean"], r["total_pnl"]), reverse=True)
    finalists = []
    # Evaluate chronological early/late blocks only for the strongest distinct
    # parameter sets, avoiding a second exhaustive pass.
    for r in ranked[:1000]:
        window = r["model"]["sigma_window"]
        mult = r["model"]["sigma_multiplier"]
        params = r["params"]
        opps_early = b.build_opps(early, window, mult, [params["latency_ms"]])
        opps_late = b.build_opps(late, window, mult, [params["latency_ms"]])
        tup = (params["latency_ms"], params["min_seconds_to_close"], params["max_seconds_to_close"],
               params["p_min"], params["edge_min"], params["max_ask"])
        re = b.evaluate(tup, opps_early, early)
        rl = b.evaluate(tup, opps_late, late)
        item = dict(r)
        item["early_block"] = compact(re)
        item["late_block"] = compact(rl)
        finalists.append(item)

    robust = [r for r in finalists
              if r["trades"] >= 15
              and min(r["third_means"] or [-1]) > 0
              and r["after_top10_mean"] > 0
              and r["after_strongest_capture_mean"] > 0
              and r["early_block"]["mean_per_market"] > 0
              and r["late_block"]["mean_per_market"] > 0
              and min(r["early_block"]["third_means"] or [-1]) >= 0
              and min(r["late_block"]["third_means"] or [-1]) >= 0]
    robust.sort(key=lambda r: (min(r["annual_lcb95"], r["early_block"]["annual_lcb95"], r["late_block"]["annual_lcb95"]),
                               r["annual_lcb95"], r["total_pnl"]), reverse=True)
    best = robust[0] if robust else finalists[0]

    report = {
        "round": "settlement-average variance arbitrage v2",
        "opened_captures": [k.name for k, _ in dev_pairs],
        "untouched_after_development": [k.name for k, _ in pairs[edge.DEV_CAPTURE_COUNT:]],
        "early_markets": len(early), "late_markets": len(late), "all_dev_markets": len(all_dev),
        "stage1_count": len(stage1), "finalist_count": len(finalists), "robust_count": len(robust),
        "best": best,
        "top100_robust": [compact(r) | {"early_block": r["early_block"], "late_block": r["late_block"]} for r in robust[:100]],
        "top100_all_dev": [compact(r) for r in ranked[:100]],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "brownian_average_v2.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(compact(best), indent=2))
    print("early_block", json.dumps(best["early_block"], indent=2))
    print("late_block", json.dumps(best["late_block"], indent=2))


if __name__ == "__main__":
    main()
