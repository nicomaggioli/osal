#!/usr/bin/env python3
"""Efficient, frozen-neighborhood search for continuous spot-basis history."""
from __future__ import annotations

import json

import continuous_spot_basis_dev as c


def compact(r):
    return {k: v for k, v in r.items() if k != "records"}


def main():
    spot = c.load_spot(["2026-07-26", "2026-07-27", "2026-07-28"])
    meta_by = c.load_metadata()
    rows_by = c.load_dense(meta_by)
    metas = sorted((m for m in meta_by.values() if m.ticker in rows_by), key=lambda m: m.open_s)
    if len(metas) != 192:
        raise RuntimeError(f"expected 192 scheduled markets, found {len(metas)}")

    model_cache = {}
    results = []
    for window, mult in [(60, 0.75), (120, 0.60), (120, 0.75), (120, 0.90), (240, 0.75)]:
        opps = {m.ticker: c.build_opportunities(m, rows_by[m.ticker], spot, window, mult) for m in metas}
        model_cache[(window, mult)] = opps
        for latency in [2, 3, 5]:
            for participation in [0.05, 0.10, 0.25]:
                for min_t in [45, 90]:
                    for max_t in [450, 600]:
                        for pmin in [0.75, 0.80, 0.85, 0.90]:
                            for edge_min in [0.01, 0.02, 0.03, 0.05]:
                                for max_ask in [0.80, 0.90, 0.95]:
                                    params = (latency, participation, min_t, max_t, pmin, edge_min, max_ask)
                                    r = c.evaluate(params, metas, rows_by, opps)
                                    r["model"] = {"sigma_window": window, "sigma_multiplier": mult}
                                    results.append(r)

    ranked = sorted(results, key=lambda r: (r["fixed_3h_lcb95"], r["after_top10_mean"], r["total_pnl"]), reverse=True)
    robust = [r for r in ranked if r["trades"] >= 30
              and min(r["chronological_thirds"] or [-1]) > 0
              and min(r["day_means"].values() or [-1]) > 0
              and r["after_top10_mean"] > 0
              and r["after_strongest_3h_block_mean"] > 0
              and r["top1_positive_share"] <= 0.15
              and r["top5_positive_share"] <= 0.50]
    chosen = robust[0] if robust else ranked[0]
    model = chosen["model"]
    p = chosen["params"]
    params = (p["latency_s"], p["participation"], p["min_seconds_to_close"], p["max_seconds_to_close"], p["p_min"], p["edge_min"], p["max_ask"])
    best = c.evaluate(params, metas, rows_by, model_cache[(model["sigma_window"], model["sigma_multiplier"])])
    best["model"] = model
    report = {
        "round": "continuous basis-anchored settlement-average variance development fast",
        "opened_days": sorted(c.TARGET_DAYS),
        "unopened_continuous_day": "2026-07-29",
        "market_count": len(metas), "grid_count": len(results), "robust_count": len(robust),
        "best": best,
        "top100_robust": [compact(r) for r in robust[:100]],
        "top100_all": [compact(r) for r in ranked[:100]],
    }
    (c.OUT / "continuous_spot_basis_dev.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(compact(best), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
