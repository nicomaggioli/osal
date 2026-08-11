#!/usr/bin/env python3
"""Development-only scheduled-slot optimization for the BRTI mechanism.

All scheduled KXBTC15M closes whose complete signal/latency window is covered by
at least one development capture enter the denominator. Missing markets, no
signals, and failed fills are zero. Ranking uses a one-sided 95% Student-t lower
bound over non-overlapping three-hour blocks, not overlapping capture files.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import explore_brownian_average as b
import explore_brownian_v2 as v2

edge = b.edge
ANNUAL = 96 * 365
DURATION = 10800


def eligible_slots(captures):
    windows = [(int(edge.capture_dt(Path(c)).timestamp()), int(edge.capture_dt(Path(c)).timestamp()) + DURATION) for c in captures]
    first = min(a for a, _ in windows)
    last = max(z for _, z in windows)
    closes = range(((first + 899) // 900) * 900, ((last + 899) // 900) * 900 + 1, 900)
    slots = []
    for close in closes:
        # Full 120-second volatility history at earliest allowed signal c-600,
        # plus two-second execution after the latest allowed signal c-45.
        if any(a <= close - 720 and z >= close - 43 for a, z in windows):
            slots.append(close - 900)
    return sorted(set(slots)), windows


def summarize(records, slots):
    by_open = {int(r["open_ms"]) // 1000: float(r["pnl"]) for r in records}
    values = [by_open.get(s, 0.0) for s in slots]
    blocks = [float(np.mean(values[i:i+12])) for i in range(0, len(values), 12) if len(values[i:i+12]) == 12]
    arr = np.array(blocks, dtype=float)
    if len(arr) >= 2:
        se = arr.std(ddof=1) / math.sqrt(len(arr))
        lcb = (arr.mean() - student_t.ppf(0.95, df=len(arr)-1) * se) * ANNUAL
    else:
        lcb = -1e18
    vals = np.array(values, dtype=float)
    thirds = [float(x.mean()) for x in np.array_split(vals, 3) if len(x)]
    ordered = sorted(values, reverse=True)
    cut = max(1, math.ceil(0.10 * len(values)))
    after_top = sum(ordered[cut:]) / len(values)
    if blocks:
        strongest = max(range(len(blocks)), key=lambda i: blocks[i])
        kept = values[:strongest*12] + values[(strongest+1)*12:]
        after_block = sum(kept) / len(kept) if kept else -1e18
    else:
        after_block = -1e18
    positives = sorted((x for x in values if x > 0), reverse=True)
    psum = sum(positives)
    return {
        "scheduled_slots": len(values), "parsed_records": len(records),
        "trades": sum(r.get("qty", 0) > 0 for r in records),
        "wins": sum(r.get("win") is True for r in records),
        "losses": sum(r.get("win") is False for r in records),
        "total_pnl": sum(values), "mean_per_scheduled_slot": sum(values)/len(values),
        "annualized": sum(values)/len(values)*ANNUAL,
        "fixed_3h_lcb95": float(lcb), "fixed_3h_block_means": blocks,
        "chronological_thirds": thirds, "after_top10_mean": after_top,
        "after_strongest_3h_block_mean": after_block,
        "top1_positive_share": positives[0]/psum if positives and psum else 0.0,
        "top5_positive_share": sum(positives[:5])/psum if positives and psum else 0.0,
    }


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[:edge.DEV_CAPTURE_COUNT]
    groups = []
    for k, p in dev_pairs:
        groups.append(edge.read_kalshi(k, edge.read_perp(p)))
    groups = b.dedupe(groups)
    markets = [m for g in groups for m in g]
    captures = [k.name for k, _ in dev_pairs]
    slots, windows = eligible_slots(captures)

    results = []
    model_cache = {}
    for window, mult in [(60,0.75),(120,0.60),(120,0.75),(120,0.90),(240,0.75)]:
        opps = b.build_opps(markets, window, mult, [1000,1500,2000,3000,5000])
        prepared = v2.prepare(opps)
        model_cache[(window,mult)] = prepared
        for latency in [1000,1500,2000,3000,5000]:
            for min_t in [45,60,90]:
                for max_t in [300,450,600]:
                    for pmin in [0.75,0.80,0.85,0.90,0.95]:
                        for edge_min in [0.01,0.02,0.03,0.05,0.08]:
                            for max_ask in [0.75,0.80,0.85,0.90,0.95]:
                                params=(latency,min_t,max_t,pmin,edge_min,max_ask)
                                r=v2.eval_prepared(params,prepared,markets,keep_records=True)
                                s=summarize(r["records"],slots)
                                s["params"]=r["params"]
                                s["model"]={"sigma_window":window,"sigma_multiplier":mult}
                                results.append(s)
    ranked=sorted(results,key=lambda r:(r["fixed_3h_lcb95"],r["after_top10_mean"],r["total_pnl"]),reverse=True)
    robust=[r for r in ranked if r["trades"]>=20 and min(r["chronological_thirds"] or [-1])>0
            and r["after_top10_mean"]>0 and r["after_strongest_3h_block_mean"]>0
            and r["top1_positive_share"]<=0.20 and r["top5_positive_share"]<=0.60]
    best=robust[0] if robust else ranked[0]
    report={
        "round":"scheduled-slot BRTI averaging-variance optimization",
        "opened_captures":captures,
        "untouched_after_development":[k.name for k,_ in pairs[edge.DEV_CAPTURE_COUNT:]],
        "scheduled_slot_count":len(slots),"scheduled_open_seconds":slots,
        "grid_count":len(results),"robust_count":len(robust),"best":best,
        "top100_robust":robust[:100],"top100_all":ranked[:100],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR/"scheduled_brownian_optimization.json").write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(best,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
