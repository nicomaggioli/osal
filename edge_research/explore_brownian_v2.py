#!/usr/bin/env python3
"""EDGE round 3b: efficient development stress for averaging-variance edge.

The search uses all eleven development captures, while preserving every later
capture untouched. Candidate books and executions are prepared once per model;
parameter evaluation never rebuilds them. The selected rule must remain
profitable in the early and late chronological development blocks separately.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

import explore_brownian_average as b

edge = b.edge


def prepare(opps):
    by = defaultdict(list)
    for o in opps:
        by[(o.capture, o.ticker)].append(o)
    for values in by.values():
        values.sort(key=lambda x: x.signal_ns)
    return by


def eval_prepared(params, prepared, markets, keep_records=False):
    latency, min_t, max_t, pmin, edge_min, max_ask = params
    records = []
    for m in markets:
        chosen = None
        for o in prepared.get((m.capture, m.ticker), []):
            # signal_ns is ascending, therefore seconds_to_close is descending.
            if o.seconds_to_close > max_t:
                continue
            if o.seconds_to_close < min_t:
                break
            if o.p_side < pmin or o.ask_avg > max_ask:
                continue
            fee_pc = edge.FEE_RATE * o.ask_avg * (1.0 - o.ask_avg)
            if o.p_side - o.ask_avg - fee_pc < edge_min:
                continue
            chosen = o
            break
        if chosen is None:
            rec = {"capture": m.capture, "ticker": m.ticker, "open_ms": m.open_ms,
                   "pnl": 0.0, "qty": 0.0, "win": None}
        else:
            value, qty, fees, win = b.pnl(chosen, latency)
            rec = {"capture": m.capture, "ticker": m.ticker, "open_ms": m.open_ms,
                   "pnl": value, "qty": qty, "fees": fees, "win": win,
                   "side": chosen.side, "p_side": chosen.p_side,
                   "ask": chosen.ask_avg, "seconds_to_close": chosen.seconds_to_close}
        records.append(rec)
    records.sort(key=lambda r: (r["open_ms"], r["ticker"]))
    pnls = np.array([r["pnl"] for r in records], dtype=float)
    n = len(records)
    mean_market = float(pnls.mean()) if n else -1e9
    cap = defaultdict(list)
    for r in records:
        cap[r["capture"]].append(float(r["pnl"]))
    cap_means = np.array([np.mean(v) for _, v in sorted(cap.items())], dtype=float)
    if len(cap_means) >= 2:
        se = cap_means.std(ddof=1) / math.sqrt(len(cap_means))
        annual_lcb = (cap_means.mean() - student_t.ppf(0.95, df=len(cap_means) - 1) * se) * 96.0 * 365.0
    else:
        annual_lcb = -1e9
    thirds = [float(x.mean()) for x in np.array_split(pnls, 3) if len(x)]
    ordered = sorted((float(x) for x in pnls), reverse=True)
    cut = max(1, math.ceil(0.10 * n)) if n else 0
    after_top = sum(ordered[cut:]) / n if n else -1e9
    cap_totals = {k: sum(v) for k, v in cap.items()}
    strongest = max(cap_totals, key=cap_totals.get) if cap_totals else None
    denom = n - len(cap[strongest]) if strongest else 0
    after_capture = sum(v for k, v in cap_totals.items() if k != strongest) / max(1, denom) if strongest else -1e9
    out = {
        "params": {"latency_ms": latency, "min_seconds_to_close": min_t,
                   "max_seconds_to_close": max_t, "p_min": pmin,
                   "edge_min": edge_min, "max_ask": max_ask},
        "markets": n, "trades": sum(r["qty"] > 0 for r in records),
        "wins": sum(r.get("win") is True for r in records),
        "losses": sum(r.get("win") is False for r in records),
        "partial_or_failed": sum(0 < r["qty"] < b.MAX_QTY for r in records),
        "failed_after_signal": sum(r["qty"] == 0 and "side" in r for r in records),
        "total_pnl": float(pnls.sum()), "mean_per_market": mean_market,
        "annualized": mean_market * 96.0 * 365.0,
        "annual_lcb95": float(annual_lcb), "capture_means": cap_means.tolist(),
        "third_means": thirds, "after_top10_mean": float(after_top),
        "after_strongest_capture_mean": float(after_capture),
    }
    if keep_records:
        out["records"] = records
    return out


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

    model_cache = {}
    stage1 = []
    latencies = [1000, 1500, 2000, 3000]
    for window in [30, 60, 120]:
        for mult in [0.60, 0.75, 0.90, 1.10]:
            prepared = prepare(b.build_opps(all_dev, window, mult, latencies))
            model_cache[(window, mult)] = prepared
            for latency in latencies:
                for min_t in [45, 60, 90]:
                    for max_t in [300, 450, 600]:
                        for pmin in [0.75, 0.80, 0.85, 0.90]:
                            for edge_min in [0.01, 0.02, 0.03, 0.05]:
                                for max_ask in [0.85, 0.90, 0.95]:
                                    params = (latency, min_t, max_t, pmin, edge_min, max_ask)
                                    r = eval_prepared(params, prepared, all_dev)
                                    r["model"] = {"sigma_window": window, "sigma_multiplier": mult}
                                    stage1.append(r)

    ranked = sorted(stage1, key=lambda r: (r["annual_lcb95"], r["after_top10_mean"], r["total_pnl"]), reverse=True)
    finalists = []
    for r in ranked[:1000]:
        model = r["model"]
        prepared = model_cache[(model["sigma_window"], model["sigma_multiplier"])]
        p = r["params"]
        params = (p["latency_ms"], p["min_seconds_to_close"], p["max_seconds_to_close"],
                  p["p_min"], p["edge_min"], p["max_ask"])
        item = dict(r)
        item["early_block"] = eval_prepared(params, prepared, early)
        item["late_block"] = eval_prepared(params, prepared, late)
        item["minimum_block_lcb95"] = min(item["annual_lcb95"], item["early_block"]["annual_lcb95"], item["late_block"]["annual_lcb95"])
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
    robust.sort(key=lambda r: (r["minimum_block_lcb95"], r["annual_lcb95"], r["total_pnl"]), reverse=True)
    best_summary = robust[0] if robust else finalists[0]
    model = best_summary["model"]
    prepared = model_cache[(model["sigma_window"], model["sigma_multiplier"])]
    p = best_summary["params"]
    params = (p["latency_ms"], p["min_seconds_to_close"], p["max_seconds_to_close"], p["p_min"], p["edge_min"], p["max_ask"])
    best = eval_prepared(params, prepared, all_dev, keep_records=True)
    best["model"] = model
    best["early_block"] = best_summary["early_block"]
    best["late_block"] = best_summary["late_block"]
    best["minimum_block_lcb95"] = best_summary["minimum_block_lcb95"]

    report = {
        "round": "settlement-average variance arbitrage v2 optimized",
        "opened_captures": [k.name for k, _ in dev_pairs],
        "untouched_after_development": [k.name for k, _ in pairs[edge.DEV_CAPTURE_COUNT:]],
        "early_markets": len(early), "late_markets": len(late), "all_dev_markets": len(all_dev),
        "stage1_count": len(stage1), "finalist_count": len(finalists), "robust_count": len(robust),
        "best": best,
        "top100_robust": [compact(r) for r in robust[:100]],
        "top100_all_dev": [compact(r) for r in ranked[:100]],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "brownian_average_v2.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(compact(best), indent=2))


if __name__ == "__main__":
    main()
