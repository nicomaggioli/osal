#!/usr/bin/env python3
"""EDGE round 2b: temporal complement with a bounded-loss fallback.

The first leg is acquired exactly as in temporal_box_v1. The opposite leg is
then acquired either when a positive spread is locked or, after a frozen wait,
when the all-in pair loss is no worse than a fixed cap. This converts many
otherwise directional orphan losses into observed two-leg bounded outcomes
without exceeding 100 total contracts of gross notional.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

import explore_temporal_box as base

edge = base.edge
Q_FIRST = 50.0


def run_market(m, params):
    latency, entry_cap, latest_before_close_s, lock_margin, max_wait_s, max_pair_loss, slip = params
    snaps = m.snapshots
    snap_ns = [s.recv_ns for s in snaps]
    first = None
    first_side = None
    first_avg = None

    for i, s in enumerate(snaps):
        to_close = (m.close_ms * 1_000_000 - s.recv_ns) / 1e9
        if to_close < latest_before_close_s:
            continue
        choices = []
        ya = base.avg_for_qty(s.yes_asks, Q_FIRST)
        na = base.avg_for_qty(s.no_asks, Q_FIRST)
        if ya is not None and ya[0] <= entry_cap:
            choices.append((ya[0], "yes", ya))
        if na is not None and na[0] <= entry_cap:
            choices.append((na[0], "no", na))
        if not choices:
            continue
        _, side, info = min(choices, key=lambda x: (x[0], x[1]))
        current_avg, current_worst, _ = info
        limit = min(0.999, current_worst + slip, entry_cap + slip)
        fill = base.execute(m, snap_ns, i, side, Q_FIRST, latency, limit)
        if fill.qty <= 0:
            continue
        first = fill
        first_side = side
        first_avg = fill.cost / fill.qty
        break

    if first is None:
        return {"pnl": 0.0, "gross": 0.0, "first_qty": 0.0, "second_qty": 0.0,
                "locked_positive": False, "fallback_locked": False, "orphaned": False}

    second_side = "no" if first_side == "yes" else "yes"
    max_second = min(first.qty, 100.0 - first.qty)
    second = base.Fill(0.0, 0.0, 0.0, len(snaps), 0, 0)
    mode = "orphan"
    first_fee_pc = first.fees / first.qty

    for i in range(min(len(snaps), first.index + 1), len(snaps)):
        s = snaps[i]
        levels = s.yes_asks if second_side == "yes" else s.no_asks
        info = base.avg_for_qty(levels, max_second)
        if info is None:
            continue
        current_avg, current_worst, _ = info
        second_fee_pc = edge.FEE_RATE * current_avg * (1.0 - current_avg)
        pair_cost = first_avg + first_fee_pc + current_avg + second_fee_pc
        elapsed = (s.recv_ns - first.execution_ns) / 1e9
        favorable = pair_cost <= 1.0 - lock_margin + 1e-12
        fallback = elapsed >= max_wait_s and pair_cost <= 1.0 + max_pair_loss + 1e-12
        if not favorable and not fallback:
            continue
        ceiling = (1.0 - lock_margin if favorable else 1.0 + max_pair_loss)
        max_price = ceiling - first_avg - first_fee_pc - second_fee_pc
        limit = min(0.999, current_worst + slip, max_price)
        if limit <= 0:
            continue
        candidate = base.execute(m, snap_ns, i, second_side, max_second, latency, limit)
        if candidate.qty > 0:
            second = candidate
            mode = "positive" if favorable else "fallback"
            break

    first_wins = (first_side == "yes" and m.outcome_yes) or (first_side == "no" and not m.outcome_yes)
    second_wins = (second_side == "yes" and m.outcome_yes) or (second_side == "no" and not m.outcome_yes)
    payoff = (first.qty if first_wins else 0.0) + (second.qty if second_wins else 0.0)
    pnl = payoff - first.cost - first.fees - second.cost - second.fees
    return {
        "pnl": pnl, "gross": first.qty + second.qty,
        "first_qty": first.qty, "second_qty": second.qty,
        "locked_positive": mode == "positive", "fallback_locked": mode == "fallback",
        "orphaned": second.qty < first.qty - 1e-9,
        "first_side": first_side, "first_avg": first_avg,
        "second_avg": second.cost / second.qty if second.qty else None,
        "mode": mode, "first_fee": first.fees, "second_fee": second.fees,
    }


def evaluate(params, markets):
    records = []
    for m in markets:
        r = run_market(m, params)
        r.update({"capture": m.capture, "ticker": m.ticker, "open_ms": m.open_ms})
        records.append(r)
    records.sort(key=lambda r: (r["open_ms"], r["ticker"]))
    pnls = np.array([r["pnl"] for r in records], dtype=float)
    n = len(records)
    mean_market = float(pnls.mean()) if n else -1e9
    annual = mean_market * 96.0 * 365.0
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
    return {
        "params": {"latency_ms": params[0], "entry_cap": params[1],
                   "latest_before_close_s": params[2], "lock_margin": params[3],
                   "max_wait_s": params[4], "max_pair_loss": params[5], "slip": params[6]},
        "markets": n, "entries": sum(r["first_qty"] > 0 for r in records),
        "positive_locks": sum(r["locked_positive"] for r in records),
        "fallback_locks": sum(r["fallback_locked"] for r in records),
        "orphaned": sum(r["orphaned"] for r in records),
        "gross_max": max((r["gross"] for r in records), default=0.0),
        "total_pnl": float(pnls.sum()), "mean_per_market": mean_market,
        "annualized": annual, "annual_lcb95": float(annual_lcb),
        "capture_means": cap_means.tolist(), "third_means": thirds,
        "after_top10_mean": float(after_top),
        "after_strongest_capture_mean": float(after_capture),
        "records": records,
    }


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = []
    for k, p in dev_pairs:
        groups.append(edge.read_kalshi(k, edge.read_perp(p)))
    groups = base.dedupe(groups)
    train = [m for g in groups[: edge.TRAIN_CAPTURE_COUNT] for m in g]
    val = [m for g in groups[edge.TRAIN_CAPTURE_COUNT:] for m in g]

    results = []
    for latency in [500, 1000, 2000]:
        for entry_cap in [0.30, 0.35, 0.40, 0.45]:
            for latest in [300, 180]:
                for margin in [0.03, 0.05, 0.08, 0.10]:
                    for wait in [30, 60, 120, 180, 300]:
                        for max_loss in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
                            results.append(evaluate((latency, entry_cap, latest, margin, wait, max_loss, 0.0), val))
    ranked = sorted(results, key=lambda r: (r["annual_lcb95"], r["after_top10_mean"], r["total_pnl"]), reverse=True)
    robust = [r for r in ranked if r["entries"] >= 8 and min(r["third_means"] or [-1]) > 0 and r["after_top10_mean"] > 0 and r["after_strongest_capture_mean"] > 0 and r["gross_max"] <= 100.0 + 1e-9]
    by_total = sorted(results, key=lambda r: (r["total_pnl"], -r["orphaned"]), reverse=True)
    best = robust[0] if robust else ranked[0]
    report = {
        "round": "bounded-orphan temporal complement v2",
        "opened_captures": [k.name for k, _ in dev_pairs],
        "untouched_after_development": [k.name for k, _ in pairs[edge.DEV_CAPTURE_COUNT:]],
        "train_markets": len(train), "validation_markets": len(val),
        "best": best,
        "top50": [{k: v for k, v in r.items() if k != "records"} for r in ranked[:50]],
        "top50_robust": [{k: v for k, v in r.items() if k != "records"} for r in robust[:50]],
        "top20_by_total": [{k: v for k, v in r.items() if k != "records"} for r in by_total[:20]],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "temporal_box_v2.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in best.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
