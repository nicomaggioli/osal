#!/usr/bin/env python3
"""EDGE round 2: temporal complement acquisition.

Mechanism: acquire one binary leg when it becomes cheap, then acquire the
opposite leg later only when the two observed all-in costs lock a positive
payout spread. Any unmatched first-leg inventory is held to settlement and is
fully included in P&L. The strategy never submits more than 100 contracts of
gross notional in one market: at most 50 first-leg contracts and at most the
same number of complementary contracts.

Only the first eleven chronological captures are opened. Later captures remain
untouched for a separately frozen holdout.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

import explore_fast as fast

edge = fast.edge

Q_FIRST = 50.0
LATENCIES = [0, 500, 1000, 2000]


@dataclass(frozen=True)
class Fill:
    qty: float
    cost: float
    fees: float
    index: int
    signal_ns: int
    execution_ns: int


def dedupe(groups):
    best = {}
    owner = {}
    for gi, group in enumerate(groups):
        for m in group:
            prior = best.get(m.ticker)
            score = (len(m.snapshots), -m.open_ms)
            if prior is None or score > (len(prior.snapshots), -prior.open_ms):
                best[m.ticker] = m
                owner[m.ticker] = gi
    out = [[] for _ in groups]
    for ticker, m in best.items():
        out[owner[ticker]].append(m)
    for g in out:
        g.sort(key=lambda m: (m.open_ms, m.ticker))
    return out


def avg_for_qty(levels, qty_target):
    qty = 0.0
    cost = 0.0
    worst = None
    for price, displayed in levels:
        take = min(qty_target - qty, displayed * edge.DEPTH_HAIRCUT)
        if take <= 0:
            continue
        qty += take
        cost += take * price
        worst = price
        if qty >= qty_target - 1e-9:
            return cost / qty, worst, qty
    if qty >= min(5.0, qty_target):
        return cost / qty, worst, qty
    return None


def fee_estimate(qty, price):
    return edge.FEE_RATE * qty * price * (1.0 - price)


def execute(m, snap_ns, signal_index, side, qty, latency_ms, max_price):
    target = snap_ns[signal_index] + latency_ms * 1_000_000
    j = bisect.bisect_left(snap_ns, target)
    if j >= len(m.snapshots):
        return Fill(0.0, 0.0, 0.0, j, snap_ns[signal_index], target)
    # Reject stale polling gaps larger than 1.75 seconds beyond requested latency.
    if snap_ns[j] - target > 1_750_000_000:
        return Fill(0.0, 0.0, 0.0, j, snap_ns[signal_index], snap_ns[j])
    levels = m.snapshots[j].yes_asks if side == "yes" else m.snapshots[j].no_asks
    q, cost, fees = edge.walk(levels, max_price, max_qty=qty)
    return Fill(q, cost, fees, j, snap_ns[signal_index], snap_ns[j])


def run_market(m, params):
    latency, entry_cap, earliest_s, latest_before_close_s, lock_margin, slip = params
    snaps = m.snapshots
    snap_ns = [s.recv_ns for s in snaps]
    first = None
    first_side = None
    first_avg = None
    first_signal = None

    for i, s in enumerate(snaps):
        since_open = (s.recv_ns - m.open_ms * 1_000_000) / 1e9
        to_close = (m.close_ms * 1_000_000 - s.recv_ns) / 1e9
        if since_open < earliest_s or to_close < latest_before_close_s:
            continue
        ya = avg_for_qty(s.yes_asks, Q_FIRST)
        na = avg_for_qty(s.no_asks, Q_FIRST)
        choices = []
        if ya is not None and ya[0] <= entry_cap:
            choices.append((ya[0], "yes", ya))
        if na is not None and na[0] <= entry_cap:
            choices.append((na[0], "no", na))
        if not choices:
            continue
        _, side, info = min(choices, key=lambda x: (x[0], x[1]))
        current_avg, current_worst, current_qty = info
        limit = min(0.999, current_worst + slip, entry_cap + slip)
        fill = execute(m, snap_ns, i, side, Q_FIRST, latency, limit)
        if fill.qty <= 0:
            continue
        first = fill
        first_side = side
        first_avg = fill.cost / fill.qty
        first_signal = i
        break

    if first is None:
        return {
            "pnl": 0.0, "gross": 0.0, "first_qty": 0.0, "second_qty": 0.0,
            "locked": False, "first_side": None, "win": None,
        }

    second_side = "no" if first_side == "yes" else "yes"
    second = Fill(0.0, 0.0, 0.0, len(snaps), 0, 0)
    max_second = min(first.qty, 100.0 - first.qty)
    # Begin monitoring only after the first execution snapshot, so the second
    # signal cannot use a book observed before the first fill.
    start_i = min(len(snaps), first.index + 1)
    for i in range(start_i, len(snaps)):
        s = snaps[i]
        levels = s.yes_asks if second_side == "yes" else s.no_asks
        info = avg_for_qty(levels, max_second)
        if info is None:
            continue
        current_avg, current_worst, current_qty = info
        # Conservative signal gate includes estimated fees on both legs and
        # requires the requested locked margin before an order is sent.
        first_fee_pc = first.fees / first.qty
        second_fee_pc = fee_estimate(1.0, current_avg)
        if first_avg + first_fee_pc + current_avg + second_fee_pc > 1.0 - lock_margin + 1e-12:
            continue
        max_price = 1.0 - lock_margin - first_avg - first_fee_pc - second_fee_pc
        limit = min(0.999, current_worst + slip, max_price)
        if limit <= 0:
            continue
        candidate = execute(m, snap_ns, i, second_side, max_second, latency, limit)
        if candidate.qty > 0:
            second = candidate
            break

    first_wins = (first_side == "yes" and m.outcome_yes) or (first_side == "no" and not m.outcome_yes)
    second_wins = (second_side == "yes" and m.outcome_yes) or (second_side == "no" and not m.outcome_yes)
    payoff = (first.qty if first_wins else 0.0) + (second.qty if second_wins else 0.0)
    pnl = payoff - first.cost - first.fees - second.cost - second.fees
    return {
        "pnl": pnl,
        "gross": first.qty + second.qty,
        "first_qty": first.qty,
        "second_qty": second.qty,
        "locked": second.qty >= first.qty - 1e-9,
        "first_side": first_side,
        "win": first_wins,
        "first_avg": first_avg,
        "first_fee": first.fees,
        "second_avg": (second.cost / second.qty if second.qty else None),
        "second_fee": second.fees,
        "first_signal_ns": first.signal_ns,
        "first_execution_ns": first.execution_ns,
        "second_signal_ns": second.signal_ns if second.qty else None,
        "second_execution_ns": second.execution_ns if second.qty else None,
    }


def metrics(params, markets):
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
        crit = student_t.ppf(0.95, df=len(cap_means) - 1)
        lcb = (cap_means.mean() - crit * se) * 96.0 * 365.0
    else:
        lcb = -1e9

    thirds = [float(x.mean()) for x in np.array_split(pnls, 3) if len(x)]
    ordered = sorted((float(x) for x in pnls), reverse=True)
    cut = max(1, math.ceil(0.10 * n)) if n else 0
    after_top = sum(ordered[cut:]) / n if n else -1e9
    cap_totals = {k: sum(v) for k, v in cap.items()}
    if cap_totals:
        strongest = max(cap_totals, key=cap_totals.get)
        denom = n - len(cap[strongest])
        after_capture = sum(v for k, v in cap_totals.items() if k != strongest) / max(1, denom)
    else:
        after_capture = -1e9

    gross_max = max((r["gross"] for r in records), default=0.0)
    return {
        "params": {
            "latency_ms": params[0], "entry_cap": params[1],
            "earliest_s": params[2], "latest_before_close_s": params[3],
            "lock_margin": params[4], "slip": params[5],
        },
        "markets": n,
        "entries": sum(r["first_qty"] > 0 for r in records),
        "fully_locked": sum(r["locked"] for r in records),
        "orphaned": sum(r["first_qty"] > r["second_qty"] + 1e-9 for r in records),
        "gross_max": gross_max,
        "total_pnl": float(pnls.sum()),
        "mean_per_market": mean_market,
        "annualized": annual,
        "annual_lcb95": float(lcb),
        "capture_means": cap_means.tolist(),
        "third_means": thirds,
        "after_top10_mean": float(after_top),
        "after_strongest_capture_mean": float(after_capture),
        "records": records,
    }


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = []
    for k, p in dev_pairs:
        brti = edge.read_perp(p)
        groups.append(edge.read_kalshi(k, brti))
    groups = dedupe(groups)
    train = [m for g in groups[: edge.TRAIN_CAPTURE_COUNT] for m in g]
    val = [m for g in groups[edge.TRAIN_CAPTURE_COUNT :] for m in g]

    grid = []
    for latency in LATENCIES:
        for cap in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            for earliest in [0, 60, 180, 300]:
                for latest in [300, 180, 120, 60]:
                    for margin in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
                        for slip in [0.0, 0.01, 0.02]:
                            grid.append((latency, cap, earliest, latest, margin, slip))

    results = []
    for p in grid:
        r = metrics(p, val)
        compact = {k: v for k, v in r.items() if k != "records"}
        results.append((r, compact))
    ranked = sorted(results, key=lambda x: (x[0]["annual_lcb95"], x[0]["after_top10_mean"], x[0]["total_pnl"]), reverse=True)
    robust = [x for x in ranked if x[0]["entries"] >= 8 and min(x[0]["third_means"] or [-1]) > 0 and x[0]["after_top10_mean"] > 0 and x[0]["after_strongest_capture_mean"] > 0 and x[0]["gross_max"] <= 100.0 + 1e-9]
    by_total = sorted(results, key=lambda x: (x[0]["total_pnl"], x[0]["entries"]), reverse=True)
    by_entries = sorted(results, key=lambda x: (x[0]["entries"], x[0]["total_pnl"]), reverse=True)
    best = (robust[0] if robust else ranked[0])[0]

    report = {
        "round": "temporal complement acquisition v1",
        "opened_captures": [k.name for k, _ in dev_pairs],
        "untouched_after_development": [k.name for k, _ in pairs[edge.DEV_CAPTURE_COUNT :]],
        "train_markets": len(train), "validation_markets": len(val),
        "grid_size": len(grid),
        "best": best,
        "top50": [x[1] for x in ranked[:50]],
        "top50_robust": [x[1] for x in robust[:50]],
        "top20_by_total_pnl": [x[1] for x in by_total[:20]],
        "top20_by_entries": [x[1] for x in by_entries[:20]],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "temporal_box_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in best.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
