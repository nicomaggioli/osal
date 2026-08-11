#!/usr/bin/env python3
"""EDGE round 3: settlement-average variance arbitrage.

Kalshi KXBTC15M settles on the arithmetic mean of sixty BRTI observations,
not on terminal BTC. Under a causal random-walk approximation, the conditional
variance of that future average is analytically smaller than terminal-price
variance. This script values the binary using that exact averaging geometry,
compares the value with observed executable asks, and validates one delayed IOC
entry followed by settlement.

Only the first eleven chronological captures are opened.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t

import explore_fast as fast

edge = fast.edge

WINDOWS = [30, 60, 120]
MULTIPLIERS = [0.75, 1.0, 1.25, 1.5]
LATENCIES = [500, 1000, 2000]
MAX_QTY = 100.0
SLIP_ALLOWANCE = 0.02


@dataclass(frozen=True)
class State:
    snap_index: int
    recv_ns: int
    seconds_to_close: float
    model_horizon: float
    mean_final: float
    sigma_1s: float
    variance_factor: float


@dataclass(frozen=True)
class Opp:
    capture: str
    ticker: str
    signal_ns: int
    seconds_to_close: float
    side: str
    p_side: float
    ask_avg: float
    ask_worst: float
    outcome_yes: bool
    executions: dict[int, tuple[float, float, float]]


def dedupe(groups):
    best = {}
    owner = {}
    for gi, group in enumerate(groups):
        for m in group:
            score = (len(m.snapshots), -m.open_ms)
            old = best.get(m.ticker)
            if old is None or score > (len(old.snapshots), -old.open_ms):
                best[m.ticker] = m
                owner[m.ticker] = gi
    out = [[] for _ in groups]
    for ticker, m in best.items():
        out[owner[ticker]].append(m)
    for g in out:
        g.sort(key=lambda m: (m.open_ms, m.ticker))
    return out


def avg_for_qty(levels, target=MAX_QTY):
    qty = 0.0
    cost = 0.0
    worst = None
    for price, displayed in levels:
        take = min(target - qty, displayed * edge.DEPTH_HAIRCUT)
        if take <= 0:
            continue
        qty += take
        cost += take * price
        worst = price
        if qty >= target - 1e-9:
            return cost / qty, worst
    if qty >= 10.0:
        return cost / qty, worst
    return None


def unique_received_rows(rows, recv_ns):
    latest = {}
    for r in rows:
        if r.recv_ns > recv_ns:
            break
        latest[r.ref_sec] = r.brti
    return latest


def build_states(m, window):
    rows = sorted(m.brti, key=lambda x: x.recv_ns)
    snaps = sorted(m.snapshots, key=lambda x: x.recv_ns)
    pointer = 0
    known = {}
    unique_secs = []
    unique_vals = []
    out = []
    end = m.close_ms // 1000
    start = end - 60

    for si, snap in enumerate(snaps):
        while pointer < len(rows) and rows[pointer].recv_ns <= snap.recv_ns:
            r = rows[pointer]
            if r.ref_sec in known:
                known[r.ref_sec] = r.brti
                if unique_secs and unique_secs[-1] == r.ref_sec:
                    unique_vals[-1] = r.brti
            else:
                known[r.ref_sec] = r.brti
                unique_secs.append(r.ref_sec)
                unique_vals.append(r.brti)
            pointer += 1
        if len(unique_vals) < max(20, window // 2):
            continue
        latest_sec = unique_secs[-1]
        current = unique_vals[-1]
        recent = unique_vals[-(window + 1):]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        if len(diffs) < 10:
            continue
        sigma = float(np.std(diffs, ddof=1))
        if not math.isfinite(sigma) or sigma < 0.05:
            sigma = 0.05

        observed = []
        sec = start
        while sec < end and sec in known:
            observed.append(known[sec])
            sec += 1
        n = len(observed)
        if n > 0:
            remaining = 60 - n
            mean_final = (sum(observed) + remaining * observed[-1]) / 60.0
            # Exact discrete random-walk variance of the unknown future samples
            # after scaling the full sixty-sample arithmetic mean.
            variance_factor = remaining * (remaining + 1) * (2 * remaining + 1) / (6.0 * 60.0 * 60.0)
            model_horizon = float(remaining)
        else:
            # h is the number of random-walk increments before the first of the
            # sixty future settlement observations. Each such increment enters
            # all sixty samples; within-window increments receive weights 60..1.
            h = max(0, start - latest_sec - 1)
            sumsq = 60 * 61 * 121 / 6.0
            variance_factor = h + sumsq / (60.0 * 60.0)
            mean_final = current
            model_horizon = float(h + 60)
        seconds_to_close = (m.close_ms * 1_000_000 - snap.recv_ns) / 1e9
        if seconds_to_close <= 0:
            continue
        out.append(State(si, snap.recv_ns, seconds_to_close, model_horizon, mean_final, sigma, variance_factor))
    return out


def build_opps(markets, window, mult, latencies):
    out = []
    for m in markets:
        snap_ns = [s.recv_ns for s in m.snapshots]
        seen_second = set()
        for st in build_states(m, window):
            second_bucket = int(math.floor(st.seconds_to_close))
            if second_bucket in seen_second:
                continue
            seen_second.add(second_bucket)
            sd = max(0.01, st.sigma_1s * mult * math.sqrt(max(st.variance_factor, 1e-12)))
            p_yes = float(norm.cdf((st.mean_final - m.strike) / sd))
            if p_yes >= 0.5:
                side = "yes"; p_side = p_yes; levels = m.snapshots[st.snap_index].yes_asks
            else:
                side = "no"; p_side = 1.0 - p_yes; levels = m.snapshots[st.snap_index].no_asks
            aa = avg_for_qty(levels)
            if aa is None:
                continue
            ask_avg, ask_worst = aa
            executions = {}
            for latency in latencies:
                target = st.recv_ns + latency * 1_000_000
                j = bisect.bisect_left(snap_ns, target)
                if j >= len(m.snapshots) or snap_ns[j] - target > 1_750_000_000:
                    executions[latency] = (0.0, 0.0, 0.0)
                    continue
                exlevels = m.snapshots[j].yes_asks if side == "yes" else m.snapshots[j].no_asks
                limit = min(0.999, ask_worst + SLIP_ALLOWANCE, max(0.0, p_side - 0.005))
                executions[latency] = edge.walk(exlevels, limit, max_qty=MAX_QTY)
            out.append(Opp(m.capture, m.ticker, st.recv_ns, st.seconds_to_close, side, p_side,
                           ask_avg, ask_worst, bool(m.outcome_yes), executions))
    return out


def pnl(o, latency):
    qty, cost, fees = o.executions[latency]
    if qty <= 0:
        return 0.0, qty, fees, None
    win = (o.side == "yes" and o.outcome_yes) or (o.side == "no" and not o.outcome_yes)
    return (qty if win else 0.0) - cost - fees, qty, fees, win


def evaluate(params, opps, markets):
    latency, min_t, max_t, pmin, edge_min, max_ask = params
    by_market = defaultdict(list)
    for o in opps:
        by_market[(o.capture, o.ticker)].append(o)
    for values in by_market.values():
        values.sort(key=lambda x: x.signal_ns)
    records = []
    for m in markets:
        chosen = None
        for o in by_market.get((m.capture, m.ticker), []):
            if not (min_t <= o.seconds_to_close <= max_t):
                continue
            if o.p_side < pmin or o.ask_avg > max_ask:
                continue
            fee_pc = edge.FEE_RATE * o.ask_avg * (1.0 - o.ask_avg)
            if o.p_side - o.ask_avg - fee_pc < edge_min:
                continue
            chosen = o
            break
        if chosen is None:
            records.append({"capture": m.capture, "ticker": m.ticker, "open_ms": m.open_ms,
                            "pnl": 0.0, "qty": 0.0, "win": None})
        else:
            value, qty, fees, win = pnl(chosen, latency)
            records.append({"capture": m.capture, "ticker": m.ticker, "open_ms": m.open_ms,
                            "pnl": value, "qty": qty, "fees": fees, "win": win,
                            "side": chosen.side, "p_side": chosen.p_side,
                            "ask": chosen.ask_avg, "seconds_to_close": chosen.seconds_to_close})
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
        annual_lcb = (cap_means.mean() - crit * se) * 96.0 * 365.0
    else:
        annual_lcb = -1e9
    thirds = [float(x.mean()) for x in np.array_split(pnls, 3) if len(x)]
    ordered = sorted((float(x) for x in pnls), reverse=True)
    cut = max(1, math.ceil(0.10 * n)) if n else 0
    after_top = sum(ordered[cut:]) / n if n else -1e9
    cap_totals = {k: sum(v) for k, v in cap.items()}
    strongest = max(cap_totals, key=cap_totals.get) if cap_totals else None
    denom = n - len(cap[strongest]) if strongest else 0
    after_capture = (sum(v for k, v in cap_totals.items() if k != strongest) / max(1, denom)) if strongest else -1e9
    return {
        "params": {"latency_ms": latency, "min_seconds_to_close": min_t,
                   "max_seconds_to_close": max_t, "p_min": pmin,
                   "edge_min": edge_min, "max_ask": max_ask},
        "markets": n,
        "trades": sum(r["qty"] > 0 for r in records),
        "wins": sum(r.get("win") is True for r in records),
        "losses": sum(r.get("win") is False for r in records),
        "partial_or_failed": sum(0 < r["qty"] < MAX_QTY for r in records),
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
        brti = edge.read_perp(p)
        groups.append(edge.read_kalshi(k, brti))
    groups = dedupe(groups)
    train = [m for g in groups[: edge.TRAIN_CAPTURE_COUNT] for m in g]
    val = [m for g in groups[edge.TRAIN_CAPTURE_COUNT :] for m in g]

    all_results = []
    diagnostic = []
    for window in WINDOWS:
        for mult in MULTIPLIERS:
            opps = build_opps(val, window, mult, LATENCIES)
            diagnostic.append({"window": window, "multiplier": mult, "opportunities": len(opps)})
            for latency in LATENCIES:
                for min_t in [45, 60, 75, 90]:
                    for max_t in [120, 180, 300]:
                        if min_t > max_t:
                            continue
                        for pmin in [0.60, 0.70, 0.80, 0.90]:
                            for edge_min in [0.02, 0.05, 0.08, 0.10]:
                                for max_ask in [0.50, 0.60, 0.70, 0.80, 0.90]:
                                    r = evaluate((latency, min_t, max_t, pmin, edge_min, max_ask), opps, val)
                                    r["model"] = {"sigma_window": window, "sigma_multiplier": mult}
                                    all_results.append(r)
    ranked = sorted(all_results, key=lambda r: (r["annual_lcb95"], r["after_top10_mean"], r["total_pnl"]), reverse=True)
    robust = [r for r in ranked if r["trades"] >= 8 and min(r["third_means"] or [-1]) > 0 and r["after_top10_mean"] > 0 and r["after_strongest_capture_mean"] > 0]
    by_total = sorted(all_results, key=lambda r: (r["total_pnl"], r["trades"]), reverse=True)
    best = robust[0] if robust else ranked[0]
    report = {
        "round": "settlement-average variance arbitrage v1",
        "opened_captures": [k.name for k, _ in dev_pairs],
        "untouched_after_development": [k.name for k, _ in pairs[edge.DEV_CAPTURE_COUNT:]],
        "train_markets": len(train), "validation_markets": len(val),
        "opportunity_diagnostics": diagnostic,
        "best": best,
        "top50": [{k: v for k, v in r.items() if k != "records"} for r in ranked[:50]],
        "top50_robust": [{k: v for k, v in r.items() if k != "records"} for r in robust[:50]],
        "top20_by_total": [{k: v for k, v in r.items() if k != "records"} for r in by_total[:20]],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "brownian_average_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in best.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
