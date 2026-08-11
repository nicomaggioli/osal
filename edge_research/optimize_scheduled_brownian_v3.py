#!/usr/bin/env python3
"""Development-only v3 optimizer for the BRTI averaging-variance mechanism.

This version removes two optimistic assumptions from v2:

1. It never substitutes the first order-book snapshot *after* simulated order
   arrival as though that future book were known at arrival.  Arrival is
   bracketed by the last snapshot at-or-before and the first snapshot at-or-
   after the arrival timestamp.
2. It never fills more liquidity than survived across the signal book and both
   arrival-bracketing books.  Quantity is limited to the intersection of those
   three displayed ladders after a 75% depth haircut; execution price is the
   worst quantile price among the three ladders plus one cent adverse slippage.

Every scheduled slot in the full elapsed development span is in the statistical
denominator.  Uncaptured gaps, incomplete markets, no-signals, and failed fills
are zero.  Ranking uses the minimum one-sided 95% Student-t lower bound from the
full elapsed span and the fully-covered capture union, using UTC-aligned,
non-overlapping three-hour blocks.

No holdout capture is opened by this script.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

import explore_brownian_average as old
import explore_brownian_v2 as old_eval

edge = old.edge

ANNUAL_SLOTS = 96 * 365
CAPTURE_SECONDS = 10800
SLOT_SECONDS = 900
BLOCK_SECONDS = 10800
MAX_QTY = 100.0
BASE_DEPTH_FRACTION = 0.25
BASE_ADVERSE_DOLLARS = 0.01
MAX_BRACKET_AGE_NS = 1_750_000_000
MIN_SIGNAL_QTY = 10.0
SLIP_LIMIT_DOLLARS = 0.02


@dataclass(frozen=True)
class RobustOpp:
    capture: str
    ticker: str
    signal_ns: int
    seconds_to_close: float
    side: str
    p_side: float
    ask_avg: float
    ask_worst: float
    outcome_yes: bool
    executions: dict[int, tuple[float, float, float, float | None]]


def contiguous_states(m, window: int):
    """Causal states with trailing volatility computed on contiguous seconds."""
    rows = sorted(m.brti, key=lambda x: x.recv_ns)
    snaps = sorted(m.snapshots, key=lambda x: x.recv_ns)
    pointer = 0
    known: dict[int, float] = {}
    out = []
    end = m.close_ms // 1000
    start = end - 60

    for si, snap in enumerate(snaps):
        while pointer < len(rows) and rows[pointer].recv_ns <= snap.recv_ns:
            r = rows[pointer]
            known[r.ref_sec] = float(r.brti)
            pointer += 1
        if not known:
            continue

        latest_sec = max(known)
        recent_rev = []
        sec = latest_sec
        while sec in known and len(recent_rev) < window + 1:
            recent_rev.append(known[sec])
            sec -= 1
        recent = list(reversed(recent_rev))
        if len(recent) < max(20, window // 2):
            continue
        diffs = np.diff(np.asarray(recent, dtype=float))
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
        if observed:
            remaining = 60 - len(observed)
            mean_final = (sum(observed) + remaining * observed[-1]) / 60.0
            variance_factor = (
                remaining * (remaining + 1) * (2 * remaining + 1)
                / (6.0 * 60.0 * 60.0)
            )
            model_horizon = float(remaining)
        else:
            current = known[latest_sec]
            h = max(0, start - latest_sec - 1)
            weighted_future_squares = 60 * 61 * 121 / 6.0
            variance_factor = h + weighted_future_squares / (60.0 * 60.0)
            mean_final = current
            model_horizon = float(h + 60)

        seconds_to_close = (m.close_ms * 1_000_000 - snap.recv_ns) / 1e9
        if seconds_to_close <= 0:
            continue
        out.append(old.State(
            snap_index=si,
            recv_ns=snap.recv_ns,
            seconds_to_close=seconds_to_close,
            model_horizon=model_horizon,
            mean_final=mean_final,
            sigma_1s=sigma,
            variance_factor=variance_factor,
        ))
    return out


def _book_segments(levels, *, depth_fraction: float, limit: float = 1.0):
    out: list[tuple[float, float]] = []
    total = 0.0
    for price, displayed in sorted(levels, key=lambda x: x[0]):
        price = float(price)
        if price > limit + 1e-12 or total >= MAX_QTY - 1e-12:
            break
        qty = min(MAX_QTY - total, max(0.0, float(displayed)) * depth_fraction)
        if qty <= 0:
            continue
        out.append((price, qty))
        total += qty
    return out


def _book_total(segments):
    return sum(q for _p, q in segments)


def _quantile_price(segments, q: float) -> float:
    cumulative = 0.0
    for price, qty in segments:
        cumulative += qty
        if q <= cumulative + 1e-12:
            return float(price)
    raise ValueError("quantile exceeds available book")


def _signal_quote(levels, *, depth_fraction: float):
    segments = _book_segments(levels, depth_fraction=depth_fraction)
    total = _book_total(segments)
    if total < MIN_SIGNAL_QTY:
        return None
    cost = sum(p * q for p, q in segments)
    return cost / total, segments[-1][0], total


def persistent_intersection_fill(
    signal_levels,
    before_levels,
    after_levels,
    *,
    limit: float,
    depth_fraction: float,
    adverse_dollars: float,
    fee_rate: float,
):
    """Lower-bound IOC fill using liquidity persistent in all three books.

    At each quantity quantile, price is the maximum of the signal, pre-arrival,
    and post-arrival ladder prices, then adverse slippage is added.  Quantity
    cannot exceed the smallest haircutted ladder.  This awards no replenishment
    and no transient improvement that appears only after arrival.
    """
    books = [
        _book_segments(signal_levels, depth_fraction=depth_fraction, limit=limit),
        _book_segments(before_levels, depth_fraction=depth_fraction, limit=limit),
        _book_segments(after_levels, depth_fraction=depth_fraction, limit=limit),
    ]
    if any(not book for book in books):
        return 0.0, 0.0, 0.0, None
    available = min(MAX_QTY, *(_book_total(book) for book in books))
    if available <= 0:
        return 0.0, 0.0, 0.0, None

    boundaries = {0.0, available}
    for book in books:
        cumulative = 0.0
        for _price, qty in book:
            cumulative += qty
            if 0 < cumulative < available:
                boundaries.add(cumulative)
    cuts = sorted(boundaries)

    qty_total = 0.0
    cost = 0.0
    fees = 0.0
    worst = None
    for lo, hi in zip(cuts, cuts[1:]):
        if hi <= lo:
            continue
        q_mid = (lo + hi) / 2.0
        price = max(_quantile_price(book, q_mid) for book in books)
        price = min(0.9999, price + adverse_dollars)
        if price > limit + 1e-12:
            break
        qty = hi - lo
        qty_total += qty
        cost += qty * price
        raw_fee = fee_rate * qty * price * (1.0 - price)
        # Conservative per-segment round-up to the nearest cent.
        fees += math.ceil(raw_fee * 100.0 - 1e-12) / 100.0
        worst = price
    return qty_total, cost, fees, worst


def build_robust_opps(
    markets,
    window: int,
    multiplier: float,
    latencies,
    *,
    depth_fraction: float = BASE_DEPTH_FRACTION,
    adverse_dollars: float = BASE_ADVERSE_DOLLARS,
    fee_rate: float = 0.07,
):
    out = []
    for m in markets:
        snaps = sorted(m.snapshots, key=lambda x: x.recv_ns)
        snap_ns = [s.recv_ns for s in snaps]
        seen_second = set()
        for st in contiguous_states(m, window):
            second_bucket = int(math.floor(st.seconds_to_close))
            if second_bucket in seen_second:
                continue
            seen_second.add(second_bucket)
            sd = max(
                0.01,
                st.sigma_1s * multiplier * math.sqrt(max(st.variance_factor, 1e-12)),
            )
            p_yes = float(norm.cdf((st.mean_final - m.strike) / sd))
            if p_yes >= 0.5:
                side = "yes"
                p_side = p_yes
                signal_levels = snaps[st.snap_index].yes_asks
            else:
                side = "no"
                p_side = 1.0 - p_yes
                signal_levels = snaps[st.snap_index].no_asks

            signal_quote = _signal_quote(signal_levels, depth_fraction=depth_fraction)
            if signal_quote is None:
                continue
            ask_avg, ask_worst, _signal_qty = signal_quote
            limit = min(
                0.999,
                ask_worst + SLIP_LIMIT_DOLLARS,
                max(0.0, p_side - 0.005),
            )

            executions = {}
            for latency in latencies:
                target = st.recv_ns + latency * 1_000_000
                before_i = bisect.bisect_right(snap_ns, target) - 1
                after_i = bisect.bisect_left(snap_ns, target)
                if (
                    before_i < st.snap_index
                    or after_i >= len(snaps)
                    or target - snap_ns[before_i] > MAX_BRACKET_AGE_NS
                    or snap_ns[after_i] - target > MAX_BRACKET_AGE_NS
                ):
                    executions[latency] = (0.0, 0.0, 0.0, None)
                    continue
                before_levels = (
                    snaps[before_i].yes_asks if side == "yes" else snaps[before_i].no_asks
                )
                after_levels = (
                    snaps[after_i].yes_asks if side == "yes" else snaps[after_i].no_asks
                )
                executions[latency] = persistent_intersection_fill(
                    signal_levels,
                    before_levels,
                    after_levels,
                    limit=limit,
                    depth_fraction=depth_fraction,
                    adverse_dollars=adverse_dollars,
                    fee_rate=fee_rate,
                )

            out.append(RobustOpp(
                capture=m.capture,
                ticker=m.ticker,
                signal_ns=st.recv_ns,
                seconds_to_close=st.seconds_to_close,
                side=side,
                p_side=p_side,
                ask_avg=ask_avg,
                ask_worst=ask_worst,
                outcome_yes=bool(m.outcome_yes),
                executions=executions,
            ))
    return out


def prepare(opps):
    by = defaultdict(list)
    for o in opps:
        by[(o.capture, o.ticker)].append(o)
    for values in by.values():
        values.sort(key=lambda x: x.signal_ns)
    return by


def evaluate_records(params, prepared, markets, *, fee_rate: float = 0.07):
    latency, min_t, max_t, pmin, edge_min, max_ask = params
    records = []
    for m in markets:
        chosen = None
        for o in prepared.get((m.capture, m.ticker), []):
            if o.seconds_to_close > max_t:
                continue
            if o.seconds_to_close < min_t:
                break
            if o.p_side < pmin or o.ask_avg > max_ask:
                continue
            fee_pc = fee_rate * o.ask_avg * (1.0 - o.ask_avg)
            if o.p_side - o.ask_avg - fee_pc < edge_min:
                continue
            chosen = o
            break
        if chosen is None:
            records.append({
                "capture": m.capture,
                "ticker": m.ticker,
                "open_s": m.open_ms // 1000,
                "pnl": 0.0,
                "qty": 0.0,
                "win": None,
                "signal": False,
            })
            continue
        qty, cost, fees, worst = chosen.executions[latency]
        if qty <= 0:
            pnl = 0.0
            win = None
        else:
            win = (
                (chosen.side == "yes" and chosen.outcome_yes)
                or (chosen.side == "no" and not chosen.outcome_yes)
            )
            pnl = (qty if win else 0.0) - cost - fees
        records.append({
            "capture": m.capture,
            "ticker": m.ticker,
            "open_s": m.open_ms // 1000,
            "pnl": pnl,
            "qty": qty,
            "fees": fees,
            "win": win,
            "signal": True,
            "side": chosen.side,
            "p_side": chosen.p_side,
            "signal_ask": chosen.ask_avg,
            "execution_worst": worst,
            "seconds_to_close": chosen.seconds_to_close,
        })
    records.sort(key=lambda r: (r["open_s"], r["ticker"]))
    return records


def capture_start(name: str) -> int:
    return int(edge.capture_dt(Path(name)).timestamp())


def slot_range(start: int, end: int):
    first = ((start + SLOT_SECONDS - 1) // SLOT_SECONDS) * SLOT_SECONDS
    last = ((end - SLOT_SECONDS) // SLOT_SECONDS) * SLOT_SECONDS
    return list(range(first, last + 1, SLOT_SECONDS)) if last >= first else []


def denominator_sets(captures):
    windows = [(capture_start(c), capture_start(c) + CAPTURE_SECONDS) for c in captures]
    union = sorted({s for a, b in windows for s in slot_range(a, b)})
    span = slot_range(min(a for a, _b in windows), max(b for _a, b in windows))
    return union, span, windows


def fixed_utc_blocks(slots, pnl_by_open):
    if not slots:
        return []
    slot_set = set(slots)
    first_block = (min(slots) // BLOCK_SECONDS) * BLOCK_SECONDS
    last_block = (max(slots) // BLOCK_SECONDS) * BLOCK_SECONDS
    blocks = []
    for block_start in range(first_block, last_block + 1, BLOCK_SECONDS):
        vals = []
        for i in range(12):
            s = block_start + i * SLOT_SECONDS
            vals.append(float(pnl_by_open.get(s, 0.0)) if s in slot_set else 0.0)
        blocks.append(float(np.mean(vals)))
    return blocks


def one_sided_lcb(block_means):
    arr = np.asarray(block_means, dtype=float)
    if len(arr) < 2:
        return -1e18
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    return float(
        (arr.mean() - student_t.ppf(0.95, df=len(arr) - 1) * se)
        * ANNUAL_SLOTS
    )


def summarize_frame(slots, pnl_by_open):
    values = [float(pnl_by_open.get(s, 0.0)) for s in slots]
    n = len(values)
    arr = np.asarray(values, dtype=float)
    blocks = fixed_utc_blocks(slots, pnl_by_open)
    thirds = [float(x.mean()) for x in np.array_split(arr, 3) if len(x)]
    ordered = sorted(values, reverse=True)
    cut = max(1, math.ceil(0.10 * n)) if n else 0
    after_top = sum(ordered[cut:]) / n if n else -1e18

    if blocks:
        strongest_start = (min(slots) // BLOCK_SECONDS) * BLOCK_SECONDS
        strongest_index = max(range(len(blocks)), key=lambda i: blocks[i])
        remove_start = strongest_start + strongest_index * BLOCK_SECONDS
        kept = [
            float(pnl_by_open.get(s, 0.0))
            for s in slots
            if not (remove_start <= s < remove_start + BLOCK_SECONDS)
        ]
        after_block = sum(kept) / len(kept) if kept else -1e18
    else:
        after_block = -1e18

    positives = sorted((x for x in values if x > 0), reverse=True)
    positive_sum = sum(positives)
    return {
        "scheduled_slots": n,
        "total_pnl": float(sum(values)),
        "mean_per_slot": float(sum(values) / n) if n else -1e18,
        "annualized": float(sum(values) / n * ANNUAL_SLOTS) if n else -1e18,
        "fixed_utc_3h_blocks": blocks,
        "fixed_utc_3h_lcb95": one_sided_lcb(blocks),
        "chronological_thirds": thirds,
        "after_top10_mean": float(after_top),
        "after_strongest_3h_mean": float(after_block),
        "top1_positive_share": positives[0] / positive_sum if positives and positive_sum else 0.0,
        "top5_positive_share": sum(positives[:5]) / positive_sum if positives and positive_sum else 0.0,
    }


def summarize(params, records, union_slots, span_slots):
    # KXBTC15M has one contract per scheduled open.  A duplicate can only be a
    # capture overlap; retain the record with the larger conservative quantity,
    # breaking ties by lower P&L to avoid optimistic ownership selection.
    chosen = {}
    for r in records:
        s = int(r["open_s"])
        old_r = chosen.get(s)
        if old_r is None or (r["qty"], -r["pnl"]) > (old_r["qty"], -old_r["pnl"]):
            chosen[s] = r
    pnl_by_open = {s: float(r["pnl"]) for s, r in chosen.items()}
    union = summarize_frame(union_slots, pnl_by_open)
    span = summarize_frame(span_slots, pnl_by_open)
    trades = sum(r["qty"] > 0 for r in chosen.values())
    wins = sum(r.get("win") is True for r in chosen.values())
    losses = sum(r.get("win") is False for r in chosen.values())
    signals = sum(bool(r.get("signal")) for r in chosen.values())
    failed = sum(bool(r.get("signal")) and r["qty"] <= 0 for r in chosen.values())
    min_lcb = min(union["fixed_utc_3h_lcb95"], span["fixed_utc_3h_lcb95"])
    return {
        "params": {
            "latency_ms": params[0],
            "min_seconds_to_close": params[1],
            "max_seconds_to_close": params[2],
            "p_min": params[3],
            "edge_min": params[4],
            "max_ask": params[5],
        },
        "signals": signals,
        "failed_after_signal": failed,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "minimum_frame_lcb95": min_lcb,
        "capture_union": union,
        "full_elapsed_span": span,
    }


def robust_pass(r):
    frames = [r["capture_union"], r["full_elapsed_span"]]
    return (
        r["trades"] >= 20
        and r["minimum_frame_lcb95"] > 0
        and all(min(f["chronological_thirds"] or [-1]) > 0 for f in frames)
        and all(f["after_top10_mean"] > 0 for f in frames)
        and all(f["after_strongest_3h_mean"] > 0 for f in frames)
        and all(f["top1_positive_share"] <= 0.20 for f in frames)
        and all(f["top5_positive_share"] <= 0.60 for f in frames)
    )


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = [edge.read_kalshi(k, edge.read_perp(p)) for k, p in dev_pairs]
    groups = old.dedupe(groups)
    markets = [m for group in groups for m in group]
    captures = [k.name for k, _p in dev_pairs]
    union_slots, span_slots, windows = denominator_sets(captures)

    latencies = [2000, 3000, 5000]
    model_specs = [
        (60, 0.60),
        (60, 0.75),
        (60, 0.90),
        (120, 0.60),
        (120, 0.75),
        (120, 0.90),
        (240, 0.75),
    ]
    results = []
    prepared_by_model = {}
    for window, multiplier in model_specs:
        opps = build_robust_opps(
            markets,
            window,
            multiplier,
            latencies,
            depth_fraction=BASE_DEPTH_FRACTION,
            adverse_dollars=BASE_ADVERSE_DOLLARS,
            fee_rate=0.07,
        )
        prepared = prepare(opps)
        prepared_by_model[(window, multiplier)] = prepared
        for latency in latencies:
            for min_t in [45, 60, 90]:
                for max_t in [300, 450, 600]:
                    for pmin in [0.75, 0.80, 0.85, 0.90, 0.95]:
                        for edge_min in [0.01, 0.02, 0.03, 0.05, 0.08]:
                            for max_ask in [0.75, 0.80, 0.85, 0.90, 0.95]:
                                params = (latency, min_t, max_t, pmin, edge_min, max_ask)
                                records = evaluate_records(params, prepared, markets)
                                s = summarize(params, records, union_slots, span_slots)
                                s["model"] = {
                                    "sigma_window": window,
                                    "sigma_multiplier": multiplier,
                                }
                                results.append(s)

    ranked = sorted(
        results,
        key=lambda r: (
            r["minimum_frame_lcb95"],
            min(r["capture_union"]["after_top10_mean"], r["full_elapsed_span"]["after_top10_mean"]),
            min(r["capture_union"]["annualized"], r["full_elapsed_span"]["annualized"]),
        ),
        reverse=True,
    )
    robust = [r for r in ranked if robust_pass(r)]
    best = robust[0] if robust else ranked[0]

    model = best["model"]
    p = best["params"]
    params = (
        p["latency_ms"],
        p["min_seconds_to_close"],
        p["max_seconds_to_close"],
        p["p_min"],
        p["edge_min"],
        p["max_ask"],
    )
    best_records = evaluate_records(
        params,
        prepared_by_model[(model["sigma_window"], model["sigma_multiplier"])],
        markets,
    )

    report = {
        "round": "persistent-liquidity settlement-average variance v3",
        "execution": {
            "order_type": "marketable IOC",
            "max_contract_notional": MAX_QTY,
            "depth_fraction": BASE_DEPTH_FRACTION,
            "adverse_dollars_per_contract": BASE_ADVERSE_DOLLARS,
            "max_bracket_age_ns": MAX_BRACKET_AGE_NS,
            "liquidity_rule": "quantity intersection of signal, pre-arrival, and post-arrival books; worst quantile price",
            "signal_liquidity_replenishment_allowed": False,
            "fee_rate": 0.07,
        },
        "opened_captures": captures,
        "untouched_after_development": [k.name for k, _p in pairs[edge.DEV_CAPTURE_COUNT:]],
        "capture_windows": [
            {"capture": c, "start_s": a, "end_s": b}
            for c, (a, b) in zip(captures, windows)
        ],
        "market_count": len(markets),
        "capture_union_slots": len(union_slots),
        "full_elapsed_span_slots": len(span_slots),
        "grid_count": len(results),
        "robust_count": len(robust),
        "best": best,
        "best_records": best_records,
        "top100_robust": robust[:100],
        "top100_all": ranked[:100],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "scheduled_brownian_v3.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
