#!/usr/bin/env python3
"""Development-only feature audit for the persistent-liquidity BRTI mechanism.

The purpose is diagnosis, not a holdout test.  It reconstructs the exact v3
candidate on the eleven development captures, computes only information known
before each candidate order, and tests mechanism-distinct causal vetoes:

* probability hysteresis (confidence must persist),
* side stability (the modeled winner must not have recently flipped),
* confidence trend (confidence must not be collapsing), and
* delayed confirmation (the signal must survive additional seconds before the
  ordinary execution latency begins).

Every rule is evaluated on the same scheduled-slot denominators as v3.  Later
captures remain unopened.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import optimize_scheduled_brownian_v3 as v3
import explore_brownian_average as old

edge = v3.edge

BASE_MODEL = (60, 0.60)
BASE_PARAMS = (2000, 45, 450, 0.90, 0.01, 0.95)


def net_edge(o, fee_rate=0.07):
    return o.p_side - o.ask_avg - fee_rate * o.ask_avg * (1.0 - o.ask_avg)


def chosen_side_probability(o, side):
    return o.p_side if o.side == side else 1.0 - o.p_side


def history_features(values, idx):
    current = values[idx]
    now = current.signal_ns
    side = current.side
    feats = {}

    same_run = 0
    p_runs = {0.75: 0, 0.80: 0, 0.85: 0, 0.90: 0, 0.925: 0, 0.95: 0}
    edge_runs = {0.00: 0, 0.01: 0, 0.02: 0, 0.03: 0, 0.05: 0}
    p_alive = {k: True for k in p_runs}
    e_alive = {k: True for k in edge_runs}
    for j in range(idx, -1, -1):
        x = values[j]
        if x.side != side:
            break
        same_run += 1
        for threshold in p_runs:
            if p_alive[threshold] and x.p_side >= threshold:
                p_runs[threshold] += 1
            else:
                p_alive[threshold] = False
        for threshold in edge_runs:
            if e_alive[threshold] and net_edge(x) >= threshold:
                edge_runs[threshold] += 1
            else:
                e_alive[threshold] = False

    feats["same_side_run"] = same_run
    for threshold, count in p_runs.items():
        feats[f"p_run_{int(round(threshold * 1000)):03d}"] = count
    for threshold, count in edge_runs.items():
        feats[f"edge_run_{int(round(threshold * 1000)):03d}"] = count

    for seconds in [3, 5, 10, 15, 30, 60, 120]:
        window = [x for x in values[: idx + 1] if now - x.signal_ns <= seconds * 1_000_000_000]
        ps = [chosen_side_probability(x, side) for x in window]
        flips = sum(
            1 for a, b in zip(window, window[1:]) if a.side != b.side
        )
        feats[f"p_min_{seconds}s"] = min(ps) if ps else current.p_side
        feats[f"p_mean_{seconds}s"] = sum(ps) / len(ps) if ps else current.p_side
        feats[f"flips_{seconds}s"] = flips
        target = now - seconds * 1_000_000_000
        before = [x for x in values[: idx + 1] if x.signal_ns <= target]
        if before:
            previous_p = chosen_side_probability(before[-1], side)
            feats[f"p_delta_{seconds}s"] = current.p_side - previous_p
        else:
            feats[f"p_delta_{seconds}s"] = None
    return feats


def base_pass(o, params=BASE_PARAMS):
    _latency, min_t, max_t, pmin, edge_min, max_ask = params
    return (
        min_t <= o.seconds_to_close <= max_t
        and o.p_side >= pmin
        and o.ask_avg <= max_ask
        and net_edge(o) >= edge_min
    )


def make_record(m, o, latency):
    if o is None:
        return {
            "capture": m.capture,
            "ticker": m.ticker,
            "open_s": m.open_ms // 1000,
            "pnl": 0.0,
            "qty": 0.0,
            "win": None,
            "signal": False,
        }
    qty, cost, fees, worst = o.executions[latency]
    if qty <= 0:
        win = None
        pnl = 0.0
    else:
        win = (
            (o.side == "yes" and o.outcome_yes)
            or (o.side == "no" and not o.outcome_yes)
        )
        pnl = (qty if win else 0.0) - cost - fees
    return {
        "capture": m.capture,
        "ticker": m.ticker,
        "open_s": m.open_ms // 1000,
        "pnl": pnl,
        "qty": qty,
        "fees": fees,
        "win": win,
        "signal": True,
        "side": o.side,
        "p_side": o.p_side,
        "signal_ask": o.ask_avg,
        "execution_worst": worst,
        "seconds_to_close": o.seconds_to_close,
        "signal_ns": o.signal_ns,
    }


def choose_with_rule(m, values, rule):
    latency = BASE_PARAMS[0]
    for idx, o in enumerate(values):
        if o.seconds_to_close > BASE_PARAMS[2]:
            continue
        if o.seconds_to_close < BASE_PARAMS[1]:
            break
        if not base_pass(o):
            continue
        features = history_features(values, idx)
        if rule(features, o, values, idx):
            return make_record(m, o, latency), features
    return make_record(m, None, latency), None


def evaluate_rule(name, rule, markets, prepared, union_slots, span_slots):
    records = []
    features = []
    for m in markets:
        rec, f = choose_with_rule(m, prepared.get((m.capture, m.ticker), []), rule)
        records.append(rec)
        if f is not None:
            features.append({"ticker": m.ticker, "capture": m.capture, **rec, **f})
    summary = v3.summarize(BASE_PARAMS, records, union_slots, span_slots)
    summary["rule"] = name
    summary["selected_feature_rows"] = features
    return summary


def compact(result):
    out = {k: v for k, v in result.items() if k != "selected_feature_rows"}
    return out


def lcb_key(r):
    return (
        r["capture_union"]["fixed_utc_3h_lcb95"],
        r["full_elapsed_span"]["fixed_utc_3h_lcb95"],
        r["capture_union"]["after_top10_mean"],
    )


def main():
    pairs = edge.pair_files()
    dev_pairs = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = [edge.read_kalshi(k, edge.read_perp(p)) for k, p in dev_pairs]
    groups = old.dedupe(groups)
    markets = [m for group in groups for m in group]
    captures = [k.name for k, _p in dev_pairs]
    union_slots, span_slots, _windows = v3.denominator_sets(captures)

    opps = v3.build_robust_opps(
        markets,
        BASE_MODEL[0],
        BASE_MODEL[1],
        [BASE_PARAMS[0]],
        depth_fraction=v3.BASE_DEPTH_FRACTION,
        adverse_dollars=v3.BASE_ADVERSE_DOLLARS,
        fee_rate=0.07,
    )
    prepared = v3.prepare(opps)

    rules = [("base", lambda f, o, values, idx: True)]

    # Family A: confidence must persist for N consecutive causal snapshots.
    for p_floor in [0.80, 0.85, 0.90, 0.925, 0.95]:
        key = f"p_run_{int(round(p_floor * 1000)):03d}"
        for n in [2, 3, 5, 8, 10, 15, 20]:
            rules.append((
                f"persistence_p{p_floor:.3f}_n{n}",
                lambda f, o, values, idx, key=key, n=n: f[key] >= n,
            ))

    # Family B: no modeled-side flip within a causal lookback window.
    for seconds in [5, 10, 15, 30, 60, 120]:
        for max_flips in [0, 1, 2]:
            rules.append((
                f"side_stability_{seconds}s_flips_le_{max_flips}",
                lambda f, o, values, idx, seconds=seconds, max_flips=max_flips:
                    f[f"flips_{seconds}s"] <= max_flips,
            ))

    # Family C: all chosen-side probabilities in the recent window exceed a floor.
    for seconds in [3, 5, 10, 15, 30, 60]:
        for floor in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            rules.append((
                f"window_min_p_{seconds}s_ge_{floor:.2f}",
                lambda f, o, values, idx, seconds=seconds, floor=floor:
                    f[f"p_min_{seconds}s"] >= floor,
            ))

    # Family D: confidence is stable or rising over a causal horizon.
    for seconds in [3, 5, 10, 15, 30, 60]:
        for delta in [-0.10, -0.05, -0.02, 0.00, 0.02, 0.05]:
            rules.append((
                f"confidence_delta_{seconds}s_ge_{delta:+.2f}",
                lambda f, o, values, idx, seconds=seconds, delta=delta:
                    f[f"p_delta_{seconds}s"] is not None
                    and f[f"p_delta_{seconds}s"] >= delta,
            ))

    results = [evaluate_rule(name, rule, markets, prepared, union_slots, span_slots)
               for name, rule in rules]
    ranked = sorted(results, key=lcb_key, reverse=True)

    base = next(r for r in results if r["rule"] == "base")
    base_rows = base["selected_feature_rows"]
    losses = [r for r in base_rows if r.get("win") is False]
    winners = [r for r in base_rows if r.get("win") is True]

    family_best = {}
    for prefix in ["persistence_", "side_stability_", "window_min_p_", "confidence_delta_"]:
        group = [r for r in results if r["rule"].startswith(prefix)]
        family_best[prefix.rstrip("_")] = compact(max(group, key=lcb_key))

    report = {
        "round": "v3 causal feature and mechanism audit",
        "base_model": {"sigma_window": BASE_MODEL[0], "sigma_multiplier": BASE_MODEL[1]},
        "base_params": {
            "latency_ms": BASE_PARAMS[0],
            "min_seconds_to_close": BASE_PARAMS[1],
            "max_seconds_to_close": BASE_PARAMS[2],
            "p_min": BASE_PARAMS[3],
            "edge_min": BASE_PARAMS[4],
            "max_ask": BASE_PARAMS[5],
        },
        "opened_captures": captures,
        "untouched_after_development": [k.name for k, _p in pairs[edge.DEV_CAPTURE_COUNT:]],
        "rule_count": len(results),
        "base": compact(base),
        "base_losses": losses,
        "winner_feature_rows": winners,
        "family_best": family_best,
        "top100": [compact(r) for r in ranked[:100]],
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "brownian_v3_feature_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps({
        "base_losses": losses,
        "family_best": family_best,
        "top10": [compact(r) for r in ranked[:10]],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
