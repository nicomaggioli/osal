#!/usr/bin/env python3
"""Explain why exploratory round 1 generated or rejected opportunities."""
from __future__ import annotations

import json
from collections import Counter

import explore_fast as fast

edge = fast.edge


def side_depth(levels):
    raw = sum(q for _, q in levels)
    haircut = raw * edge.DEPTH_HAIRCUT
    best = levels[0][0] if levels else None
    return {"levels": len(levels), "raw_qty": raw, "haircut_qty": haircut, "best_ask": best}


def main():
    pairs = edge.pair_files()
    dev = pairs[: edge.DEV_CAPTURE_COUNT]
    groups = []
    for k, p in dev:
        brti = edge.read_perp(p)
        groups.append(edge.read_kalshi(k, brti))
    train = [m for g in groups[: edge.TRAIN_CAPTURE_COUNT] for m in g]
    val = [m for g in groups[edge.TRAIN_CAPTURE_COUNT :] for m in g]
    model = edge.fit_error_model(edge.state_examples(train))

    reason = Counter()
    markets = []
    for m in val:
        seen = set()
        states = []
        latest = None
        for s in m.snapshots:
            st = fast.fast_state(m, s.recv_ns)
            if st is None:
                reason["state_none"] += 1
                continue
            rem, fc, sigma, cur = st
            latest = {"remaining": rem, "forecast": fc, "sigma": sigma, "current": cur,
                      "seconds_to_close_at_snapshot": (m.close_ms * 1_000_000 - s.recv_ns) / 1e9,
                      "yes_depth": side_depth(s.yes_asks), "no_depth": side_depth(s.no_asks)}
            if rem > 30:
                reason["remaining_gt_30"] += 1
                continue
            if rem in seen:
                reason["duplicate_remaining"] += 1
                continue
            seen.add(rem)
            mu, sd, n = model[rem]
            z = (fc + mu - m.strike) / sd
            p_yes = float(edge.norm.cdf(z))
            side = "yes" if p_yes >= 0.5 else "no"
            p_side = p_yes if side == "yes" else 1.0 - p_yes
            levels = s.yes_asks if side == "yes" else s.no_asks
            aa = edge.avg_ask(levels)
            if aa is None:
                reason["avg_ask_none"] += 1
            else:
                reason["opportunity"] += 1
            states.append({
                "remaining": rem, "forecast": fc, "strike": m.strike,
                "truth": m.true_end_avg, "forecast_error": m.true_end_avg - fc,
                "p_yes": p_yes, "p_side": p_side, "side": side,
                "model_mu": mu, "model_sd": sd, "model_n": n,
                "avg_ask": aa,
                "yes_depth": side_depth(s.yes_asks),
                "no_depth": side_depth(s.no_asks),
                "seconds_to_close_at_snapshot": (m.close_ms * 1_000_000 - s.recv_ns) / 1e9,
            })
        markets.append({
            "capture": m.capture, "ticker": m.ticker, "strike": m.strike,
            "truth": m.true_end_avg, "outcome_yes": m.outcome_yes,
            "snapshot_count": len(m.snapshots), "latest_state": latest,
            "states_30_to_0": states,
        })

    report = {
        "train_markets": len(train), "validation_markets": len(val),
        "reason_counts": dict(reason), "markets": markets,
    }
    edge.OUT_DIR.mkdir(exist_ok=True)
    (edge.OUT_DIR / "diagnose_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"train_markets": len(train), "validation_markets": len(val), "reason_counts": dict(reason)}, indent=2))


if __name__ == "__main__":
    main()
