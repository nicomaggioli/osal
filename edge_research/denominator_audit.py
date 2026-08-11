#!/usr/bin/env python3
"""Conservative denominator and serial-dependence audit.

Every scheduled 15-minute slot covered by a capture is included, even when no
complete market, no signal, or no fill is available. A second audit includes
all scheduled slots across the entire elapsed development span, assigning zero
P&L to uncaptured gaps. This prevents incomplete data or inactive intervals
from silently disappearing from the annualization denominator.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import explore_v1 as edge

DURATION_S = 10800
SLOT_S = 900
ANNUAL_SLOTS = 96 * 365


def start_s(name: str) -> int:
    return int(edge.capture_dt(Path(name)).timestamp())


def slot_range(a: int, b: int):
    first = ((a + SLOT_S - 1) // SLOT_S) * SLOT_S
    last_open = ((b - SLOT_S) // SLOT_S) * SLOT_S
    return list(range(first, last_open + 1, SLOT_S)) if last_open >= first else []


def one_sided_t_lcb(block_means):
    vals = np.array(block_means, dtype=float)
    if len(vals) < 2:
        return -1e18
    se = vals.std(ddof=1) / math.sqrt(len(vals))
    return (vals.mean() - student_t.ppf(0.95, df=len(vals) - 1) * se) * ANNUAL_SLOTS


def fixed_blocks(values, size):
    return [float(np.mean(values[i:i + size])) for i in range(0, len(values), size) if len(values[i:i + size]) == size]


def moving_block_lcb(values, block_len, reps=50000, seed=20260811):
    rng = random.Random(seed + block_len)
    vals = list(map(float, values))
    n = len(vals)
    starts = list(range(0, n - block_len + 1))
    if not starts or n == 0:
        return -1e18
    means = []
    blocks_needed = math.ceil(n / block_len)
    for _ in range(reps):
        sample = []
        for _j in range(blocks_needed):
            s = rng.choice(starts)
            sample.extend(vals[s:s + block_len])
        means.append(sum(sample[:n]) / n * ANNUAL_SLOTS)
    means.sort()
    return means[max(0, math.floor(0.05 * reps) - 1)]


def summarize(slots, pnl_by_open):
    values = [float(pnl_by_open.get(s, 0.0)) for s in slots]
    n = len(values)
    mean = sum(values) / n if n else 0.0
    thirds = [float(x.mean()) for x in np.array_split(np.array(values, dtype=float), 3) if len(x)]
    ordered = sorted(values, reverse=True)
    remove_n = max(1, math.ceil(0.10 * n)) if n else 0
    after_top = sum(ordered[remove_n:]) / n if n else 0.0
    return {
        "scheduled_slots": n,
        "slots_with_parsed_market": sum(s in pnl_by_open for s in slots),
        "zero_or_missing_slots": sum(float(pnl_by_open.get(s, 0.0)) == 0.0 for s in slots),
        "total_pnl": sum(values),
        "mean_per_scheduled_slot": mean,
        "annualized": mean * ANNUAL_SLOTS,
        "chronological_thirds": thirds,
        "after_top10_mean": after_top,
        "fixed_3h_block_count": len(fixed_blocks(values, 12)),
        "fixed_3h_t_lcb95": one_sided_t_lcb(fixed_blocks(values, 12)),
        "fixed_6h_block_count": len(fixed_blocks(values, 24)),
        "fixed_6h_t_lcb95": one_sided_t_lcb(fixed_blocks(values, 24)),
        "moving_block_3h_lcb95": moving_block_lcb(values, 12),
        "moving_block_6h_lcb95": moving_block_lcb(values, 24),
    }


def main():
    report = json.loads(Path("edge_outputs/brownian_average_v2.json").read_text())
    records = report["best"]["records"]
    pnl_by_open = {int(r["open_ms"]) // 1000: float(r["pnl"]) for r in records}
    captures = report["opened_captures"]
    windows = [(start_s(c), start_s(c) + DURATION_S) for c in captures]

    union = sorted({s for a, b in windows for s in slot_range(a, b)})
    span = slot_range(min(a for a, _ in windows), max(b for _, b in windows))

    result = {
        "annual_slots": ANNUAL_SLOTS,
        "capture_windows": [{"capture": c, "start_s": a, "end_s": b} for c, (a, b) in zip(captures, windows)],
        "capture_union": summarize(union, pnl_by_open),
        "full_elapsed_span": summarize(span, pnl_by_open),
        "records_outside_capture_union": sorted(s for s in pnl_by_open if s not in set(union)),
    }
    Path("edge_outputs/denominator_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
