#!/usr/bin/env python3
"""Run exploratory round 1 with a linear-time causal state cache.

The original research logic remains unchanged. This wrapper only replaces the
naive repeated BRTI scan with an equivalent one-pass computation per market.
"""
from __future__ import annotations

import math

import numpy as np

import explore_v1 as edge


_CACHE: dict[int, dict[int, tuple[int, float, float, float] | None]] = {}


def build_cache(m: edge.Market) -> dict[int, tuple[int, float, float, float] | None]:
    rows = sorted(m.brti, key=lambda x: x.recv_ns)
    snaps = sorted(m.snapshots, key=lambda x: x.recv_ns)
    end = m.close_ms // 1000
    start = end - 60
    values: list[float | None] = [None] * 60
    pointer = 0
    out: dict[int, tuple[int, float, float, float] | None] = {}

    for snap in snaps:
        while pointer < len(rows) and rows[pointer].recv_ns <= snap.recv_ns:
            row = rows[pointer]
            if start <= row.ref_sec < end:
                values[row.ref_sec - start] = row.brti
            pointer += 1

        prefix: list[float] = []
        for value in values:
            if value is None:
                break
            prefix.append(float(value))
        if not prefix:
            out[snap.recv_ns] = None
            continue

        remaining = 60 - len(prefix)
        current = prefix[-1]
        forecast = (sum(prefix) + remaining * current) / 60.0
        recent = prefix[-11:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        sigma = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0
        out[snap.recv_ns] = (remaining, forecast, sigma, current)

    return out


def fast_state(m: edge.Market, recv_ns: int):
    key = id(m)
    cache = _CACHE.get(key)
    if cache is None:
        cache = build_cache(m)
        _CACHE[key] = cache
    return cache.get(recv_ns)


edge.brti_state_at = fast_state

if __name__ == "__main__":
    edge.main()
