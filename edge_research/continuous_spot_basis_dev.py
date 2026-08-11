#!/usr/bin/env python3
"""Continuous-history development for basis-anchored averaging variance.

This script opens only July 27-28, 2026 (192 scheduled KXBTC15M markets).
July 29 and every later full-book capture remain unopened here.

Signal mechanism
----------------
The Kalshi floor strike fixes the BRTI arithmetic mean at the opening boundary.
Subtracting the simultaneous 60-second Binance BTCFDUSD mean estimates the
cross-venue level basis. That basis is added to subsequent Binance one-second
prices. The future 60-sample settlement mean is valued with the exact discrete
random-walk average variance. A qualifying same-side public taker print supplies
an observed ask anchor; after fixed latency, modeled IOC fills are capped by a
small fraction of independently observed same-side taker volume.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm, t as student_t

ROOT = Path(os.environ.get("BTPRED_ROOT", "/tmp/btpred-data"))
OUT = Path("edge_outputs")
OUT.mkdir(exist_ok=True)
TARGET_DAYS = {"2026-07-27", "2026-07-28"}
ANNUAL_SLOTS = 96 * 365
MAX_ORDER_QTY = 100.0
FEE_RATE = 0.07
SLIP = 0.02


@dataclass(frozen=True)
class MarketMeta:
    ticker: str
    open_s: int
    close_s: int
    strike: float
    yes_result: bool


@dataclass(frozen=True)
class SecRow:
    sec: int
    to_close: int
    yes_high: float | None
    yes_low: float | None
    yes_volume: float
    no_volume: float


@dataclass(frozen=True)
class Opportunity:
    index: int
    sec: int
    to_close: int
    side: str
    p_side: float
    ask: float
    observed_signal_volume: float


def parse_iso(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def fnum(v: str) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def load_spot(days: list[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for day in days:
        path = ROOT / "spot_1s" / f"BTCFDUSD-1s-{day}.zip"
        with zipfile.ZipFile(path) as z:
            with z.open(z.namelist()[0]) as raw:
                text = (line.decode("utf-8") for line in raw)
                for row in csv.reader(text):
                    stamp = int(row[0])
                    sec = stamp // (1_000_000 if stamp > 10**15 else 1000)
                    out[sec] = float(row[4])
    return out


def load_metadata() -> dict[str, MarketMeta]:
    path = ROOT / "kalshi_ticks" / "KXBTC15M-markets-2026-07-27_2026-07-30.csv.gz"
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            close_day = r["close_time"][:10]
            if close_day not in TARGET_DAYS:
                continue
            out[r["ticker"]] = MarketMeta(
                r["ticker"], parse_iso(r["open_time"]), parse_iso(r["close_time"]),
                float(r["floor_strike"]), r["result"].lower() == "yes",
            )
    return out


def load_dense(meta: dict[str, MarketMeta]) -> dict[str, list[SecRow]]:
    out: dict[str, list[SecRow]] = defaultdict(list)
    for day in sorted(TARGET_DAYS):
        path = ROOT / "kalshi_ticks" / f"KXBTC15M-1s-{day}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ticker = r["ticker"]
                if ticker not in meta:
                    continue
                out[ticker].append(SecRow(
                    sec=parse_iso(r["second_utc"]),
                    to_close=int(r["seconds_to_close"]),
                    yes_high=fnum(r["yes_high_dollars"]),
                    yes_low=fnum(r["yes_low_dollars"]),
                    yes_volume=float(r["taker_yes_contracts_fp"] or 0),
                    no_volume=float(r["taker_no_contracts_fp"] or 0),
                ))
    for rows in out.values():
        rows.sort(key=lambda x: x.sec)
    return out


def variance_factor_before(current_sec: int, close_s: int) -> float:
    start = close_s - 60
    h = max(0, start - current_sec - 1)
    return h + (60 * 61 * 121) / (6.0 * 60.0 * 60.0)


def variance_factor_inside(remaining: int) -> float:
    return remaining * (remaining + 1) * (2 * remaining + 1) / (6.0 * 60.0 * 60.0)


def build_opportunities(meta: MarketMeta, rows: list[SecRow], spot: dict[int, float], window: int, mult: float) -> list[Opportunity]:
    start_values = [spot.get(s) for s in range(meta.open_s - 60, meta.open_s)]
    if any(v is None for v in start_values):
        return []
    basis = meta.strike - sum(float(v) for v in start_values) / 60.0
    adjusted_inside: list[float] = []
    out: list[Opportunity] = []
    settlement_start = meta.close_s - 60

    for i, row in enumerate(rows):
        current = spot.get(row.sec)
        if current is None:
            continue
        hist = [spot.get(s) for s in range(row.sec - window, row.sec + 1)]
        if any(v is None for v in hist):
            continue
        diffs = [float(hist[j]) - float(hist[j - 1]) for j in range(1, len(hist))]
        sigma = max(0.05, float(np.std(diffs, ddof=1)))
        adjusted_current = float(current) + basis

        if row.sec < settlement_start:
            mean_final = adjusted_current
            vf = variance_factor_before(row.sec, meta.close_s)
        else:
            # Dense rows and spot data are exactly one second apart. Build the
            # observed prefix directly from the fixed settlement interval.
            observed = [spot.get(s) for s in range(settlement_start, min(row.sec, meta.close_s - 1) + 1)]
            if any(v is None for v in observed):
                continue
            vals = [float(v) + basis for v in observed]
            remaining = 60 - len(vals)
            mean_final = (sum(vals) + remaining * vals[-1]) / 60.0
            vf = variance_factor_inside(remaining)

        sd = max(0.01, sigma * mult * math.sqrt(max(vf, 1e-12)))
        p_yes = float(norm.cdf((mean_final - meta.strike) / sd))
        if p_yes >= 0.5:
            side = "yes"; p_side = p_yes
            if row.yes_high is None or row.yes_volume <= 0:
                continue
            ask = row.yes_high; volume = row.yes_volume
        else:
            side = "no"; p_side = 1.0 - p_yes
            if row.yes_low is None or row.no_volume <= 0:
                continue
            ask = 1.0 - row.yes_low; volume = row.no_volume
        if not (0.0 < ask < 1.0):
            continue
        out.append(Opportunity(i, row.sec, row.to_close, side, p_side, ask, volume))
    return out


def execute(meta: MarketMeta, rows: list[SecRow], o: Opportunity, latency: int, participation: float):
    j = o.index + latency
    if j >= len(rows) or rows[j].sec != o.sec + latency:
        return 0.0, 0.0, 0.0, None
    r = rows[j]
    if o.side == "yes":
        if r.yes_high is None or r.yes_volume <= 0:
            return 0.0, 0.0, 0.0, None
        price = r.yes_high; volume = r.yes_volume
    else:
        if r.yes_low is None or r.no_volume <= 0:
            return 0.0, 0.0, 0.0, None
        price = 1.0 - r.yes_low; volume = r.no_volume
    limit = min(0.999, o.ask + SLIP, max(0.0, o.p_side - 0.005))
    if price > limit + 1e-12:
        return 0.0, 0.0, 0.0, None
    qty = min(MAX_ORDER_QTY, participation * volume)
    if qty <= 0:
        return 0.0, 0.0, 0.0, None
    cost = qty * price
    fee = math.ceil(FEE_RATE * qty * price * (1.0 - price) * 100.0 - 1e-12) / 100.0
    win = (o.side == "yes" and meta.yes_result) or (o.side == "no" and not meta.yes_result)
    pnl = (qty if win else 0.0) - cost - fee
    return pnl, qty, fee, win


def block_lcb(values, block=12):
    blocks = [float(np.mean(values[i:i + block])) for i in range(0, len(values), block) if len(values[i:i + block]) == block]
    arr = np.array(blocks, dtype=float)
    if len(arr) < 2:
        return -1e18, blocks
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    lcb = (arr.mean() - student_t.ppf(0.95, df=len(arr) - 1) * se) * ANNUAL_SLOTS
    return float(lcb), blocks


def evaluate(params, metas, rows_by, opps_by):
    latency, participation, min_t, max_t, pmin, edge_min, max_ask = params
    records = []
    for m in metas:
        chosen = None
        for o in opps_by.get(m.ticker, []):
            if not (min_t <= o.to_close <= max_t):
                continue
            if o.p_side < pmin or o.ask > max_ask:
                continue
            fee_pc = FEE_RATE * o.ask * (1.0 - o.ask)
            if o.p_side - o.ask - fee_pc < edge_min:
                continue
            chosen = o
            break
        if chosen is None:
            records.append({"ticker": m.ticker, "open_s": m.open_s, "pnl": 0.0, "qty": 0.0, "win": None})
            continue
        pnl, qty, fee, win = execute(m, rows_by[m.ticker], chosen, latency, participation)
        records.append({"ticker": m.ticker, "open_s": m.open_s, "pnl": pnl, "qty": qty,
                        "win": win, "side": chosen.side, "p_side": chosen.p_side,
                        "ask": chosen.ask, "to_close": chosen.to_close, "fee": fee})
    records.sort(key=lambda r: r["open_s"])
    values = np.array([r["pnl"] for r in records], dtype=float)
    lcb, blocks = block_lcb(values.tolist(), 12)
    thirds = [float(x.mean()) for x in np.array_split(values, 3) if len(x)]
    days = defaultdict(list)
    for r in records:
        days[datetime.fromtimestamp(r["open_s"], timezone.utc).date().isoformat()].append(float(r["pnl"]))
    day_means = {d: sum(v) / len(v) for d, v in sorted(days.items())}
    ordered = sorted((float(x) for x in values), reverse=True)
    cut = max(1, math.ceil(0.10 * len(values)))
    after_top = sum(ordered[cut:]) / len(values)
    strongest_block = max(blocks) if blocks else 0.0
    after_block = (sum(values) - strongest_block * 12) / max(1, len(values) - 12)
    positives = sorted((float(x) for x in values if x > 0), reverse=True)
    pos_sum = sum(positives)
    return {
        "params": {"latency_s": latency, "participation": participation,
                   "min_seconds_to_close": min_t, "max_seconds_to_close": max_t,
                   "p_min": pmin, "edge_min": edge_min, "max_ask": max_ask},
        "markets": len(records), "trades": sum(r["qty"] > 0 for r in records),
        "wins": sum(r.get("win") is True for r in records),
        "losses": sum(r.get("win") is False for r in records),
        "failed_after_signal": sum(r["qty"] == 0 and "side" in r for r in records),
        "partial_fills": sum(0 < r["qty"] < MAX_ORDER_QTY for r in records),
        "total_pnl": float(values.sum()), "mean_per_scheduled_market": float(values.mean()),
        "annualized": float(values.mean()) * ANNUAL_SLOTS,
        "fixed_3h_lcb95": lcb, "block_means": blocks,
        "chronological_thirds": thirds, "day_means": day_means,
        "after_top10_mean": after_top, "after_strongest_3h_block_mean": after_block,
        "top1_positive_share": positives[0] / pos_sum if positives and pos_sum else 0.0,
        "top5_positive_share": sum(positives[:5]) / pos_sum if positives and pos_sum else 0.0,
        "records": records,
    }


def main():
    spot = load_spot(["2026-07-26", "2026-07-27", "2026-07-28"])
    meta_by = load_metadata()
    rows_by = load_dense(meta_by)
    metas = sorted((m for m in meta_by.values() if m.ticker in rows_by), key=lambda m: m.open_s)
    if len(metas) != 192:
        raise RuntimeError(f"expected 192 scheduled markets, found {len(metas)}")

    model_cache = {}
    results = []
    for window in [60, 120, 240]:
        for mult in [0.60, 0.75, 0.90, 1.10]:
            opps = {m.ticker: build_opportunities(m, rows_by[m.ticker], spot, window, mult) for m in metas}
            model_cache[(window, mult)] = opps
            for latency in [1, 2, 3, 5]:
                for participation in [0.05, 0.10, 0.25]:
                    for min_t in [45, 60, 90]:
                        for max_t in [300, 450, 600]:
                            for pmin in [0.70, 0.75, 0.80, 0.85, 0.90]:
                                for edge_min in [0.01, 0.02, 0.03, 0.05]:
                                    for max_ask in [0.80, 0.90, 0.95]:
                                        p = (latency, participation, min_t, max_t, pmin, edge_min, max_ask)
                                        r = evaluate(p, metas, rows_by, opps)
                                        r["model"] = {"sigma_window": window, "sigma_multiplier": mult}
                                        results.append(r)
    ranked = sorted(results, key=lambda r: (r["fixed_3h_lcb95"], r["after_top10_mean"], r["total_pnl"]), reverse=True)
    robust = [r for r in ranked if r["trades"] >= 30
              and min(r["chronological_thirds"] or [-1]) > 0
              and min(r["day_means"].values() or [-1]) > 0
              and r["after_top10_mean"] > 0
              and r["after_strongest_3h_block_mean"] > 0
              and r["top1_positive_share"] <= 0.15
              and r["top5_positive_share"] <= 0.50]
    best_summary = robust[0] if robust else ranked[0]
    model = best_summary["model"]
    opps = model_cache[(model["sigma_window"], model["sigma_multiplier"])]
    p = best_summary["params"]
    params = (p["latency_s"], p["participation"], p["min_seconds_to_close"], p["max_seconds_to_close"], p["p_min"], p["edge_min"], p["max_ask"])
    best = evaluate(params, metas, rows_by, opps)
    best["model"] = model
    report = {
        "round": "continuous basis-anchored settlement-average variance development",
        "opened_days": sorted(TARGET_DAYS),
        "unopened_continuous_day": "2026-07-29",
        "market_count": len(metas), "grid_count": len(results), "robust_count": len(robust),
        "best": best,
        "top100_robust": [{k: v for k, v in r.items() if k != "records"} for r in robust[:100]],
        "top100_all": [{k: v for k, v in r.items() if k != "records"} for r in ranked[:100]],
    }
    (OUT / "continuous_spot_basis_dev.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in best.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
