#!/usr/bin/env python3
"""EDGE exploratory round 1: exact-settlement-feed running-average model.

This program deliberately opens only the first 11 chronological 3-hour captures.
Captures 12-15 remain unopened for a later frozen holdout. Capture 16 was used
previously for schema inspection and is excluded from the eventual holdout.
"""
from __future__ import annotations

import bisect
import gzip
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import norm, t as student_t

BTC_DIR = Path(os.environ.get("BTPRED_ROOT", "/tmp/btpred-data")) / "kalshi_15m_live"
PERP_DIR = Path(os.environ.get("BTPRED_ROOT", "/tmp/btpred-data")) / "kalshi_perp"
OUT_DIR = Path("edge_outputs")
OUT_DIR.mkdir(exist_ok=True)

CAPTURE_RE = re.compile(r"-(\d{8}T\d{6}Z)-")
DEV_CAPTURE_COUNT = 11
TRAIN_CAPTURE_COUNT = 7
MAX_CONTRACTS = 100.0
DEPTH_HAIRCUT = 0.50
FEE_RATE = 0.07


def parse_iso_ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def capture_dt(path: Path) -> datetime:
    m = CAPTURE_RE.search(path.name)
    if not m:
        raise ValueError(path)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def compact_asks(book: dict, side: str, needed_raw: float = 500.0) -> tuple[tuple[float, float], ...]:
    # Kalshi orderbook contains YES bids and NO bids. Buying one outcome crosses
    # the complementary bid: YES ask = 1 - highest NO bid, and vice versa.
    source = book.get("no_dollars" if side == "yes" else "yes_dollars") or []
    levels = sorted(((1.0 - float(p), float(q)) for p, q in source), key=lambda x: x[0])
    out: list[tuple[float, float]] = []
    cum = 0.0
    for p, q in levels:
        if q <= 0:
            continue
        p = max(0.0, min(1.0, round(p, 6)))
        out.append((p, q))
        cum += q
        if cum >= needed_raw:
            break
    return tuple(out)


@dataclass(frozen=True)
class KSnap:
    recv_ns: int
    ticker: str
    capture: str
    open_ms: int
    close_ms: int
    strike: float
    yes_asks: tuple[tuple[float, float], ...]
    no_asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class BRTI:
    recv_ns: int
    ref_sec: int
    brti: float
    perp_mid: float
    settlement_mark: float


@dataclass
class Market:
    ticker: str
    capture: str
    open_ms: int
    close_ms: int
    strike: float
    snapshots: list[KSnap]
    brti: list[BRTI]
    true_end_avg: float | None = None
    outcome_yes: bool | None = None


@dataclass(frozen=True)
class Opportunity:
    ticker: str
    capture: str
    signal_ns: int
    remaining: int
    side: str
    p_side: float
    current_avg_ask: float
    current_worst_ask: float
    outcome_yes: bool
    exec_by_latency: dict[int, tuple[float, float, float]]  # latency -> qty,cost,fees


def read_perp(path: Path) -> list[BRTI]:
    rows: list[BRTI] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "snapshot":
                continue
            d = o.get("derived") or {}
            if d.get("reference_ts_ms") is None or d.get("brti_usd") is None:
                continue
            rows.append(BRTI(
                recv_ns=int(o["received_ns"]),
                ref_sec=int(d["reference_ts_ms"]) // 1000,
                brti=float(d["brti_usd"]),
                perp_mid=float(d.get("perp_mid_usd") or d["brti_usd"]),
                settlement_mark=float(d.get("settlement_mark_usd") or d["brti_usd"]),
            ))
    rows.sort(key=lambda x: x.recv_ns)
    return rows


def read_kalshi(path: Path, brti: list[BRTI]) -> list[Market]:
    by_ticker: dict[str, list[KSnap]] = defaultdict(list)
    meta: dict[str, tuple[int, int, float]] = {}
    cap = path.name
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != "snapshot":
                continue
            m = o.get("market") or {}
            ticker = o.get("ticker") or m.get("ticker")
            if not ticker or not m.get("open_time") or not m.get("close_time"):
                continue
            open_ms = parse_iso_ms(m["open_time"])
            close_ms = parse_iso_ms(m["close_time"])
            strike = float(m["floor_strike"])
            meta[ticker] = (open_ms, close_ms, strike)
            ob = o.get("orderbook") or {}
            by_ticker[ticker].append(KSnap(
                recv_ns=int(o["received_ns"]), ticker=ticker, capture=cap,
                open_ms=open_ms, close_ms=close_ms, strike=strike,
                yes_asks=compact_asks(ob, "yes"),
                no_asks=compact_asks(ob, "no"),
            ))
    markets: list[Market] = []
    for ticker, snaps in by_ticker.items():
        snaps.sort(key=lambda x: x.recv_ns)
        open_ms, close_ms, strike = meta[ticker]
        mk = Market(ticker, cap, open_ms, close_ms, strike, snaps, brti)
        assign_truth(mk)
        if mk.true_end_avg is not None:
            markets.append(mk)
    return markets


def first_available_brti_by_ref(rows: list[BRTI]) -> dict[int, BRTI]:
    out: dict[int, BRTI] = {}
    for r in rows:
        out.setdefault(r.ref_sec, r)
    return out


def assign_truth(m: Market) -> None:
    by_ref = first_available_brti_by_ref(m.brti)
    end = m.close_ms // 1000
    vals = [by_ref[s].brti for s in range(end - 60, end) if s in by_ref]
    if len(vals) < 58:
        return
    # Require no internal gap except at most two seconds. Missing seconds are
    # linearly interpolated only for truth labeling, never for the signal.
    full: list[float] = []
    known_secs = sorted(s for s in range(end - 60, end) if s in by_ref)
    for s in range(end - 60, end):
        if s in by_ref:
            full.append(by_ref[s].brti)
        else:
            lo = max((k for k in known_secs if k < s), default=None)
            hi = min((k for k in known_secs if k > s), default=None)
            if lo is None or hi is None:
                return
            w = (s - lo) / (hi - lo)
            full.append(by_ref[lo].brti * (1 - w) + by_ref[hi].brti * w)
    m.true_end_avg = float(np.mean(full))
    m.outcome_yes = bool(m.true_end_avg >= m.strike - 1e-12)


def pair_files() -> list[tuple[Path, Path]]:
    kfiles = sorted(BTC_DIR.glob("KXBTC15M-*.jsonl.gz"), key=capture_dt)
    pfiles = sorted(PERP_DIR.glob("KXBTCPERP-*.jsonl.gz"), key=capture_dt)
    pairs: list[tuple[Path, Path]] = []
    for k in kfiles:
        p = min(pfiles, key=lambda x: abs((capture_dt(x) - capture_dt(k)).total_seconds()))
        delta = abs((capture_dt(p) - capture_dt(k)).total_seconds())
        if delta <= 120:
            pairs.append((k, p))
    return pairs


def brti_state_at(m: Market, recv_ns: int) -> tuple[int, float, float, float] | None:
    # Returns remaining samples, forecast final average, local 10s sigma, and
    # latest BRTI using only reference observations received by recv_ns.
    rows = m.brti
    recvs = [x.recv_ns for x in rows]
    i = bisect.bisect_right(recvs, recv_ns) - 1
    if i < 0:
        return None
    known: dict[int, float] = {}
    for r in rows[: i + 1]:
        known[r.ref_sec] = r.brti
    end = m.close_ms // 1000
    start = end - 60
    observed_secs = [s for s in range(start, end) if s in known]
    if not observed_secs:
        return None
    # The settlement interval must be known contiguously from its first second.
    n = 0
    vals: list[float] = []
    for s in range(start, end):
        if s not in known:
            break
        vals.append(known[s])
        n += 1
    if n == 0:
        return None
    remaining = 60 - n
    current = vals[-1]
    forecast = (sum(vals) + remaining * current) / 60.0
    diffs = np.diff(vals[-11:]) if len(vals) >= 3 else np.array([0.0])
    sigma = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0
    return remaining, forecast, sigma, current


def state_examples(markets: list[Market], max_remaining: int = 30) -> list[dict]:
    rows: list[dict] = []
    for m in markets:
        assert m.true_end_avg is not None and m.outcome_yes is not None
        seen_rem: set[int] = set()
        for s in m.snapshots:
            st = brti_state_at(m, s.recv_ns)
            if st is None:
                continue
            rem, fc, sigma, cur = st
            if rem < 0 or rem > max_remaining or rem in seen_rem:
                continue
            seen_rem.add(rem)
            rows.append({
                "ticker": m.ticker, "capture": m.capture, "recv_ns": s.recv_ns,
                "remaining": rem, "forecast": fc, "sigma": sigma, "current": cur,
                "strike": m.strike, "error": m.true_end_avg - fc,
                "outcome_yes": m.outcome_yes,
            })
    return rows


def fit_error_model(rows: list[dict]) -> dict[int, tuple[float, float, int]]:
    by: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by[int(r["remaining"])].append(float(r["error"]))
    all_e = [float(r["error"]) for r in rows]
    global_mu = float(np.mean(all_e))
    global_sd = max(float(np.std(all_e, ddof=1)), 1e-6)
    model: dict[int, tuple[float, float, int]] = {}
    for rem in range(0, 31):
        vals: list[float] = []
        for d in range(0, 4):
            for rr in {rem - d, rem + d}:
                if rr in by:
                    vals.extend(by[rr])
            if len(vals) >= 40:
                break
        n = len(vals)
        if n >= 8:
            mu = float(np.mean(vals))
            sd = max(float(np.std(vals, ddof=1)), 0.02)
            # Conservative variance inflation for model uncertainty and
            # second-level interpolation/capture timing.
            sd *= 1.35
        else:
            mu, sd = global_mu, global_sd * 1.5
        model[rem] = (mu, sd, n)
    return model


def walk(levels: tuple[tuple[float, float], ...], limit_price: float, max_qty: float = MAX_CONTRACTS) -> tuple[float, float, float]:
    qty = 0.0
    cost = 0.0
    fee = 0.0
    for p, displayed in levels:
        if p > limit_price + 1e-12 or qty >= max_qty - 1e-12:
            break
        take = min(max_qty - qty, displayed * DEPTH_HAIRCUT)
        if take <= 0:
            continue
        qty += take
        cost += take * p
        # Conservative: fee rounded up separately at every price level.
        raw = FEE_RATE * take * p * (1.0 - p)
        fee += math.ceil(raw * 100.0 - 1e-12) / 100.0
    return qty, cost, fee


def avg_ask(levels: tuple[tuple[float, float], ...], qty_target: float = MAX_CONTRACTS) -> tuple[float, float] | None:
    qty = 0.0
    cost = 0.0
    worst = 0.0
    for p, displayed in levels:
        take = min(qty_target - qty, displayed * DEPTH_HAIRCUT)
        if take <= 0:
            continue
        qty += take
        cost += take * p
        worst = p
        if qty >= qty_target - 1e-9:
            return cost / qty, worst
    if qty >= 10.0:
        return cost / qty, worst
    return None


def build_opportunities(markets: list[Market], model: dict[int, tuple[float, float, int]], latencies: list[int]) -> list[Opportunity]:
    out: list[Opportunity] = []
    for m in markets:
        assert m.outcome_yes is not None
        snap_ns = [s.recv_ns for s in m.snapshots]
        seen_rem: set[int] = set()
        for idx, s in enumerate(m.snapshots):
            st = brti_state_at(m, s.recv_ns)
            if st is None:
                continue
            rem, fc, _sigma, _cur = st
            if rem < 0 or rem > 30 or rem in seen_rem:
                continue
            seen_rem.add(rem)
            mu, sd, _n = model[rem]
            z = (fc + mu - m.strike) / sd
            p_yes = float(norm.cdf(z))
            if p_yes >= 0.5:
                side = "yes"; p_side = p_yes; levels = s.yes_asks
            else:
                side = "no"; p_side = 1.0 - p_yes; levels = s.no_asks
            aa = avg_ask(levels)
            if aa is None:
                continue
            current_avg, current_worst = aa
            execs: dict[int, tuple[float, float, float]] = {}
            for latency_ms in latencies:
                j = bisect.bisect_left(snap_ns, s.recv_ns + latency_ms * 1_000_000)
                if j >= len(m.snapshots):
                    execs[latency_ms] = (0.0, 0.0, 0.0)
                    continue
                ex = m.snapshots[j]
                exlevels = ex.yes_asks if side == "yes" else ex.no_asks
                # Fixed IOC limit: current worst price plus 2 cents, never above
                # the probability less a one-cent safety margin.
                limit = min(0.999, current_worst + 0.02, max(0.0, p_side - 0.01))
                execs[latency_ms] = walk(exlevels, limit)
            out.append(Opportunity(
                ticker=m.ticker, capture=m.capture, signal_ns=s.recv_ns,
                remaining=rem, side=side, p_side=p_side,
                current_avg_ask=current_avg, current_worst_ask=current_worst,
                outcome_yes=bool(m.outcome_yes), exec_by_latency=execs,
            ))
    return out


def pnl_for_opp(o: Opportunity, latency: int) -> tuple[float, float, float, bool]:
    qty, cost, fee = o.exec_by_latency[latency]
    if qty <= 0:
        return 0.0, 0.0, fee, False
    win = (o.side == "yes" and o.outcome_yes) or (o.side == "no" and not o.outcome_yes)
    pnl = (qty if win else 0.0) - cost - fee
    return pnl, qty, fee, win


def evaluate(params: tuple, opps: list[Opportunity], market_keys: list[tuple[str, str]]) -> dict:
    latency, max_rem, pmin, edge_min, max_ask = params
    by_market: dict[tuple[str, str], list[Opportunity]] = defaultdict(list)
    for o in opps:
        by_market[(o.capture, o.ticker)].append(o)
    records: list[dict] = []
    for key in market_keys:
        candidates = sorted(by_market.get(key, []), key=lambda x: x.signal_ns)
        chosen = None
        for o in candidates:
            if o.remaining > max_rem or o.p_side < pmin or o.current_avg_ask > max_ask:
                continue
            # Conservative signal-side fee estimate for 100 contracts at current avg.
            fee_pc = FEE_RATE * o.current_avg_ask * (1.0 - o.current_avg_ask)
            if o.p_side - o.current_avg_ask - fee_pc < edge_min:
                continue
            chosen = o
            break
        if chosen is None:
            records.append({"capture": key[0], "ticker": key[1], "pnl": 0.0, "qty": 0.0, "win": None})
            continue
        pnl, qty, fee, win = pnl_for_opp(chosen, latency)
        records.append({
            "capture": key[0], "ticker": key[1], "pnl": pnl, "qty": qty,
            "win": win, "side": chosen.side, "remaining": chosen.remaining,
            "p_side": chosen.p_side, "ask": chosen.current_avg_ask, "fee": fee,
        })
    pnls = np.array([r["pnl"] for r in records], dtype=float)
    n = len(records)
    mean_market = float(pnls.mean()) if n else -1e9
    annual = mean_market * 96.0 * 365.0
    cap_groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        cap_groups[r["capture"]].append(float(r["pnl"]))
    capture_means = np.array([np.mean(v) for _, v in sorted(cap_groups.items())], dtype=float)
    if len(capture_means) >= 2:
        se = float(capture_means.std(ddof=1) / math.sqrt(len(capture_means)))
        crit = float(student_t.ppf(0.95, df=len(capture_means) - 1))
        lcb_mean = float(capture_means.mean() - crit * se)
    else:
        lcb_mean = -1e9
    annual_lcb = lcb_mean * 96.0 * 365.0
    # Chronological thirds include no-trade markets.
    thirds = [float(x.mean()) for x in np.array_split(pnls, 3) if len(x)]
    trade_pnls = sorted([float(r["pnl"]) for r in records if r["qty"] > 0], reverse=True)
    remove_n = max(1, math.ceil(0.10 * len(trade_pnls))) if trade_pnls else 0
    removed = trade_pnls[remove_n:] if remove_n else []
    after_top10_mean = (sum(removed) / n) if n else -1e9
    capture_totals = {k: float(sum(v)) for k, v in cap_groups.items()}
    if capture_totals:
        strongest = max(capture_totals, key=capture_totals.get)
        after_strong = sum(v for k, v in capture_totals.items() if k != strongest) / max(1, n - len(cap_groups[strongest]))
    else:
        after_strong = -1e9
    return {
        "params": {"latency_ms": latency, "max_remaining": max_rem, "p_min": pmin, "edge_min": edge_min, "max_ask": max_ask},
        "markets": n, "trades": int(sum(r["qty"] > 0 for r in records)),
        "wins": int(sum(r.get("win") is True for r in records)),
        "losses": int(sum(r.get("win") is False for r in records)),
        "partial_or_failed": int(sum(0 < r["qty"] < MAX_CONTRACTS for r in records)),
        "total_pnl": float(pnls.sum()), "mean_per_market": mean_market,
        "annualized": annual, "annual_lcb95": annual_lcb,
        "capture_means": capture_means.tolist(), "third_means": thirds,
        "after_top10_mean": after_top10_mean, "after_strongest_capture_mean": after_strong,
        "records": records,
    }


def main() -> None:
    pairs = pair_files()
    inventory = [{"kalshi": k.name, "perp": p.name, "dt": capture_dt(k).isoformat()} for k, p in pairs]
    if len(pairs) < DEV_CAPTURE_COUNT + 4:
        raise RuntimeError(f"Need at least 15 paired captures, found {len(pairs)}")
    dev_pairs = pairs[:DEV_CAPTURE_COUNT]
    markets_by_capture: list[list[Market]] = []
    for k, p in dev_pairs:
        brti = read_perp(p)
        mk = read_kalshi(k, brti)
        markets_by_capture.append(mk)
        print(k.name, len(mk), "complete markets")
    train_markets = [m for group in markets_by_capture[:TRAIN_CAPTURE_COUNT] for m in group]
    val_markets = [m for group in markets_by_capture[TRAIN_CAPTURE_COUNT:] for m in group]
    train_rows = state_examples(train_markets)
    model = fit_error_model(train_rows)
    latencies = [0, 250, 500, 1000, 1500, 2000]
    val_opps = build_opportunities(val_markets, model, latencies)
    market_keys = [(m.capture, m.ticker) for m in val_markets]

    grid = []
    for latency in latencies:
        for max_rem in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
            for pmin in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
                for edge in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
                    for max_ask in [0.80, 0.90, 0.95, 0.97, 0.99]:
                        grid.append((latency, max_rem, pmin, edge, max_ask))
    results = []
    for i, params in enumerate(grid):
        r = evaluate(params, val_opps, market_keys)
        # Drop per-market records for bulk ranking to keep output compact.
        compact = {k: v for k, v in r.items() if k != "records"}
        results.append((r, compact))
    ranked = sorted(results, key=lambda x: (x[0]["annual_lcb95"], x[0]["after_top10_mean"], x[0]["total_pnl"]), reverse=True)
    robust = [x for x in ranked if x[0]["trades"] >= 8 and min(x[0]["third_means"] or [-1]) > 0 and x[0]["after_top10_mean"] > 0 and x[0]["after_strongest_capture_mean"] > 0]
    best_full = (robust[0] if robust else ranked[0])[0]

    model_json = {str(k): {"mu": v[0], "sd": v[1], "n": v[2]} for k, v in model.items()}
    report = {
        "round": "BRTI running-average Gaussian residual v1",
        "inventory": inventory,
        "opened_captures": [k.name for k, _ in dev_pairs],
        "reserved_unopened_holdout": [k.name for k, _ in pairs[11:15]],
        "excluded_schema_inspection_capture": pairs[15][0].name,
        "train_markets": len(train_markets), "validation_markets": len(val_markets),
        "train_state_rows": len(train_rows), "validation_opportunities": len(val_opps),
        "error_model": model_json,
        "best": best_full,
        "top50": [x[1] for x in ranked[:50]],
        "top50_robust": [x[1] for x in robust[:50]],
    }
    (OUT_DIR / "explore_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    md = [
        "# EDGE exploratory round 1 — BRTI running-average residual model",
        "",
        f"Opened development captures: {len(dev_pairs)}; training markets: {len(train_markets)}; validation markets: {len(val_markets)}.",
        f"Reserved unopened captures: {', '.join(report['reserved_unopened_holdout'])}.",
        "",
        "## Best development/validation configuration",
        "```json", json.dumps({k:v for k,v in best_full.items() if k != 'records'}, indent=2), "```",
        "",
        "This is exploratory only. No reserved capture was opened and this output is not a winning claim.",
    ]
    (OUT_DIR / "explore_v1.md").write_text("\n".join(md))
    print(json.dumps({k:v for k,v in best_full.items() if k != 'records'}, indent=2))


if __name__ == "__main__":
    main()
