"""Tiny subset of NumPy used by the isolated EDGE research scripts.

This avoids network package installation inside the reproducibility workflow.
It is intentionally minimal and deterministic, not a general NumPy replacement.
"""
from __future__ import annotations
import math


class Array(list):
    def mean(self):
        return mean(self)

    def std(self, ddof=0):
        return std(self, ddof=ddof)

    def sum(self):
        return float(sum(float(x) for x in self))

    def tolist(self):
        return list(self)


def array(values, dtype=None):
    return Array(float(x) for x in values)


def asarray(values, dtype=None):
    return array(values, dtype=dtype)


def mean(values):
    vals = [float(x) for x in values]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values, ddof=0):
    vals = [float(x) for x in values]
    n = len(vals)
    if n <= ddof:
        return float("nan")
    mu = sum(vals) / n
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (n - ddof))


def diff(values):
    vals = [float(x) for x in values]
    return Array(vals[i] - vals[i - 1] for i in range(1, len(vals)))


def array_split(values, sections):
    vals = list(values)
    n = len(vals)
    q, r = divmod(n, sections)
    out = []
    start = 0
    for i in range(sections):
        size = q + (1 if i < r else 0)
        out.append(Array(vals[start:start + size]))
        start += size
    return out
