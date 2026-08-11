"""Minimal deterministic distribution helpers used by EDGE research.

Only the operations required by the research scripts are implemented. The
Student t quantiles below are one-sided 95% critical values from the standard
Student t distribution table; interpolation is deliberately avoided.
"""
from __future__ import annotations

import math


class norm:
    @staticmethod
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


_T95 = {
    1: 6.313751515,
    2: 2.919985580,
    3: 2.353363435,
    4: 2.131846786,
    5: 2.015048373,
    6: 1.943180281,
    7: 1.894578605,
    8: 1.859548038,
    9: 1.833112933,
    10: 1.812461123,
    11: 1.795884819,
    12: 1.782287556,
    13: 1.770933396,
    14: 1.761310136,
    15: 1.753050356,
    16: 1.745883676,
    17: 1.739606726,
    18: 1.734063607,
    19: 1.729132812,
    20: 1.724718243,
    21: 1.720742903,
    22: 1.717144374,
    23: 1.713871528,
    24: 1.710882079,
    25: 1.708140761,
    26: 1.705617920,
    27: 1.703288446,
    28: 1.701130934,
    29: 1.699127027,
    30: 1.697260887,
}


class t:
    @staticmethod
    def ppf(q: float, df: int) -> float:
        q = float(q)
        df = int(df)
        if abs(q - 0.95) > 1e-12:
            raise NotImplementedError("minimal EDGE shim implements only q=0.95")
        if df <= 0:
            raise ValueError("degrees of freedom must be positive")
        if df in _T95:
            return _T95[df]
        # First-order Cornish-Fisher correction to the one-sided normal 95%
        # quantile. Used only if later research has more than 31 clusters.
        z = 1.6448536269514722
        return z + (z**3 + z) / (4.0 * df) + (5*z**5 + 16*z**3 + 3*z) / (96.0 * df * df)
