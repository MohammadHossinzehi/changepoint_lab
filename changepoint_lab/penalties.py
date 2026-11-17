"""Penalty selection.

All costs are on the 2 * negative log likelihood scale, so the classic
information criteria translate directly into a per changepoint penalty:
adding a changepoint adds ``p`` segment parameters plus the location itself.
This matches the convention of the R ``changepoint`` package.
"""

from __future__ import annotations

import math


def resolve_penalty(penalty, n: int, n_params: int) -> float:
    """Return a numeric per changepoint penalty.

    penalty: a number, or one of
      "bic"  -> (p + 1) * log(n)   (Schwarz; consistent, the default)
      "aic"  -> 2 * (p + 1)         (permissive, tends to over segment)
      "hq"   -> 2 * (p + 1) * log(log(n))   (Hannan Quinn, in between)
    """
    if isinstance(penalty, (int, float)):
        if penalty < 0 or not math.isfinite(penalty):
            raise ValueError("numeric penalty must be finite and >= 0")
        return float(penalty)
    k = n_params + 1
    name = str(penalty).lower()
    if name == "bic":
        return k * math.log(max(n, 2))
    if name == "aic":
        return 2.0 * k
    if name == "hq":
        return 2.0 * k * math.log(max(math.log(max(n, 3)), 1.0))
    raise ValueError(f"unknown penalty {penalty!r}; use a number, 'bic', 'aic' or 'hq'")
