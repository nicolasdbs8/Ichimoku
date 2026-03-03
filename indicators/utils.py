from __future__ import annotations

import pandas as pd


def kumo_top(span_a: pd.Series, span_b: pd.Series) -> pd.Series:
    return pd.concat([span_a, span_b], axis=1).max(axis=1)


def kumo_bottom(span_a: pd.Series, span_b: pd.Series) -> pd.Series:
    return pd.concat([span_a, span_b], axis=1).min(axis=1)


def kijun_slope(kijun: pd.Series, n: int = 10) -> pd.Series:
    return kijun - kijun.shift(n)


def kumo_thickness_pct(span_a: pd.Series, span_b: pd.Series, close: pd.Series) -> pd.Series:
    top = kumo_top(span_a, span_b)
    bot = kumo_bottom(span_a, span_b)
    return (top - bot) / close
