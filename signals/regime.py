from __future__ import annotations

import pandas as pd

from indicators.utils import kumo_top, kijun_slope


def btc_regime_on(df4h: pd.DataFrame, slope_bars: int = 10) -> bool:
    """
    Regime ON if:
      close > kumo_top AND kijun[t]-kijun[t-slope_bars] > 0
    df4h must already contain: close, span_a, span_b, kijun
    """
    if df4h.empty:
        return False

    last = df4h.iloc[-1]
    if pd.isna(last.get("span_a")) or pd.isna(last.get("span_b")) or pd.isna(last.get("kijun")):
        return False

    kt = kumo_top(df4h["span_a"], df4h["span_b"])
    slope = kijun_slope(df4h["kijun"], slope_bars)

    if kt.isna().iloc[-1] or slope.isna().iloc[-1]:
        return False

    return (df4h["close"].iloc[-1] > kt.iloc[-1]) and (slope.iloc[-1] > 0)
