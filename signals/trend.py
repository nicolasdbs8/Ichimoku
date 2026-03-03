from __future__ import annotations

import pandas as pd

from indicators.utils import kumo_top, kijun_slope


def trend_on_1h(df1h: pd.DataFrame, slope_bars: int = 10) -> bool:
    """
    Trend ON if:
      A) close > kumo_top
      OR
      B) tenkan > kijun AND kijun_slope>0
    df1h must contain: close, span_a, span_b, tenkan, kijun
    """
    if df1h.empty:
        return False

    last = df1h.iloc[-1]
    needed = ["close", "span_a", "span_b", "tenkan", "kijun"]
    for k in needed:
        if k not in df1h.columns or pd.isna(last.get(k)):
            return False

    kt = kumo_top(df1h["span_a"], df1h["span_b"]).iloc[-1]
    if pd.isna(kt):
        return False

    if df1h["close"].iloc[-1] > kt:
        return True

    slope = kijun_slope(df1h["kijun"], slope_bars).iloc[-1]
    if pd.isna(slope):
        return False

    return (df1h["tenkan"].iloc[-1] > df1h["kijun"].iloc[-1]) and (slope > 0)
