from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class IchimokuParams:
    tenkan: int = 9
    kijun: int = 26
    spanb: int = 52
    displacement: int = 26


def _midpoint_hh_ll(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    hh = high.rolling(window=window, min_periods=window).max()
    ll = low.rolling(window=window, min_periods=window).min()
    return (hh + ll) / 2.0


def ichimoku(df: pd.DataFrame, p: IchimokuParams = IchimokuParams()) -> pd.DataFrame:
    """
    Input df must contain columns: ts, open, high, low, close, volume
    Output columns appended:
      tenkan, kijun, span_a, span_b, chikou
    Shifts:
      span_a/span_b shifted +displacement
      chikou shifted -displacement
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for ichimoku: {missing}")

    out = df.copy()

    tenkan = _midpoint_hh_ll(out["high"], out["low"], p.tenkan)
    kijun = _midpoint_hh_ll(out["high"], out["low"], p.kijun)

    span_a = ((tenkan + kijun) / 2.0).shift(p.displacement)
    span_b = _midpoint_hh_ll(out["high"], out["low"], p.spanb).shift(p.displacement)

    chikou = out["close"].shift(-p.displacement)

    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["span_a"] = span_a
    out["span_b"] = span_b
    out["chikou"] = chikou
    return out
