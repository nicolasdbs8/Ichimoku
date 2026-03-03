from __future__ import annotations

import pandas as pd


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR (Wilder) using True Range and Wilder's smoothing (EMA alpha=1/period).
    Requires columns: high, low, close
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for ATR: {missing}")

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing: EMA with alpha = 1/period, adjust=False
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr
