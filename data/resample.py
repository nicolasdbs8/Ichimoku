from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd

from data.fetch_ohlcv import timeframe_to_ms, last_closed_candle_end_utc


@dataclass(frozen=True)
class MissingReport:
    expected: int
    actual: int
    missing: int
    missing_pct: float


def _ensure_utc_sorted(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def detect_missing_candles(df: pd.DataFrame, timeframe: str) -> MissingReport:
    """
    Detect missing candles by checking if ts increments by exactly timeframe.
    We DO NOT fabricate candles; we only report missing.
    """
    df = _ensure_utc_sorted(df)
    if df.empty:
        return MissingReport(expected=0, actual=0, missing=0, missing_pct=0.0)

    tf_ms = timeframe_to_ms(timeframe)
    start = df["ts"].iloc[0]
    end = df["ts"].iloc[-1]
    # Number of expected candles inclusive endpoints on a regular grid:
    span_ms = int((end - start).total_seconds() * 1000)
    expected = (span_ms // tf_ms) + 1
    actual = len(df)
    missing = max(0, expected - actual)
    missing_pct = (missing / expected) if expected > 0 else 0.0
    return MissingReport(expected=expected, actual=actual, missing=missing, missing_pct=missing_pct)


def resample_from_15m(df15: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Build 1h / 4h candles from 15m. Assumes df15 is 15m, UTC, close times aligned.
    Resample uses:
    open=first, high=max, low=min, close=last, volume=sum.
    """
    df15 = _ensure_utc_sorted(df15)
    if df15.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    # set index
    dfi = df15.set_index("ts")

    target_tf = target_tf.lower()
    rule = {"1h": "1h", "4h": "4h"}.get(target_tf)
    if rule is None:
        raise ValueError(f"Unsupported resample target: {target_tf}")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    out = dfi.resample(rule, label="right", closed="right").agg(agg).dropna().reset_index()
    # label="right" means timestamps at candle end boundary, consistent for "last closed"
    out = out.rename(columns={"ts": "ts"})
    out = out[["ts", "open", "high", "low", "close", "volume"]]
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def compute_freshness(df: pd.DataFrame, timeframe: str) -> Dict[str, object]:
    """
    Freshness: compare last ts in df to last closed boundary for timeframe.
    """
    if df.empty:
        return {"status": "empty", "last_ts": None, "expected_last_close": str(last_closed_candle_end_utc(timeframe))}
    df = _ensure_utc_sorted(df)
    last_ts = df["ts"].iloc[-1]
    expected = last_closed_candle_end_utc(timeframe)
    is_fresh = last_ts >= expected
    return {
        "status": "fresh" if is_fresh else "late",
        "last_ts": last_ts.isoformat(),
        "expected_last_close": expected.isoformat(),
    }


def build_multitf(df15: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df1h = resample_from_15m(df15, "1h")
    df4h = resample_from_15m(df15, "4h")
    return df15, df1h, df4h
