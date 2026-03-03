from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

import ccxt  # type: ignore
import pandas as pd


@dataclass(frozen=True)
class FetchResult:
    df: pd.DataFrame
    fetched_rows: int
    used_since_ms: Optional[int]


def _to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    # ccxt: [ms, open, high, low, close, volume]
    if not ohlcv:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(ohlcv, columns=["ms", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df = df.drop(columns=["ms"])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return df


def make_exchange(exchange_name: str, enable_rate_limit: bool = True):
    ex_class = getattr(ccxt, exchange_name)
    ex = ex_class({"enableRateLimit": enable_rate_limit})
    return ex


def fetch_ohlcv_incremental(
    ex,
    symbol: str,
    timeframe: str,
    since_ms: Optional[int],
    limit: int = 500,
    max_batches: int = 10,
    max_retries: int = 5,
) -> FetchResult:
    """
    Fetches up to max_batches*limit candles incrementally, starting at since_ms.
    - Respects rate limits (ccxt enableRateLimit)
    - Retries with exponential backoff
    - Returns UTC timestamps
    """
    all_rows: List[List[float]] = []
    used_since = since_ms

    for _ in range(max_batches):
        attempt = 0
        while True:
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=used_since, limit=limit)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"fetch_ohlcv failed for {symbol} {timeframe}: {e}") from e
                sleep_s = min(30, 2 ** attempt)
                time.sleep(sleep_s)

        if not batch:
            break

        # Append and advance since_ms to last_ts+1 to avoid duplicates
        all_rows.extend(batch)
        last_ms = int(batch[-1][0])
        used_since = last_ms + 1

        # If batch smaller than limit, likely reached end
        if len(batch) < limit:
            break

    df = _to_df(all_rows)
    return FetchResult(df=df, fetched_rows=len(df), used_since_ms=since_ms)


def timeframe_to_ms(timeframe: str) -> int:
    # Minimal parser for "15m", "1h", "4h"
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60_000
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def last_closed_candle_end_utc(timeframe: str, now_utc: Optional[datetime] = None) -> pd.Timestamp:
    """
    Returns the timestamp of the last fully closed candle boundary (end time).
    Example: for 15m, if now is 12:37, last close boundary is 12:30.
    """
    now = now_utc or datetime.now(timezone.utc)
    ms = int(now.timestamp() * 1000)
    tf_ms = timeframe_to_ms(timeframe)
    last_boundary_ms = (ms // tf_ms) * tf_ms
    return pd.to_datetime(last_boundary_ms, unit="ms", utc=True)
