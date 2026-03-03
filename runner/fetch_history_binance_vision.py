from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests


BINANCE_VISION_BASE = "https://data.binance.vision"


def month_range(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    endm = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= endm:
        yield cur.year, cur.month
        cur = (cur + pd.offsets.MonthBegin(1)).tz_localize("UTC")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_zip_klines(zip_bytes: bytes) -> pd.DataFrame:
    """
    Binance kline zip contains one CSV with columns:
    open_time, open, high, low, close, volume, close_time, quote_asset_volume,
    number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore

    We keep: ts (UTC), open, high, low, close, volume
    ts = open_time in milliseconds (spot/futures). (Some spot datasets may use microseconds for newer periods; we normalize.)
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # take first csv
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("Zip has no CSV")
        with zf.open(names[0]) as f:
            df = pd.read_csv(
                f,
                header=None,
                usecols=[0, 1, 2, 3, 4, 5],
                names=["open_time", "open", "high", "low", "close", "volume"],
            )

    # open_time can be ms or µs depending on dataset – normalize to ms by magnitude.
    ot = df["open_time"].astype("int64")
    # if values look like microseconds (>= 1e15), convert to ms
    if ot.max() >= 10**15:
        ot = ot // 1000
    df["ts"] = pd.to_datetime(ot, unit="ms", utc=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["ts", "open", "high", "low", "close", "volume"]].dropna()
    return df


def download(url: str, timeout: int = 60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def build_monthly_url(market: str, symbol_noslash: str, timeframe: str, year: int, month: int) -> str:
    # market: "spot" or "futures/um"
    mm = f"{month:02d}"
    if market == "spot":
        # https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2025-01.zip
        return (
            f"{BINANCE_VISION_BASE}/data/spot/monthly/klines/"
            f"{symbol_noslash}/{timeframe}/{symbol_noslash}-{timeframe}-{year}-{mm}.zip"
        )
    if market == "futures/um":
        # https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2025-01.zip
        return (
            f"{BINANCE_VISION_BASE}/data/futures/um/monthly/klines/"
            f"{symbol_noslash}/{timeframe}/{symbol_noslash}-{timeframe}-{year}-{mm}.zip"
        )
    raise ValueError("Unknown market")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch historical klines from Binance Public Data (no API).")
    ap.add_argument("--market", choices=["spot", "futures_um"], default="spot")
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--start", required=True, help="ISO UTC e.g. 2025-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="ISO UTC e.g. 2026-01-01T00:00:00Z")
    ap.add_argument("--out_csv", required=True, help="Output cache csv path")
    args = ap.parse_args()

    start = pd.to_datetime(args.start.replace("Z", "+00:00"), utc=True)
    end = pd.to_datetime(args.end.replace("Z", "+00:00"), utc=True)

    sym = args.symbol.replace("/", "")
    market = "spot" if args.market == "spot" else "futures/um"

    out_path = Path(args.out_csv)
    ensure_dir(out_path.parent)

    frames = []
    ok_months = 0
    miss_months = 0

    for y, m in month_range(start, end):
        url = build_monthly_url(market, sym, args.timeframe, y, m)
        try:
            b = download(url)
            dfm = parse_zip_klines(b)
            frames.append(dfm)
            ok_months += 1
            print(f"[binance_vision] OK {y}-{m:02d} rows={len(dfm)}")
        except requests.HTTPError as e:
            # not all months exist (or lag in publishing) -> treat as missing
            miss_months += 1
            print(f"[binance_vision] MISS {y}-{m:02d} url={url} err={e}")
        except Exception as e:
            raise RuntimeError(f"Failed month {y}-{m:02d}: {e}") from e

    if not frames:
        raise RuntimeError("No data downloaded (all months missing). Check market/symbol/timeframe.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Clip to [start, end]
    df = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

    df.to_csv(out_path, index=False)
    print(
        "[binance_vision] done="
        + str(
            {
                "market": market,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "start": args.start,
                "end": args.end,
                "ok_months": ok_months,
                "missing_months": miss_months,
                "rows": len(df),
                "out_csv": str(out_path),
            }
        )
    )


if __name__ == "__main__":
    main()
