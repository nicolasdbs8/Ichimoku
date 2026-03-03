from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yaml

from data.fetch_ohlcv import make_exchange, fetch_ohlcv_incremental
from data.store import ensure_cache_dir, cache_path, load_cache_csv, save_cache_csv


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def iso_to_ms(iso: str) -> int:
    # Accept "Z" suffix
    s = iso.replace("Z", "+00:00")
    ts = pd.to_datetime(s, utc=True)
    return int(ts.value // 1_000_000)  # ns -> ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch full OHLCV history into cache (GitHub runner friendly).")
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    ap.add_argument("--timeframe", default="15m", help="e.g. 15m")
    ap.add_argument("--start", required=True, help="ISO UTC start e.g. 2025-01-01T00:00:00Z")
    ap.add_argument("--end", default=None, help="ISO UTC end (optional). If omitted: fetch until latest available.")
    ap.add_argument("--limit", type=int, default=500, help="ccxt limit per call")
    ap.add_argument("--max_batches", type=int, default=5000, help="safety cap on loops")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")

    exchange_name = (settings.get("exchange") or {}).get("name", "kraken")
    enable_rl = (settings.get("exchange") or {}).get("enable_rate_limit", True)

    ensure_cache_dir(repo_root)
    cpath = cache_path(repo_root, args.symbol, args.timeframe)

    df_cached = load_cache_csv(cpath)
    if not df_cached.empty:
        df_cached["ts"] = pd.to_datetime(df_cached["ts"], utc=True)
        df_cached = df_cached.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    start_ms = iso_to_ms(args.start)
    end_ms = iso_to_ms(args.end) if args.end else None

    # Decide where to start: if cache already has data, continue from last candle
    since_ms = start_ms
    if not df_cached.empty:
        last_ts = pd.to_datetime(df_cached["ts"].iloc[-1], utc=True)
        last_ms = int(last_ts.value // 1_000_000) + 1
        since_ms = max(since_ms, last_ms)

    ex = make_exchange(exchange_name, enable_rate_limit=enable_rl)

    total_new = 0
    batches = 0

    while batches < args.max_batches:
        fetched = fetch_ohlcv_incremental(
            ex,
            symbol=args.symbol,
            timeframe=args.timeframe,
            since_ms=since_ms,
            limit=args.limit,
            max_batches=1,     # exactly one batch per loop; we control looping here
            max_retries=8,
        )
        batches += 1

        df_new = fetched.df
        if df_new is None or df_new.empty:
            break

        df_new["ts"] = pd.to_datetime(df_new["ts"], utc=True)
        df_new = df_new.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

        # If end provided, stop once we pass it
        if end_ms is not None:
            df_new = df_new[df_new["ts"] <= pd.to_datetime(args.end, utc=True)].copy()

        if df_new.empty:
            break

        df_cached = pd.concat([df_cached, df_new], ignore_index=True)
        df_cached["ts"] = pd.to_datetime(df_cached["ts"], utc=True)
        df_cached = df_cached.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

        save_cache_csv(cpath, df_cached)

        total_new += len(df_new)

        last_ts = pd.to_datetime(df_cached["ts"].iloc[-1], utc=True)
        since_ms = int(last_ts.value // 1_000_000) + 1

        # stop condition if end reached
        if end_ms is not None and since_ms >= end_ms:
            break

        # progress log
        print(
            f"[fetch_history] batch={batches} total_rows={len(df_cached)} "
            f"last_ts={last_ts.isoformat()} new_rows={len(df_new)}"
        )

    print(
        "[fetch_history] done=" + str(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "start": args.start,
                "end": args.end,
                "batches": batches,
                "new_rows": total_new,
                "cache_rows": len(df_cached),
                "cache_file": str(cpath),
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    )


if __name__ == "__main__":
    main()
