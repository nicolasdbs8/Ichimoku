from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import yaml
import pandas as pd

from data.fetch_ohlcv import make_exchange, fetch_ohlcv_incremental
from data.store import ensure_cache_dir, cache_path, load_cache_csv, save_cache_csv
from data.resample import detect_missing_candles, build_multitf, compute_freshness


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected mapping at root")
    return data


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")
    universe = load_yaml(repo_root / "config" / "universe.yaml")

    exchange_name = (settings.get("exchange") or {}).get("name", "kraken")
    enable_rl = (settings.get("exchange") or {}).get("enable_rate_limit", True)
    tf_signal = (settings.get("timeframes") or {}).get("signal", "15m")

    # pick BTC symbol from universe (first item where is_btc == true)
    symbols = universe.get("symbols", [])
    btc_rows = [s for s in symbols if isinstance(s, dict) and s.get("is_btc") is True and s.get("enabled", True)]
    if not btc_rows:
        raise RuntimeError("No BTC symbol found in config/universe.yaml (need is_btc: true).")
    btc_symbol = btc_rows[0]["symbol"]

    ensure_cache_dir(repo_root)
    cpath = cache_path(repo_root, btc_symbol, tf_signal)

    # Load cache -> compute since
    cached = load_cache_csv(cpath)
    since_ms = None
    if not cached.empty:
        # fetch from last_ts + 1ms
        last_ts = pd.to_datetime(cached["ts"].iloc[-1], utc=True)
        since_ms = int(last_ts.value // 1_000_000) + 1  # ns -> ms

    ex = make_exchange(exchange_name, enable_rate_limit=enable_rl)

    fetched = fetch_ohlcv_incremental(
        ex,
        symbol=btc_symbol,
        timeframe=tf_signal,
        since_ms=since_ms,
        limit=500,
        max_batches=5,
        max_retries=5,
    )

    # Merge cache + fetched
    df15 = pd.concat([cached, fetched.df], ignore_index=True)
    df15["ts"] = pd.to_datetime(df15["ts"], utc=True)
    df15 = df15.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Save updated cache (note: in GitHub Actions filesystem is ephemeral, but OK for now)
    save_cache_csv(cpath, df15)

    # Missing candles report on 15m
    miss15 = detect_missing_candles(df15, tf_signal)

    # Build 1h and 4h from 15m
    df15, df1h, df4h = build_multitf(df15)

    health = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "exchange": exchange_name,
        "symbol": btc_symbol,
        "timeframes": {"signal": tf_signal, "trend": "1h", "regime": "4h"},
        "fetch": {
            "since_ms": since_ms,
            "fetched_rows": fetched.fetched_rows,
            "cache_rows_after": len(df15),
        },
        "missing_15m": {
            "expected": miss15.expected,
            "actual": miss15.actual,
            "missing": miss15.missing,
            "missing_pct": round(miss15.missing_pct, 6),
        },
        "freshness": {
            "15m": compute_freshness(df15, "15m"),
            "1h": compute_freshness(df1h, "1h"),
            "4h": compute_freshness(df4h, "4h"),
        },
        "last_rows": {
            "15m": df15.tail(1).to_dict(orient="records"),
            "1h": df1h.tail(1).to_dict(orient="records"),
            "4h": df4h.tail(1).to_dict(orient="records"),
        },
        "notes": [
            "Phase2: fetched BTC 15m, resampled 1h/4h, reported missing + freshness.",
            "Next: implement Ichimoku indicators on these aligned frames.",
        ],
    }

    print("[scan_once] data_health=" + json.dumps(health, ensure_ascii=False))


if __name__ == "__main__":
    main()
