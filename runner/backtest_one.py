from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd

from data.store import cache_path, load_cache_csv
from indicators.ichimoku import IchimokuParams
from signals.ichimoku_a_plus import APlusConfig
from backtest.execution import ExecutionConfig
from backtest.engine import BacktestConfig, run_backtest_one_symbol
from backtest.report import write_report


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="e.g. BTC/USDT (default: BTC symbol from universe)")
    ap.add_argument("--start", default=None, help="ISO datetime (UTC) start")
    ap.add_argument("--end", default=None, help="ISO datetime (UTC) end")
    ap.add_argument("--out", default="data/outputs/backtest_one", help="output dir")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")
    universe = load_yaml(repo_root / "config" / "universe.yaml")

    # resolve symbol
    symbol = args.symbol
    if symbol is None:
        rows = [
            s for s in universe.get("symbols", [])
            if isinstance(s, dict) and s.get("enabled", True) and s.get("is_btc") is True
        ]
        if not rows:
            raise RuntimeError("No BTC symbol in universe.yaml (need is_btc: true) or provide --symbol.")
        symbol = rows[0]["symbol"]

    tf_signal = (settings.get("timeframes") or {}).get("signal", "15m")

    df15 = load_cache_csv(cache_path(repo_root, symbol, tf_signal))
    if df15.empty:
        raise RuntimeError("Cache is empty for this symbol/timeframe. Run scan_once first to fetch data.")

    ich_cfg = settings.get("ichimoku") or {}
    setup_cfg = settings.get("setup_a_plus") or {}
    exe_cfg = settings.get("backtest") or {}
    dedup_cfg = settings.get("dedup") or {}

    ich = IchimokuParams(
        tenkan=int(ich_cfg.get("tenkan", 9)),
        kijun=int(ich_cfg.get("kijun", 26)),
        spanb=int(ich_cfg.get("spanb", 52)),
        displacement=int(ich_cfg.get("displacement", 26)),
    )

    aplus = APlusConfig(
        score_threshold=int(setup_cfg.get("score_threshold", 80)),
        retest_window_bars=int(setup_cfg.get("retest_window_bars", 12)),
        retest_eps=float(setup_cfg.get("retest_eps", 0.0015)),
        kijun_slope_bars=int(setup_cfg.get("kijun_slope_bars", 10)),
        max_kijun_distance_pct=float(setup_cfg.get("max_kijun_distance_pct", 0.008)),
        min_kumo_thickness_pct=float(setup_cfg.get("min_kumo_thickness_pct", 0.0035)),
    )

    exe = ExecutionConfig(
        fee_rate=float((settings.get("costs") or {}).get("fee_rate", exe_cfg.get("fee_rate", 0.001))),
        slippage_rate=float((settings.get("costs") or {}).get("slippage_rate", exe_cfg.get("slippage_rate", 0.0005))),
        risk_per_trade=float(exe_cfg.get("risk_per_trade", 0.01)),
        atr_period=int(exe_cfg.get("atr_period", 14)),
        atr_stop_mult=float(exe_cfg.get("atr_stop_mult", 2.0)),
    )

    bt = BacktestConfig(
        symbol=symbol,
        timeframe_signal=tf_signal,
        start_ts=args.start,
        end_ts=args.end,
        kijun_slope_bars=int(setup_cfg.get("kijun_slope_bars", 10)),
    )

    trades, meta = run_backtest_one_symbol(df15, bt, ich, aplus, exe)

    out_dir = repo_root / args.out
    summary = write_report(out_dir, trades, meta)

    print("[backtest_one] wrote:", str(out_dir))
    print("[backtest_one] summary:", summary)


if __name__ == "__main__":
    main()
