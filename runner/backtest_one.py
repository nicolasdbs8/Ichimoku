from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import yaml

from data.store import cache_path, load_cache_csv
from indicators.ichimoku import ichimoku, IchimokuParams
from indicators.atr import atr_wilder
from data.resample import resample_from_15m  # we will use for 15m->1h/4h

from signals.regime import btc_regime_on
from signals.trend import trend_on_1h
from signals.ichimoku_a_plus import APlusConfig

# --- Locate run_backtest_one in your backtest package without guessing the file name ---
run_backtest_one = None

_IMPORT_CANDIDATES = [
    "backtest.engine_one",
    "backtest.engine",
    "backtest.one",
    "backtest.one_symbol",
    "backtest.backtest_one",
    "backtest.runner",
    "backtest.core",
]

for mod in _IMPORT_CANDIDATES:
    try:
        m = __import__(mod, fromlist=["run_backtest_one"])
        run_backtest_one = getattr(m, "run_backtest_one", None)
        if callable(run_backtest_one):
            break
    except Exception:
        continue

if not callable(run_backtest_one):
    raise ImportError(
        "Could not import a callable run_backtest_one from backtest/. "
        "Fix by exposing run_backtest_one in one of these modules: "
        + ", ".join(_IMPORT_CANDIDATES)
        + ". Tip: search in the repo for `def run_backtest_one` and add it to backtest/__init__.py or adjust the import."
    )
# --- end ---


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return df


def _parse_iso(iso: Optional[str]) -> Optional[pd.Timestamp]:
    if not iso:
        return None
    return pd.to_datetime(iso.replace("Z", "+00:00"), utc=True)


def resample_from_1h(df1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV from 1h base to higher TF (e.g. 4h) conservatively:
    - require complete buckets (e.g. 4x 1h for 4h), otherwise drop.
    """
    dfi = df1h.copy()
    dfi["ts"] = pd.to_datetime(dfi["ts"], utc=True)
    dfi = dfi.set_index("ts").sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    out = dfi.resample(rule, label="right", closed="right").agg(agg)
    # completeness check via count of "close"
    cnt = dfi["close"].resample(rule, label="right", closed="right").count()

    # Expected bars per bucket for common rules
    expected = None
    if rule == "4h":
        expected = 4
    elif rule == "2h":
        expected = 2
    elif rule == "1d":
        expected = 24

    if expected is not None:
        out = out[cnt >= expected]

    out = out.dropna().reset_index()
    out.rename(columns={"ts": "ts"}, inplace=True)
    return out


def build_multitf_for_backtest(df_signal: pd.DataFrame, tf_signal: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_signal, df_trend, df_regime) where:
      - tf_signal == 15m  -> trend=1h, regime=4h (built from 15m)
      - tf_signal == 1h   -> trend=4h, regime=4h (built from 1h)
    """
    if tf_signal == "15m":
        df15 = df_signal
        df1h = resample_from_15m(df15, "1h")
        df4h = resample_from_15m(df15, "4h")
        return df15, df1h, df4h

    if tf_signal == "1h":
        df1h = df_signal
        df4h = resample_from_1h(df1h, "4h")
        return df1h, df4h, df4h

    raise ValueError(f"Unsupported tf_signal for backtest: {tf_signal} (supported: 15m, 1h)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest one symbol (supports --timeframe override).")
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    ap.add_argument("--start", default=None, help="ISO UTC start (optional)")
    ap.add_argument("--end", default=None, help="ISO UTC end (optional)")
    ap.add_argument("--out", default="data/outputs/backtest_one", help="Output directory")
    ap.add_argument("--timeframe", default=None, help="Override signal timeframe (15m or 1h)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")

    tf_default = (settings.get("timeframes") or {}).get("signal", "15m")
    tf_signal = (args.timeframe or tf_default).strip()

    # Load cache for the selected tf_signal
    df = load_cache_csv(cache_path(repo_root, args.symbol, tf_signal))
    df = _normalize_ts(df)

    if df.empty:
        raise RuntimeError(
            f"Cache is empty for {args.symbol} {tf_signal}. "
            f"Expected file like data/cache/{args.symbol.replace('/','_')}__{tf_signal}.csv"
        )

    start_ts = _parse_iso(args.start)
    end_ts = _parse_iso(args.end)
    if start_ts is not None:
        df = df[df["ts"] >= start_ts].copy()
    if end_ts is not None:
        df = df[df["ts"] <= end_ts].copy()
    df = _normalize_ts(df)

    if df.empty:
        raise RuntimeError("Cache exists but is empty after start/end slicing.")

    # Build multi-TF frames (trend/regime depend on tf_signal)
    df_sig, df_trend, df_regime = build_multitf_for_backtest(df, tf_signal)

    # Indicators
    ich_cfg = settings.get("ichimoku") or {}
    setup_cfg = settings.get("setup_a_plus") or {}
    ich = IchimokuParams(
        tenkan=int(ich_cfg.get("tenkan", 9)),
        kijun=int(ich_cfg.get("kijun", 26)),
        spanb=int(ich_cfg.get("spanb", 52)),
        displacement=int(ich_cfg.get("displacement", 26)),
    )

    df_sig = ichimoku(df_sig, ich)
    df_trend = ichimoku(df_trend, ich)
    df_regime = ichimoku(df_regime, ich)
    df_sig["atr"] = atr_wilder(df_sig, period=14)

    aplus = APlusConfig(
        score_threshold=int(setup_cfg.get("score_threshold", 80)),
        retest_window_bars=int(setup_cfg.get("retest_window_bars", 12)),
        retest_eps=float(setup_cfg.get("retest_eps", 0.0015)),
        kijun_slope_bars=int(setup_cfg.get("kijun_slope_bars", 10)),
        max_kijun_distance_pct=float(setup_cfg.get("max_kijun_distance_pct", 0.008)),
        min_kumo_thickness_pct=float(setup_cfg.get("min_kumo_thickness_pct", 0.0035)),
    )

    # Run backtest using your existing engine. It should call should_entry/should_exit which
    # rely on df_sig/df_trend/df_regime (now consistent for 15m or 1h runs).
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_backtest_one(
        symbol=args.symbol,
        df_signal=df_sig,
        df_trend=df_trend,
        df_regime=df_regime,
        aplus=aplus,
        out_dir=out_dir,
        fee_rate=float((settings.get("costs") or {}).get("fee_rate", 0.00075)),
        slippage_rate=float((settings.get("costs") or {}).get("slippage_rate", 0.00075)),
    )

    # Print summary for CI logs
    print("[backtest_one] summary=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
