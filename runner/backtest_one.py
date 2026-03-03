from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import yaml

from data.store import cache_path, load_cache_csv
from data.resample import resample_from_15m

from indicators.ichimoku import ichimoku, IchimokuParams
from indicators.atr import atr_wilder

from signals.regime import btc_regime_on
from signals.trend import trend_on_1h
from signals.ichimoku_a_plus import APlusConfig, a_plus_entry_signal, a_plus_exit_signal


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _parse_iso(iso: Optional[str]) -> Optional[pd.Timestamp]:
    if not iso:
        return None
    return pd.to_datetime(iso.replace("Z", "+00:00"), utc=True)


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return df


def asof_slice(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["ts"] <= ts].copy()


def resample_from_1h(df1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV from 1h base to higher TF conservatively:
    - label='right', closed='right'
    - require complete buckets (e.g. 4x1h for 4h), otherwise drop.
    """
    dfi = df1h.copy()
    dfi["ts"] = pd.to_datetime(dfi["ts"], utc=True)
    dfi = dfi.set_index("ts").sort_index()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = dfi.resample(rule, label="right", closed="right").agg(agg)

    cnt = dfi["close"].resample(rule, label="right", closed="right").count()

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
    return out


def build_multitf_for_backtest(df_signal: pd.DataFrame, tf_signal: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_signal, df_trend, df_regime)

    - If tf_signal == '15m': trend=1h, regime=4h built from 15m (canonical)
    - If tf_signal == '1h' : trend=4h, regime=4h built from 1h (fallback mode)
      (trend_on_1h() is name-only; it works on any TF as long as columns exist)
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

    raise ValueError(f"Unsupported tf_signal: {tf_signal} (supported: 15m, 1h)")


def apply_costs_entry(price: float, fee_rate: float, slippage_rate: float) -> float:
    return price * (1.0 + fee_rate + slippage_rate)


def apply_costs_exit(price: float, fee_rate: float, slippage_rate: float) -> float:
    return price * (1.0 - fee_rate - slippage_rate)


def compute_summary(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {
            "num_trades": 0,
            "hit_rate": 0.0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    r = trades["r_multiple"].astype(float)
    num = int(len(trades))
    wins = r[r > 0]
    losses = r[r <= 0]

    hit = float((r > 0).mean())

    avg_r = float(r.mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = float(hit * avg_win + (1.0 - hit) * avg_loss)

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0  # negative or 0
    profit_factor = (gross_win / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_win > 0 else 0.0)

    # equity curve in "R units"
    eq = r.cumsum()
    dd = (eq - eq.cummax())
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "num_trades": num,
        "hit_rate": round(hit, 6),
        "avg_r": round(avg_r, 6),
        "avg_win_r": round(avg_win, 6),
        "avg_loss_r": round(avg_loss, 6),
        "expectancy_r": round(expectancy, 6),
        "profit_factor": round(profit_factor, 6) if profit_factor != float("inf") else float("inf"),
        "max_drawdown": round(max_dd, 6),
    }


def run_backtest_one(
    *,
    symbol: str,
    df_signal: pd.DataFrame,
    df_trend: pd.DataFrame,
    df_regime: pd.DataFrame,
    aplus: APlusConfig,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
    atr_stop_mult: float,
) -> pd.DataFrame:
    """
    Execution model (conservative):
    - Evaluate signals on CLOSED bars only (index i).
    - Entry: next bar open (i+1 open).
    - Stop: active from entry bar; intrabar stop if low <= stop => exit at stop (same bar).
    - Exit (invalidation): if exit signal at bar i close => exit next bar open (i+1 open).
    Position sizing: R-based, with base_equity=1.0 so qty scales to risk_per_trade only.
    """
    base_equity = 1.0

    trades: List[Dict[str, Any]] = []
    in_pos = False

    entry_price_eff = 0.0
    entry_ts: Optional[pd.Timestamp] = None
    stop_price = 0.0
    qty = 0.0

    slope_bars = int(aplus.kijun_slope_bars)

    n = len(df_signal)
    for i in range(n - 1):  # need i+1 open for entry/exit
        t = pd.to_datetime(df_signal["ts"].iloc[i], utc=True)

        v_sig = df_signal.iloc[: i + 1].copy()
        v_tr = asof_slice(df_trend, t)
        v_rg = asof_slice(df_regime, t)

        reg_on = btc_regime_on(v_rg, slope_bars=slope_bars)
        tr_on = trend_on_1h(v_tr, slope_bars=slope_bars)

        # ---------- manage open position ----------
        if in_pos:
            # stop intrabar on NEXT bar (the bar we're iterating is closed; stop needs actual bar range of current bar)
            bar = df_signal.iloc[i]
            low = float(bar["low"])

            if low <= stop_price:
                exit_px_eff = apply_costs_exit(stop_price, fee_rate, slippage_rate)
                pnl = qty * (exit_px_eff - entry_price_eff)
                r_mult = pnl / (base_equity * risk_per_trade) if risk_per_trade > 0 else 0.0

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_ts": entry_ts.isoformat(),
                        "exit_ts": pd.to_datetime(bar["ts"], utc=True).isoformat(),
                        "entry_price": entry_price_eff,
                        "exit_price": exit_px_eff,
                        "qty": qty,
                        "pnl": pnl,
                        "r_multiple": r_mult,
                        "exit_reason": "STOP_INTRABAR",
                    }
                )
                in_pos = False
                entry_ts = None
                continue

            # exit signal at close -> exit next open
            exit_sig = a_plus_exit_signal(v_sig)
            if exit_sig is not None:
                nxt = df_signal.iloc[i + 1]
                exit_px_raw = float(nxt["open"])
                exit_px_eff = apply_costs_exit(exit_px_raw, fee_rate, slippage_rate)

                pnl = qty * (exit_px_eff - entry_price_eff)
                r_mult = pnl / (base_equity * risk_per_trade) if risk_per_trade > 0 else 0.0

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_ts": entry_ts.isoformat(),
                        "exit_ts": pd.to_datetime(nxt["ts"], utc=True).isoformat(),
                        "entry_price": entry_price_eff,
                        "exit_price": exit_px_eff,
                        "qty": qty,
                        "pnl": pnl,
                        "r_multiple": r_mult,
                        "exit_reason": exit_sig.get("reason") or "EXIT",
                    }
                )
                in_pos = False
                entry_ts = None
                continue

            # keep holding
            continue

        # ---------- flat: look for entry ----------
        if not (reg_on and tr_on):
            continue

        entry = a_plus_entry_signal(v_sig, aplus)
        if entry is None:
            continue

        # entry next open
        nxt = df_signal.iloc[i + 1]
        entry_px_raw = float(nxt["open"])
        entry_price_eff = apply_costs_entry(entry_px_raw, fee_rate, slippage_rate)
        entry_ts = pd.to_datetime(nxt["ts"], utc=True)

        # ATR-based stop from signal context (use ATR at bar i)
        atr_val = float(v_sig["atr"].iloc[-1]) if "atr" in v_sig.columns and pd.notna(v_sig["atr"].iloc[-1]) else None
        if atr_val is None or atr_val <= 0:
            # cannot size/stop safely without ATR -> skip trade
            entry_ts = None
            continue

        stop_price = entry_px_raw - atr_stop_mult * atr_val  # stop defined on raw prices
        risk_dist = entry_px_raw - stop_price
        if risk_dist <= 0:
            entry_ts = None
            continue

        risk_amt = base_equity * risk_per_trade
        qty = risk_amt / risk_dist

        in_pos = True

    return pd.DataFrame(trades)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest one symbol from cache (runner-friendly).")
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    ap.add_argument("--start", default=None, help="ISO UTC start (optional)")
    ap.add_argument("--end", default=None, help="ISO UTC end (optional)")
    ap.add_argument("--out", default="data/outputs/backtest_one", help="Output directory")
    ap.add_argument("--timeframe", default=None, help="Override tf_signal (15m or 1h)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")

    tf_default = (settings.get("timeframes") or {}).get("signal", "15m")
    tf_signal = (args.timeframe or tf_default).strip()

    df = load_cache_csv(cache_path(repo_root, args.symbol, tf_signal))
    df = _normalize_ts(df)
    if df.empty:
        raise RuntimeError(
            f"Cache is empty for {args.symbol} {tf_signal}. "
            f"Expected: data/cache/{args.symbol.replace('/','_')}__{tf_signal}.csv"
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

    df_sig, df_trend, df_regime = build_multitf_for_backtest(df, tf_signal)

    # Indicators
    ich_cfg = settings.get("ichimoku") or {}
    setup_cfg = settings.get("setup_a_plus") or {}
    costs_cfg = settings.get("costs") or {}
    bt_cfg = settings.get("backtest") or {}

    ich = IchimokuParams(
        tenkan=int(ich_cfg.get("tenkan", 9)),
        kijun=int(ich_cfg.get("kijun", 26)),
        spanb=int(ich_cfg.get("spanb", 52)),
        displacement=int(ich_cfg.get("displacement", 26)),
    )

    df_sig = ichimoku(df_sig, ich)
    df_trend = ichimoku(df_trend, ich)
    df_regime = ichimoku(df_regime, ich)

    atr_period = int(bt_cfg.get("atr_period", 14))
    df_sig["atr"] = atr_wilder(df_sig, period=atr_period)

    aplus = APlusConfig(
        score_threshold=int(setup_cfg.get("score_threshold", 80)),
        retest_window_bars=int(setup_cfg.get("retest_window_bars", 12)),
        retest_eps=float(setup_cfg.get("retest_eps", 0.0015)),
        kijun_slope_bars=int(setup_cfg.get("kijun_slope_bars", 10)),
        max_kijun_distance_pct=float(setup_cfg.get("max_kijun_distance_pct", 0.008)),
        min_kumo_thickness_pct=float(setup_cfg.get("min_kumo_thickness_pct", 0.0035)),
    )

    fee_rate = float(costs_cfg.get("fee_rate", 0.001))
    slippage_rate = float(costs_cfg.get("slippage_rate", 0.0005))

    risk_per_trade = float(bt_cfg.get("risk_per_trade", 0.01))
    atr_stop_mult = float(bt_cfg.get("atr_stop_mult", 2.0))

    trades = run_backtest_one(
        symbol=args.symbol,
        df_signal=df_sig,
        df_trend=df_trend,
        df_regime=df_regime,
        aplus=aplus,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        risk_per_trade=risk_per_trade,
        atr_stop_mult=atr_stop_mult,
    )

    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)

    summary = {
        "meta": {
            "symbol": args.symbol,
            "tf_signal": tf_signal,
            "start_ts": args.start,
            "end_ts": args.end,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "risk_per_trade": risk_per_trade,
            "atr_period": atr_period,
            "atr_stop_mult": atr_stop_mult,
            "score_threshold": int(aplus.score_threshold),
        }
    }
    summary.update(compute_summary(trades))

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[backtest_one] wrote:", str(trades_path))
    print("[backtest_one] summary=" + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
