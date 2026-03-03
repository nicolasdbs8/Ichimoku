from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from data.resample import build_multitf
from indicators.ichimoku import ichimoku, IchimokuParams
from indicators.atr import atr_wilder
from signals.regime import btc_regime_on
from signals.trend import trend_on_1h
from signals.ichimoku_a_plus import APlusConfig, a_plus_entry_signal, a_plus_exit_signal
from backtest.execution import (
    ExecutionConfig,
    Position,
    apply_costs_entry,
    apply_costs_exit,
    compute_qty,
    stop_hit_intrabar,
    stop_fill_price,
)


def _asof_slice(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["ts"] <= ts].copy()


def _last_row_asof(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
    if df.empty:
        return None
    sub = df[df["ts"] <= ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    timeframe_signal: str = "15m"
    start_ts: Optional[str] = None  # ISO or None
    end_ts: Optional[str] = None
    # gating params
    kijun_slope_bars: int = 10


def run_backtest_one_symbol(
    df15_raw: pd.DataFrame,
    bt: BacktestConfig,
    ich: IchimokuParams,
    aplus: APlusConfig,
    exe: ExecutionConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backtest rules (Phase 7):
    - Signals computed on 15m close (last closed candle)
    - Entry: next open after signal candle
    - Stop: intrabar if low <= stop (conservative)
    - Exit: invalidation (via a_plus_exit_signal) -> next open
    - Costs: fee + slippage each side
    - Position sizing: risk_per_trade fraction of equity using (entry-stop) risk
    - Independent per symbol (single position max)
    """

    df15 = df15_raw.copy()
    df15["ts"] = pd.to_datetime(df15["ts"], utc=True)
    df15 = df15.sort_values("ts").reset_index(drop=True)

    if bt.start_ts:
        df15 = df15[df15["ts"] >= pd.to_datetime(bt.start_ts, utc=True)].copy()
    if bt.end_ts:
        df15 = df15[df15["ts"] <= pd.to_datetime(bt.end_ts, utc=True)].copy()

    # Build 1h / 4h from 15m and compute Ichimoku on each TF
    df15, df1h, df4h = build_multitf(df15)
    df15i = ichimoku(df15, ich)
    df1hi = ichimoku(df1h, ich)
    df4hi = ichimoku(df4h, ich)

    # ATR for stop sizing
    df15i["atr"] = atr_wilder(df15i, period=exe.atr_period)

    equity = 1.0
    pos = Position(is_open=False)
    trades: List[Dict[str, Any]] = []

    # iterate over bars; we need i+1 for next open fills
    for i in range(len(df15i) - 1):
        t = pd.to_datetime(df15i["ts"].iloc[i], utc=True)

        # Build "as-of" views to avoid lookahead
        v15 = df15i.iloc[: i + 1].copy()
        v1h = _asof_slice(df1hi, t)
        v4h = _asof_slice(df4hi, t)

        # Gates as-of time t
        reg_on = btc_regime_on(v4h, slope_bars=bt.kijun_slope_bars)
        tr_on = trend_on_1h(v1h, slope_bars=bt.kijun_slope_bars)

        # next bar prices for next-open execution
        next_open = float(df15i["open"].iloc[i + 1])
        next_ts = pd.to_datetime(df15i["ts"].iloc[i + 1], utc=True)

        bar_low = float(df15i["low"].iloc[i + 1])   # intrabar stop happens during next bar
        bar_high = float(df15i["high"].iloc[i + 1])

        # --------- Manage open position (stop first, then exit next-open) ----------
        if pos.is_open:
            assert pos.entry_price is not None and pos.stop_price is not None

            # Intrabar stop on bar i+1
            if stop_hit_intrabar(bar_low, pos.stop_price):
                exit_px_raw = stop_fill_price(pos.stop_price)
                exit_px = apply_costs_exit(exit_px_raw, exe.fee_rate, exe.slippage_rate)

                pnl = (exit_px - pos.entry_price) * pos.qty
                risk_per_unit = pos.entry_price - pos.stop_price
                r_mult = (pnl / (risk_per_unit * pos.qty)) if (risk_per_unit > 0 and pos.qty > 0) else 0.0

                equity += pnl

                trades.append(
                    {
                        "symbol": bt.symbol,
                        "entry_ts": pos.entry_ts.isoformat(),
                        "exit_ts": next_ts.isoformat(),
                        "entry_price": pos.entry_price,
                        "exit_price": exit_px,
                        "qty": pos.qty,
                        "pnl": pnl,
                        "r_multiple": r_mult,
                        "exit_reason": "STOP_INTRABAR",
                    }
                )
                pos = Position(is_open=False)
                continue  # position closed; do not process exit/entry on same step

            # Exit signal computed on candle close at time t (i), executed next open (i+1)
            exit_sig = a_plus_exit_signal(v15)
            if exit_sig is not None:
                exit_px = apply_costs_exit(next_open, exe.fee_rate, exe.slippage_rate)

                pnl = (exit_px - pos.entry_price) * pos.qty
                risk_per_unit = pos.entry_price - pos.stop_price
                r_mult = (pnl / (risk_per_unit * pos.qty)) if (risk_per_unit > 0 and pos.qty > 0) else 0.0

                equity += pnl

                trades.append(
                    {
                        "symbol": bt.symbol,
                        "entry_ts": pos.entry_ts.isoformat(),
                        "exit_ts": next_ts.isoformat(),
                        "entry_price": pos.entry_price,
                        "exit_price": exit_px,
                        "qty": pos.qty,
                        "pnl": pnl,
                        "r_multiple": r_mult,
                        "exit_reason": exit_sig.get("reason", "EXIT"),
                    }
                )
                pos = Position(is_open=False)
                continue

        # --------- Entry logic (signal at time t, enter next open) ----------
        if not pos.is_open and reg_on and tr_on:
            entry_sig = a_plus_entry_signal(v15, aplus)
            if entry_sig is not None:
                # stop = entry - atr_mult*ATR
                atr = float(v15["atr"].iloc[-1]) if "atr" in v15.columns and pd.notna(v15["atr"].iloc[-1]) else None
                if atr is None or atr <= 0:
                    continue  # cannot size/stop without ATR

                entry_px = apply_costs_entry(next_open, exe.fee_rate, exe.slippage_rate)
                stop_px = entry_px - exe.atr_stop_mult * atr

                qty = compute_qty(equity, exe.risk_per_trade, entry_px, stop_px)
                if qty <= 0:
                    continue

                pos = Position(
                    is_open=True,
                    entry_ts=next_ts,
                    entry_price=entry_px,
                    stop_price=stop_px,
                    qty=qty,
                    reason="ENTRY_A_PLUS",
                )

    trades_df = pd.DataFrame(trades)

    meta = {
        "symbol": bt.symbol,
        "tf_signal": bt.timeframe_signal,
        "start_ts": bt.start_ts,
        "end_ts": bt.end_ts,
        "fee_rate": exe.fee_rate,
        "slippage_rate": exe.slippage_rate,
        "risk_per_trade": exe.risk_per_trade,
        "atr_period": exe.atr_period,
        "atr_stop_mult": exe.atr_stop_mult,
        "score_threshold": aplus.score_threshold,
    }
    return trades_df, meta
