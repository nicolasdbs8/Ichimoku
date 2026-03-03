from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    fee_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 5 bps
    risk_per_trade: float = 0.01  # 1% equity risked per trade
    atr_period: int = 14
    atr_stop_mult: float = 2.0


@dataclass
class Position:
    is_open: bool = False
    entry_ts: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    qty: float = 0.0
    reason: Optional[str] = None


def apply_costs_entry(price: float, fee_rate: float, slippage_rate: float) -> float:
    # Long entry: worse price due to slippage; fees paid on notional
    px = price * (1.0 + slippage_rate)
    px = px * (1.0 + fee_rate)
    return px


def apply_costs_exit(price: float, fee_rate: float, slippage_rate: float) -> float:
    # Long exit: worse price due to slippage; fees paid on notional
    px = price * (1.0 - slippage_rate)
    px = px * (1.0 - fee_rate)
    return px


def compute_qty(equity: float, risk_per_trade: float, entry_px: float, stop_px: float) -> float:
    risk_per_unit = entry_px - stop_px
    if risk_per_unit <= 0:
        return 0.0
    risk_budget = equity * risk_per_trade
    return risk_budget / risk_per_unit


def stop_hit_intrabar(bar_low: float, stop_px: float) -> bool:
    return bar_low <= stop_px


def stop_fill_price(stop_px: float) -> float:
    # Conservative: assume filled exactly at stop price (could be worse with gaps; later we can model gaps)
    return stop_px
