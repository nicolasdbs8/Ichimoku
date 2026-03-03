from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd


@dataclass
class SymbolState:
    state: str = "FLAT"  # FLAT or IN_SIGNALLED_LONG
    cooldown_until_ts: Optional[pd.Timestamp] = None


class DedupStateStore:
    """
    In-memory store (MVP). Later we persist to SQLite.
    """
    def __init__(self):
        self._state: Dict[str, SymbolState] = {}

    def get(self, symbol: str) -> SymbolState:
        return self._state.get(symbol, SymbolState())

    def set(self, symbol: str, st: SymbolState) -> None:
        self._state[symbol] = st


def in_cooldown(st: SymbolState, now_ts: pd.Timestamp) -> bool:
    if st.cooldown_until_ts is None:
        return False
    return now_ts < st.cooldown_until_ts


def enter_long(st: SymbolState, now_ts: pd.Timestamp, cooldown_bars: int, bar_minutes: int = 15) -> SymbolState:
    cooldown = pd.Timedelta(minutes=bar_minutes * cooldown_bars)
    return SymbolState(state="IN_SIGNALLED_LONG", cooldown_until_ts=now_ts + cooldown)


def exit_to_flat(st: SymbolState) -> SymbolState:
    return SymbolState(state="FLAT", cooldown_until_ts=st.cooldown_until_ts)
