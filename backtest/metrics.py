from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


@dataclass(frozen=True)
class Summary:
    num_trades: int
    hit_rate: float
    avg_r: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float
    profit_factor: float
    max_drawdown: float


def equity_curve_from_trades(trades: pd.DataFrame, start_equity: float = 1.0) -> pd.DataFrame:
    """
    trades must include:
      - exit_ts
      - pnl (in quote currency)
    Returns df with columns: ts, equity
    """
    if trades.empty:
        return pd.DataFrame(columns=["ts", "equity"])

    df = trades.copy()
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df = df.sort_values("exit_ts").reset_index(drop=True)

    eq = start_equity
    rows = []
    for _, r in df.iterrows():
        eq += float(r["pnl"])
        rows.append({"ts": r["exit_ts"], "equity": eq})
    return pd.DataFrame(rows)


def max_drawdown(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    eq = equity_df["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())


def summarize(trades: pd.DataFrame) -> Summary:
    if trades.empty:
        return Summary(
            num_trades=0,
            hit_rate=0.0,
            avg_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            expectancy_r=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
        )

    r = trades["r_multiple"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]

    num = len(trades)
    hit = float((r > 0).mean())

    avg_r = float(r.mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = avg_r

    gross_win = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    eq = equity_curve_from_trades(trades, start_equity=1.0)
    mdd = max_drawdown(eq)

    return Summary(
        num_trades=num,
        hit_rate=hit,
        avg_r=avg_r,
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        expectancy_r=expectancy,
        profit_factor=pf,
        max_drawdown=mdd,
    )
