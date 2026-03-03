from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from backtest.metrics import summarize


def write_report(out_dir: Path, trades: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "trades.csv"
    summary_path = out_dir / "summary.json"

    trades.to_csv(trades_path, index=False)

    s = summarize(trades)
    summary = {
        "meta": meta,
        "num_trades": s.num_trades,
        "hit_rate": s.hit_rate,
        "avg_r": s.avg_r,
        "avg_win_r": s.avg_win_r,
        "avg_loss_r": s.avg_loss_r,
        "expectancy_r": s.expectancy_r,
        "profit_factor": s.profit_factor,
        "max_drawdown": s.max_drawdown,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
