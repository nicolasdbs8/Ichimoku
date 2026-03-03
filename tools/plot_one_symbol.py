from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from data.store import load_cache_csv, cache_path
from indicators.ichimoku import ichimoku, IchimokuParams
from indicators.utils import kumo_top, kumo_bottom
from indicators.atr import atr_wilder


def main():
    repo_root = Path(__file__).resolve().parents[1]
    symbol = "BTC/USDT"
    tf = "15m"

    df = load_cache_csv(cache_path(repo_root, symbol, tf))
    if df.empty:
        raise RuntimeError("Cache empty. Run scan_once first to fetch BTC 15m.")

    out = ichimoku(df, IchimokuParams())
    out["kumo_top"] = kumo_top(out["span_a"], out["span_b"])
    out["kumo_bottom"] = kumo_bottom(out["span_a"], out["span_b"])
    out["atr14"] = atr_wilder(out, 14)

    tail = out.tail(300).copy()
    tail = tail.set_index("ts")

    plt.figure()
    plt.plot(tail.index, tail["close"], label="close")
    plt.plot(tail.index, tail["tenkan"], label="tenkan")
    plt.plot(tail.index, tail["kijun"], label="kijun")
    plt.plot(tail.index, tail["kumo_top"], label="kumo_top")
    plt.plot(tail.index, tail["kumo_bottom"], label="kumo_bottom")
    plt.legend()
    plt.title(f"{symbol} {tf} Ichimoku")
    plt.show()


if __name__ == "__main__":
    main()
