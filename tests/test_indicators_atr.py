import pandas as pd
from indicators.atr import atr_wilder


def test_atr_constant_range_converges():
    # If high-low constant and no gaps, ATR should converge to that constant
    n = 60
    ts = pd.date_range("2026-03-01 00:00:00+00:00", periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": 100.0,
            "high": 110.0,
            "low": 100.0,
            "close": 105.0,
            "volume": 1.0,
        }
    )
    atr = atr_wilder(df, period=14)
    # after enough points, should be close to 10
    assert abs(float(atr.iloc[-1]) - 10.0) < 1e-6
