import pandas as pd
from indicators.ichimoku import ichimoku, IchimokuParams


def test_no_nan_leaks_ichimoku_edges():
    # Small sample where indicators cannot be fully computed
    n = 60
    ts = pd.date_range("2026-03-01 00:00:00+00:00", periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": range(n),
            "high": [x + 1 for x in range(n)],
            "low": [x - 1 for x in range(n)],
            "close": range(n),
            "volume": 1.0,
        }
    )
    p = IchimokuParams(tenkan=9, kijun=26, spanb=52, displacement=26)
    out = ichimoku(df, p)

    # Before kijun window, kijun must be NaN
    assert out["kijun"].iloc[p.kijun - 2] != out["kijun"].iloc[p.kijun - 2]  # NaN check
    assert out["kijun"].iloc[p.kijun - 1] == out["kijun"].iloc[p.kijun - 1]  # not NaN

    # span_a is shifted forward +26, so early part must be NaN
    assert out["span_a"].iloc[0] != out["span_a"].iloc[0]  # NaN

    # chikou shifted backward -26, so tail must be NaN
    assert out["chikou"].iloc[-1] != out["chikou"].iloc[-1]  # NaN
