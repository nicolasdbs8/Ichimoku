import pandas as pd
from indicators.ichimoku import ichimoku, IchimokuParams


def make_df(n=200):
    ts = pd.date_range("2026-03-01 00:00:00+00:00", periods=n, freq="15min")
    # simple increasing series
    base = pd.Series(range(n), dtype="float64")
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": base + 1,
            "high": base + 2,
            "low": base,
            "close": base + 1.5,
            "volume": 1.0,
        }
    )
    return df


def test_ichimoku_columns_exist():
    df = make_df()
    out = ichimoku(df, IchimokuParams())
    for col in ["tenkan", "kijun", "span_a", "span_b", "chikou"]:
        assert col in out.columns


def test_ichimoku_shifts():
    df = make_df(120)
    p = IchimokuParams(tenkan=9, kijun=26, spanb=52, displacement=26)
    out = ichimoku(df, p)

    # chikou is close shifted -26 => at index i, chikou[i] == close[i+26]
    i = 10
    assert out["chikou"].iloc[i] == out["close"].iloc[i + p.displacement]

    # span_a shifted +26 => at index i+26, span_a[i+26] == (tenkan[i]+kijun[i])/2
    i = 40
    expected = (out["tenkan"].iloc[i] + out["kijun"].iloc[i]) / 2.0
    assert out["span_a"].iloc[i + p.displacement] == expected
