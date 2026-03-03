import pandas as pd
from data.resample import resample_from_15m


def test_resample_alignment_1h_boundaries():
    # Build 15m data that ends exactly on hour boundaries
    ts = pd.date_range("2026-03-01 00:15:00+00:00", periods=8, freq="15min")  # ends 02:00
    df15 = pd.DataFrame(
        {
            "ts": ts,
            "open": range(8),
            "high": range(8),
            "low": range(8),
            "close": range(8),
            "volume": [1.0] * 8,
        }
    )
    df1h = resample_from_15m(df15, "1h")
    # With label="right" boundaries should be :00
    assert all(t.minute == 0 for t in df1h["ts"])
