import pandas as pd
from data.resample import resample_from_15m


def test_no_use_of_incomplete_current_candle():
    # If data ends at 00:45, the last closed 1h boundary is 00:00 (because 01:00 not complete)
    # Our resample drops NaNs, but should not fabricate a 01:00 candle without 4x 15m.
    ts = pd.date_range("2026-03-01 00:15:00+00:00", periods=3, freq="15min")  # 00:15,00:30,00:45
    df15 = pd.DataFrame(
        {
            "ts": ts,
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    df1h = resample_from_15m(df15, "1h")
    # Should not produce a 01:00 candle (incomplete hour)
    assert not any(t.hour == 1 and t.minute == 0 for t in df1h["ts"])
