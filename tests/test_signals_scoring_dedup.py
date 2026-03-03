import pandas as pd
from signals.dedup import SymbolState, in_cooldown, enter_long
from signals.ichimoku_a_plus import APlusConfig, compute_score


def test_dedup_cooldown():
    st = SymbolState()
    now = pd.Timestamp("2026-03-01 00:00:00+00:00")
    st2 = enter_long(st, now, cooldown_bars=12, bar_minutes=15)
    assert in_cooldown(st2, now + pd.Timedelta(minutes=10)) is True
    assert in_cooldown(st2, now + pd.Timedelta(hours=4)) is False


def test_compute_score_runs():
    # Minimal df with required columns and no NaNs on last row
    n = 200
    ts = pd.date_range("2026-03-01 00:00:00+00:00", periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "ts": ts,
            "close": [100.0] * n,
            "low": [99.0] * n,
            "tenkan": [101.0] * n,
            "kijun": [100.0] * n,
            "span_a": [101.0] * n,
            "span_b": [100.0] * n,
            "chikou": [200.0] * n,
        }
    )
    cfg = APlusConfig()
    pack = compute_score(df, cfg)
    assert "score" in pack and "points" in pack
