import pandas as pd
from signals.regime import btc_regime_on
from signals.trend import trend_on_1h


def test_regime_off_on_empty():
    assert btc_regime_on(pd.DataFrame()) is False


def test_trend_off_on_empty():
    assert trend_on_1h(pd.DataFrame()) is False
