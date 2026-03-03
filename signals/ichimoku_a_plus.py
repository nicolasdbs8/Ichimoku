from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import pandas as pd

from indicators.utils import kumo_top, kumo_bottom, kijun_slope, kumo_thickness_pct


@dataclass(frozen=True)
class APlusConfig:
    score_threshold: int = 80
    retest_window_bars: int = 12
    retest_eps: float = 0.0015
    kijun_slope_bars: int = 10
    max_kijun_distance_pct: float = 0.008
    min_kumo_thickness_pct: float = 0.0035


def _last_row_ok(df: pd.DataFrame, cols: List[str]) -> bool:
    if df.empty:
        return False
    last = df.iloc[-1]
    for c in cols:
        if c not in df.columns or pd.isna(last.get(c)):
            return False
    return True


def retest_kijun(df15: pd.DataFrame, window_bars: int, eps: float) -> bool:
    if len(df15) < window_bars + 2:
        return False
    recent = df15.iloc[-(window_bars + 1):-1]  # exclude current signal candle
    cond = recent["low"] <= (recent["kijun"] * (1.0 + eps))
    return bool(cond.any())


def compute_score(df15: pd.DataFrame, cfg: APlusConfig) -> Dict[str, Any]:
    """
    Returns dict with:
      score, points_breakdown, flags used
    Requires df15 columns: close, low, tenkan, kijun, span_a, span_b, chikou
    """
    last = df15.iloc[-1]
    close = float(last["close"])
    tenkan = float(last["tenkan"])
    kij = float(last["kijun"])
    sa = last["span_a"]
    sb = last["span_b"]
    chik = last["chikou"]

    kt = kumo_top(df15["span_a"], df15["span_b"]).iloc[-1]
    kb = kumo_bottom(df15["span_a"], df15["span_b"]).iloc[-1]
    slope = kijun_slope(df15["kijun"], cfg.kijun_slope_bars).iloc[-1]
    thick = kumo_thickness_pct(df15["span_a"], df15["span_b"], df15["close"]).iloc[-1]

    points = {}
    points["close_above_kumo"] = 20 if close > float(kt) else 0
    points["tenkan_above_kijun"] = 10 if tenkan > kij else 0
    points["future_kumo_bull"] = 10 if (not pd.isna(sa) and not pd.isna(sb) and float(sa) > float(sb)) else 0
    points["chikou_confirm"] = 10 if (not pd.isna(chik) and not pd.isna(kt) and float(chik) > float(kt)) else 0
    points["kijun_slope_pos"] = 10 if (not pd.isna(slope) and float(slope) > 0) else 0

    rt = retest_kijun(df15, cfg.retest_window_bars, cfg.retest_eps)
    points["retest_kijun"] = 10 if rt else 0

    dist = abs(close - kij) / close
    points["kijun_distance_ok"] = 10 if dist <= cfg.max_kijun_distance_pct else 0

    points["kumo_thickness_ok"] = 10 if (not pd.isna(thick) and float(thick) >= cfg.min_kumo_thickness_pct) else 0

    score = int(sum(points.values()))

    return {
        "score": score,
        "points": points,
        "metrics": {
            "kumo_top": float(kt) if not pd.isna(kt) else None,
            "kumo_bottom": float(kb) if not pd.isna(kb) else None,
            "kijun_slope": float(slope) if not pd.isna(slope) else None,
            "kijun_dist_pct": float(dist),
            "kumo_thickness_pct": float(thick) if not pd.isna(thick) else None,
            "retest_kijun": rt,
        },
    }


def a_plus_entry_signal(
    df15: pd.DataFrame,
    cfg: APlusConfig,
) -> Optional[Dict[str, Any]]:
    """
    Returns payload dict if ENTRY A+ triggered on last candle close, else None.
    Requires df15 last row has valid indicator values (no NaN leaks).
    """
    needed = ["ts", "close", "low", "tenkan", "kijun", "span_a", "span_b", "chikou"]
    if not _last_row_ok(df15, needed):
        return None

    last = df15.iloc[-1]
    close = float(last["close"])
    tenkan = float(last["tenkan"])
    kij = float(last["kijun"])

    kt = kumo_top(df15["span_a"], df15["span_b"]).iloc[-1]
    slope = kijun_slope(df15["kijun"], cfg.kijun_slope_bars).iloc[-1]

    # Preconditions (15m)
    pre_close_above = close > float(kt) if not pd.isna(kt) else False
    pre_tk = tenkan > kij
    pre_future = (float(last["span_a"]) > float(last["span_b"])) if (not pd.isna(last["span_a"]) and not pd.isna(last["span_b"])) else False
    pre_chikou = (float(last["chikou"]) > float(kt)) if (not pd.isna(last["chikou"]) and not pd.isna(kt)) else False
    pre_slope = (float(slope) > 0) if not pd.isna(slope) else False

    # Retest & Trigger
    rt = retest_kijun(df15, cfg.retest_window_bars, cfg.retest_eps)
    trigger = close > tenkan  # close above tenkan on signal candle

    if not (pre_close_above and pre_tk and pre_future and pre_chikou and pre_slope and rt and trigger):
        return None

    score_pack = compute_score(df15, cfg)
    if score_pack["score"] < cfg.score_threshold:
        return None

    return {
        "type": "ENTRY",
        "setup": "ICHIMOKU_A_PLUS_LONG",
        "ts": pd.to_datetime(last["ts"], utc=True).isoformat(),
        "score": score_pack["score"],
        "score_detail": score_pack,
    }


def a_plus_exit_signal(df15: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    EXIT signal for invalidation:
      - close < kijun  (next open exit model)
      - or TK cross confirmed: tenkan < kijun AND close < kumo_top
    """
    needed = ["ts", "close", "tenkan", "kijun", "span_a", "span_b"]
    if not _last_row_ok(df15, needed):
        return None

    last = df15.iloc[-1]
    close = float(last["close"])
    tenkan = float(last["tenkan"])
    kij = float(last["kijun"])
    kt = kumo_top(df15["span_a"], df15["span_b"]).iloc[-1]

    if close < kij:
        return {"type": "EXIT", "reason": "KIJUN_BREAK", "ts": pd.to_datetime(last["ts"], utc=True).isoformat()}

    if (tenkan < kij) and (not pd.isna(kt) and close < float(kt)):
        return {"type": "EXIT", "reason": "TK_CROSS_CONFIRMED", "ts": pd.to_datetime(last["ts"], utc=True).isoformat()}

    return None
