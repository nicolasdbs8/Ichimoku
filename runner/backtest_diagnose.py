from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import pandas as pd

from data.store import cache_path, load_cache_csv
from data.resample import build_multitf

from indicators.ichimoku import ichimoku, IchimokuParams
from indicators.atr import atr_wilder
from indicators.utils import kumo_top, kijun_slope

from signals.regime import btc_regime_on
from signals.trend import trend_on_1h
from signals.ichimoku_a_plus import APlusConfig, retest_kijun, compute_score


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def asof_slice(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["ts"] <= ts].copy()


def entry_preconditions(v15: pd.DataFrame, cfg: APlusConfig) -> Dict[str, Any]:
    """
    Compute A+ preconditions (without score threshold).
    Assumes v15 includes ichimoku columns and is sliced "as-of" time t (no future rows).
    """
    needed = ["ts", "close", "low", "tenkan", "kijun", "span_a", "span_b", "chikou"]
    if v15.empty or any(c not in v15.columns for c in needed):
        return {"valid": False, "reason": "missing_columns"}

    last = v15.iloc[-1]
    if any(pd.isna(last.get(c)) for c in needed):
        return {"valid": False, "reason": "nan_last_row"}

    close = float(last["close"])
    tenkan = float(last["tenkan"])
    kij = float(last["kijun"])

    kt_series = kumo_top(v15["span_a"], v15["span_b"])
    kt = kt_series.iloc[-1]
    slope = kijun_slope(v15["kijun"], cfg.kijun_slope_bars).iloc[-1]

    pre_close_above = (not pd.isna(kt)) and (close > float(kt))
    pre_tk = tenkan > kij
    pre_future = (float(last["span_a"]) > float(last["span_b"])) if (not pd.isna(last["span_a"]) and not pd.isna(last["span_b"])) else False
    pre_chikou = (not pd.isna(last["chikou"]) and not pd.isna(kt) and float(last["chikou"]) > float(kt))
    pre_slope = (not pd.isna(slope) and float(slope) > 0)

    rt = retest_kijun(v15, cfg.retest_window_bars, cfg.retest_eps)
    trigger = close > tenkan

    flags = {
        "close_above_kumo": pre_close_above,
        "tenkan_above_kijun": pre_tk,
        "future_kumo_bull": pre_future,
        "chikou_confirm": pre_chikou,
        "kijun_slope_pos": pre_slope,
        "retest_kijun": rt,
        "trigger_close_gt_tenkan": trigger,
    }

    raw_ok = all(flags.values())
    return {"valid": True, "raw_ok": raw_ok, "flags": flags}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose why A+ signals do/don't appear (BTC only).")
    ap.add_argument("--start", default=None, help="ISO UTC start (optional)")
    ap.add_argument("--end", default=None, help="ISO UTC end (optional)")
    ap.add_argument("--out", default="data/outputs/diagnose", help="output dir")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")
    universe = load_yaml(repo_root / "config" / "universe.yaml")

    # Resolve BTC symbol
    btc_rows = [
        s for s in universe.get("symbols", [])
        if isinstance(s, dict) and s.get("enabled", True) and s.get("is_btc") is True
    ]
    if not btc_rows:
        raise RuntimeError("No BTC symbol found in config/universe.yaml (need is_btc: true).")
    symbol = btc_rows[0]["symbol"]

    tf_signal = (settings.get("timeframes") or {}).get("signal", "15m")

    df15 = load_cache_csv(cache_path(repo_root, symbol, tf_signal))
    if df15.empty:
        raise RuntimeError("Cache empty. Run scan_once first (in the workflow) to fetch data.")

    df15["ts"] = pd.to_datetime(df15["ts"], utc=True)
    df15 = df15.sort_values("ts").reset_index(drop=True)

    if args.start:
        df15 = df15[df15["ts"] >= pd.to_datetime(args.start, utc=True)].copy()
    if args.end:
        df15 = df15[df15["ts"] <= pd.to_datetime(args.end, utc=True)].copy()

    # Build multi-TF and indicators
    df15, df1h, df4h = build_multitf(df15)

    ich_cfg = settings.get("ichimoku") or {}
    setup_cfg = settings.get("setup_a_plus") or {}

    ich = IchimokuParams(
        tenkan=int(ich_cfg.get("tenkan", 9)),
        kijun=int(ich_cfg.get("kijun", 26)),
        spanb=int(ich_cfg.get("spanb", 52)),
        displacement=int(ich_cfg.get("displacement", 26)),
    )

    aplus = APlusConfig(
        score_threshold=int(setup_cfg.get("score_threshold", 80)),  # we still report thresholds vs this
        retest_window_bars=int(setup_cfg.get("retest_window_bars", 12)),
        retest_eps=float(setup_cfg.get("retest_eps", 0.0015)),
        kijun_slope_bars=int(setup_cfg.get("kijun_slope_bars", 10)),
        max_kijun_distance_pct=float(setup_cfg.get("max_kijun_distance_pct", 0.008)),
        min_kumo_thickness_pct=float(setup_cfg.get("min_kumo_thickness_pct", 0.0035)),
    )

    df15i = ichimoku(df15, ich)
    df1hi = ichimoku(df1h, ich)
    df4hi = ichimoku(df4h, ich)
    df15i["atr"] = atr_wilder(df15i, period=14)  # not required for signals, but useful later

    bars_total = max(0, len(df15i) - 1)  # i+1 is used in other engines; here we just analyze i
    regime_on = 0
    trend_on = 0
    both_on = 0

    raw_setups = 0
    scored_ge_50 = 0
    scored_ge_60 = 0
    scored_ge_70 = 0
    scored_ge_80 = 0

    score_samples = []  # store scores of raw_setups (for quick sanity)

    # Condition block counts (only when gates pass)
    block_counts = {
        "close_above_kumo": 0,
        "tenkan_above_kijun": 0,
        "future_kumo_bull": 0,
        "chikou_confirm": 0,
        "kijun_slope_pos": 0,
        "retest_kijun": 0,
        "trigger_close_gt_tenkan": 0,
        "nan_last_row": 0,
    }

    # For "near miss" analysis: exactly one condition failing while others true (gates pass)
    near_miss_one_fail = {k: 0 for k in block_counts.keys() if k != "nan_last_row"}

    slope_bars = int(setup_cfg.get("kijun_slope_bars", 10))

    for i in range(len(df15i)):
        t = pd.to_datetime(df15i["ts"].iloc[i], utc=True)

        v15 = df15i.iloc[: i + 1].copy()
        v1h = asof_slice(df1hi, t)
        v4h = asof_slice(df4hi, t)

        reg = btc_regime_on(v4h, slope_bars=slope_bars)
        tr = trend_on_1h(v1h, slope_bars=slope_bars)

        if reg:
            regime_on += 1
        if tr:
            trend_on += 1
        if reg and tr:
            both_on += 1
        else:
            continue  # only diagnose setup when gates pass (because otherwise no signal anyway)

        pc = entry_preconditions(v15, aplus)
        if not pc.get("valid", False):
            if pc.get("reason") == "nan_last_row":
                block_counts["nan_last_row"] += 1
            continue

        flags = pc["flags"]

        # Count blocks (any condition false)
        for k, ok in flags.items():
            if not ok:
                block_counts[k] += 1

        # Near miss: exactly one fails
        fails = [k for k, ok in flags.items() if not ok]
        if len(fails) == 1:
            near_miss_one_fail[fails[0]] += 1

        if pc["raw_ok"]:
            raw_setups += 1
            pack = compute_score(v15, aplus)
            score = int(pack["score"])
            score_samples.append(score)

            if score >= 50:
                scored_ge_50 += 1
            if score >= 60:
                scored_ge_60 += 1
            if score >= 70:
                scored_ge_70 += 1
            if score >= 80:
                scored_ge_80 += 1

    def pct(x: int, denom: int) -> float:
        return round(100.0 * x / denom, 3) if denom > 0 else 0.0

    result: Dict[str, Any] = {
        "meta": {
            "symbol": symbol,
            "tf_signal": tf_signal,
            "start_ts": args.start,
            "end_ts": args.end,
            "score_threshold": aplus.score_threshold,
            "retest_window_bars": aplus.retest_window_bars,
            "kijun_slope_bars": aplus.kijun_slope_bars,
        },
        "counts": {
            "bars_total": bars_total,
            "regime_on": regime_on,
            "trend_on": trend_on,
            "both_on": both_on,
            "raw_setups": raw_setups,
            "score_ge_50": scored_ge_50,
            "score_ge_60": scored_ge_60,
            "score_ge_70": scored_ge_70,
            "score_ge_80": scored_ge_80,
        },
        "pct_of_bars": {
            "regime_on_pct": pct(regime_on, bars_total),
            "trend_on_pct": pct(trend_on, bars_total),
            "both_on_pct": pct(both_on, bars_total),
            "raw_setups_pct_of_gated": pct(raw_setups, both_on),
        },
        "blocks_when_gated": block_counts,
        "near_miss_exactly_one_fail_when_gated": near_miss_one_fail,
        "score_samples_raw_setups": {
            "n": len(score_samples),
            "min": min(score_samples) if score_samples else None,
            "p25": int(pd.Series(score_samples).quantile(0.25)) if score_samples else None,
            "p50": int(pd.Series(score_samples).quantile(0.50)) if score_samples else None,
            "p75": int(pd.Series(score_samples).quantile(0.75)) if score_samples else None,
            "max": max(score_samples) if score_samples else None,
        },
    }

    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnose.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[backtest_diagnose] wrote:", str(out_path))
    print("[backtest_diagnose] summary:", json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
