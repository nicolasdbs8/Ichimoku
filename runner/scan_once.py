from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import time

import yaml
import pandas as pd

from data.fetch_ohlcv import make_exchange, fetch_ohlcv_incremental
from data.store import ensure_cache_dir, cache_path, load_cache_csv, save_cache_csv
from data.resample import detect_missing_candles, build_multitf, compute_freshness

from indicators.ichimoku import ichimoku, IchimokuParams
from signals.regime import btc_regime_on
from signals.trend import trend_on_1h
from signals.ichimoku_a_plus import APlusConfig, a_plus_entry_signal, a_plus_exit_signal

from runner.notify import make_telegram_client_from_env
from notifier.templates import format_entry, format_exit

# Phase 6 (SQLite)
from storage.db import connect, init_schema, insert_signal, load_state, save_state


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected mapping at root")
    return data


def json_safe(obj):
    import pandas as _pd

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, _pd.Timestamp):
        return obj.isoformat()
    return obj


def main() -> None:
    t0 = time.time()

    repo_root = Path(__file__).resolve().parents[1]
    settings = load_yaml(repo_root / "config" / "settings.yaml")
    universe = load_yaml(repo_root / "config" / "universe.yaml")

    # ---- Phase 6: DB init ----
    conn = connect(repo_root)
    init_schema(conn)

    exchange_name = (settings.get("exchange") or {}).get("name", "kraken")
    enable_rl = (settings.get("exchange") or {}).get("enable_rate_limit", True)
    tf_signal = (settings.get("timeframes") or {}).get("signal", "15m")

    symbols = universe.get("symbols", [])
    btc_rows = [
        s
        for s in symbols
        if isinstance(s, dict) and s.get("enabled", True) and s.get("is_btc") is True
    ]
    if not btc_rows:
        raise RuntimeError("No BTC symbol found in config/universe.yaml (need is_btc: true).")
    btc_symbol = btc_rows[0]["symbol"]

    # -----------------------
    # Phase 2: data fetch + cache + resample + freshness
    # -----------------------
    ensure_cache_dir(repo_root)
    cpath = cache_path(repo_root, btc_symbol, tf_signal)

    cached = load_cache_csv(cpath)
    since_ms = None
    if not cached.empty:
        last_ts = pd.to_datetime(cached["ts"].iloc[-1], utc=True)
        since_ms = int(last_ts.value // 1_000_000) + 1  # ns -> ms

    ex = make_exchange(exchange_name, enable_rate_limit=enable_rl)

    fetched = fetch_ohlcv_incremental(
        ex,
        symbol=btc_symbol,
        timeframe=tf_signal,
        since_ms=since_ms,
        limit=500,
        max_batches=5,
        max_retries=5,
    )

    df15 = pd.concat([cached, fetched.df], ignore_index=True)
    df15["ts"] = pd.to_datetime(df15["ts"], utc=True)
    df15 = (
        df15.sort_values("ts")
        .drop_duplicates(subset=["ts"], keep="last")
        .reset_index(drop=True)
    )

    # Save updated cache (note: GitHub Actions FS is ephemeral; OK for MVP)
    save_cache_csv(cpath, df15)

    miss15 = detect_missing_candles(df15, tf_signal)

    # Build 1h and 4h from 15m (complete buckets only)
    df15, df1h, df4h = build_multitf(df15)

    # -----------------------
    # Phase 4: indicators + gates + signals (BTC only in scan_once MVP)
    # -----------------------
    ich_cfg = settings.get("ichimoku") or {}
    setup_cfg = settings.get("setup_a_plus") or {}
    dedup_cfg = settings.get("dedup") or {}

    p = IchimokuParams(
        tenkan=int(ich_cfg.get("tenkan", 9)),
        kijun=int(ich_cfg.get("kijun", 26)),
        spanb=int(ich_cfg.get("spanb", 52)),
        displacement=int(ich_cfg.get("displacement", 26)),
    )

    df15i = ichimoku(df15, p)
    df1hi = ichimoku(df1h, p)
    df4hi = ichimoku(df4h, p)

    slope_bars = int(setup_cfg.get("kijun_slope_bars", 10))
    reg_on = btc_regime_on(df4hi, slope_bars=slope_bars)
    tr_on = trend_on_1h(df1hi, slope_bars=slope_bars)

    cfg = APlusConfig(
        score_threshold=int(setup_cfg.get("score_threshold", 80)),
        retest_window_bars=int(setup_cfg.get("retest_window_bars", 12)),
        retest_eps=float(setup_cfg.get("retest_eps", 0.0015)),
        kijun_slope_bars=int(setup_cfg.get("kijun_slope_bars", 10)),
        max_kijun_distance_pct=float(setup_cfg.get("max_kijun_distance_pct", 0.008)),
        min_kumo_thickness_pct=float(setup_cfg.get("min_kumo_thickness_pct", 0.0035)),
    )

    # Current time = last closed 15m bar ts (since we only use closed bars)
    if not df15i.empty:
        now_ts = pd.to_datetime(df15i["ts"].iloc[-1], utc=True)
    else:
        now_ts = pd.Timestamp.utcnow().tz_localize("UTC")

    # -----------------------
    # Phase 6: persistent state (dedup across runs)
    # -----------------------
    db_state = load_state(conn, btc_symbol)
    if db_state:
        current_state = db_state["state"] or "FLAT"
        cooldown_until = (
            pd.to_datetime(db_state["cooldown_until"], utc=True)
            if db_state.get("cooldown_until")
            else None
        )
        last_exit_reason = db_state.get("last_exit_reason")
        last_signal_ts = db_state.get("last_signal_ts")
    else:
        current_state = "FLAT"
        cooldown_until = None
        last_exit_reason = None
        last_signal_ts = None

    in_cooldown_flag = cooldown_until is not None and now_ts < cooldown_until

    signals = []

    # EXIT check first
    exit_sig = a_plus_exit_signal(df15i)
    if current_state == "IN_SIGNALLED_LONG" and exit_sig is not None:
        payload = {"symbol": btc_symbol, **exit_sig}
        signals.append(payload)

        insert_signal(
            conn,
            ts=exit_sig["ts"],
            symbol=btc_symbol,
            tf=tf_signal,
            type_="EXIT",
            score=None,
            reason=exit_sig.get("reason"),
            payload=payload,
        )

        save_state(
            conn,
            symbol=btc_symbol,
            state="FLAT",
            cooldown_until=cooldown_until.isoformat() if cooldown_until else None,
            last_signal_ts=exit_sig["ts"],
            last_exit_reason=exit_sig.get("reason"),
        )

        current_state = "FLAT"
        last_exit_reason = exit_sig.get("reason")
        last_signal_ts = exit_sig["ts"]

    # ENTRY check
    if current_state == "FLAT" and (not in_cooldown_flag) and tr_on and reg_on:
        entry = a_plus_entry_signal(df15i, cfg)
        if entry is not None:
            payload = {"symbol": btc_symbol, **entry}
            signals.append(payload)

            cooldown_bars = int(dedup_cfg.get("cooldown_bars", 12))
            cooldown_delta = pd.Timedelta(minutes=15 * cooldown_bars)
            new_cooldown = now_ts + cooldown_delta

            insert_signal(
                conn,
                ts=entry["ts"],
                symbol=btc_symbol,
                tf=tf_signal,
                type_="ENTRY",
                score=entry.get("score"),
                reason=None,
                payload=payload,
            )

            save_state(
                conn,
                symbol=btc_symbol,
                state="IN_SIGNALLED_LONG",
                cooldown_until=new_cooldown.isoformat(),
                last_signal_ts=entry["ts"],
                last_exit_reason=None,
            )

            current_state = "IN_SIGNALLED_LONG"
            cooldown_until = new_cooldown
            last_signal_ts = entry["ts"]
            last_exit_reason = None

    # -----------------------
    # Phase 5: Telegram notifications (robust, non-blocking)
    # -----------------------
    tg_enabled = bool((settings.get("telegram") or {}).get("enabled", False))
    client = make_telegram_client_from_env() if tg_enabled else None

    sent = 0
    if client is not None and signals:
        for s in signals:
            try:
                if s.get("type") == "ENTRY":
                    ok = client.send_message(format_entry(s))
                    sent += 1 if ok else 0
                elif s.get("type") == "EXIT":
                    ok = client.send_message(format_exit(s))
                    sent += 1 if ok else 0
            except Exception:
                # never crash scan_once due to notifier
                pass

    health = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "exchange": exchange_name,
        "symbol": btc_symbol,
        "timeframes": {"signal": tf_signal, "trend": "1h", "regime": "4h"},
        "fetch": {
            "since_ms": since_ms,
            "fetched_rows": fetched.fetched_rows,
            "cache_rows_after": len(df15),
        },
        "missing_15m": {
            "expected": miss15.expected,
            "actual": miss15.actual,
            "missing": miss15.missing,
            "missing_pct": round(miss15.missing_pct, 6),
        },
        "freshness": {
            "15m": compute_freshness(df15, "15m"),
            "1h": compute_freshness(df1h, "1h"),
            "4h": compute_freshness(df4h, "4h"),
        },
        "gates": {"btc_regime_4h": reg_on, "trend_1h": tr_on},
        "signals": signals,
        "state": {
            "current_state": current_state,
            "cooldown_until": cooldown_until.isoformat() if cooldown_until is not None else None,
            "in_cooldown": in_cooldown_flag,
            "last_signal_ts": last_signal_ts,
            "last_exit_reason": last_exit_reason,
        },
        "telegram": {
            "enabled": tg_enabled,
            "configured": client is not None,
            "sent_messages": sent,
            "signals_count": len(signals),
        },
        "observability": {
            "scan_latency_ms": int((time.time() - t0) * 1000),
            "symbols_scanned": 1,
        },
        "notes": [
            "Phase2+4+5+6: fetched BTC 15m, resampled 1h/4h, computed Ichimoku, gates, A+ signals, Telegram notify, and persisted state/signals in SQLite.",
            "Next: scan all symbols + persist per-symbol state + add bars_meta for data quality if needed.",
        ],
    }

    print("[scan_once] data_health=" + json.dumps(json_safe(health), ensure_ascii=False))


if __name__ == "__main__":
    main()
