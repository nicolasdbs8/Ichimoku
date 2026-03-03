from __future__ import annotations

from typing import Dict, Any


def tradingview_link(symbol: str) -> str:
    # crude mapping: "BTC/USDT" -> "BTCUSDT"
    tv = symbol.replace("/", "").replace(":", "")
    return f"https://www.tradingview.com/chart/?symbol={tv}"


def format_entry(payload: Dict[str, Any]) -> str:
    symbol = payload["symbol"]
    ts = payload.get("ts")
    score = payload.get("score")
    link = tradingview_link(symbol)

    return (
        "🟢 ICHIMOKU A+ LONG\n"
        f"Symbol: {symbol}\n"
        f"Time: {ts}\n"
        f"Score: {score}/100\n"
        f"Chart: {link}\n"
    )


def format_exit(payload: Dict[str, Any]) -> str:
    symbol = payload["symbol"]
    ts = payload.get("ts")
    reason = payload.get("reason")
    link = tradingview_link(symbol)

    return (
        "🔴 EXIT SIGNAL\n"
        f"Symbol: {symbol}\n"
        f"Time: {ts}\n"
        f"Reason: {reason}\n"
        f"Chart: {link}\n"
    )


def format_regime(payload: Dict[str, Any]) -> str:
    # optional message if you later emit regime flips
    ts = payload.get("ts")
    on = payload.get("btc_regime_4h")
    return (
        "⚠️ BTC REGIME UPDATE\n"
        f"Time: {ts}\n"
        f"BTC Regime 4h: {'ON' if on else 'OFF'}\n"
    )
