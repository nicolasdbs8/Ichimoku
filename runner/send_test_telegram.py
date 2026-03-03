from __future__ import annotations

from runner.notify import make_telegram_client_from_env


def main() -> None:
    client = make_telegram_client_from_env()
    if client is None:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in env.")

    ok = client.send_message("✅ Ichimoku bot: Telegram test message (Phase 5).")
    if not ok:
        raise RuntimeError("Telegram send failed (check token/chat_id).")

    print("[send_test_telegram] ok=true")


if __name__ == "__main__":
    main()
