from __future__ import annotations

import os
from typing import Optional

from notifier.telegram import TelegramClient, TelegramConfig


def make_telegram_client_from_env() -> Optional[TelegramClient]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    cfg = TelegramConfig(token=token, chat_id=chat_id, timeout_sec=10, min_interval_sec=1.0)
    return TelegramClient(cfg)
