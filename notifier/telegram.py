from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str
    timeout_sec: int = 10
    min_interval_sec: float = 1.0  # rate limit client-side


class TelegramClient:
    def __init__(self, cfg: TelegramConfig):
        self.cfg = cfg
        self._last_send_ts: float = 0.0

    def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_send_ts
        if elapsed < self.cfg.min_interval_sec:
            time.sleep(self.cfg.min_interval_sec - elapsed)

    def send_message(self, text: str, disable_web_preview: bool = True) -> bool:
        """
        Returns True if sent, False otherwise.
        Never logs token.
        """
        self._rate_limit()

        url = f"https://api.telegram.org/bot{self.cfg.token}/sendMessage"
        payload = {
            "chat_id": self.cfg.chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_preview,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.cfg.timeout_sec)
            self._last_send_ts = time.time()
            if resp.status_code != 200:
                return False
            data = resp.json()
            return bool(data.get("ok"))
        except Exception:
            # swallow network errors: bot must not crash
            return False
