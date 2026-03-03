from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Health:
    ts_utc: str
    status: str
    details: dict

    @staticmethod
    def ok(details: dict) -> "Health":
        return Health(
            ts_utc=datetime.now(timezone.utc).isoformat(),
            status="ok",
            details=details,
        )
