from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import json


DB_FILENAME = "bot_state.db"


def db_path(repo_root: Path) -> Path:
    return repo_root / DB_FILENAME


def connect(repo_root: Path) -> sqlite3.Connection:
    path = db_path(repo_root)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            tf TEXT NOT NULL,
            type TEXT NOT NULL,
            score INTEGER,
            reason TEXT,
            payload_json TEXT NOT NULL
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            symbol TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            cooldown_until TEXT,
            last_signal_ts TEXT,
            last_exit_reason TEXT
        );
        """
    )

    conn.commit()


# -------------------------
# Signal persistence
# -------------------------

def insert_signal(
    conn: sqlite3.Connection,
    ts: str,
    symbol: str,
    tf: str,
    type_: str,
    score: Optional[int],
    reason: Optional[str],
    payload: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO signal_events (ts, symbol, tf, type, score, reason, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            symbol,
            tf,
            type_,
            score,
            reason,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    conn.commit()


# -------------------------
# State persistence
# -------------------------

def load_state(conn: sqlite3.Connection, symbol: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT state, cooldown_until, last_signal_ts, last_exit_reason FROM state WHERE symbol=?",
        (symbol,),
    )
    row = cur.fetchone()
    if not row:
        return None

    return {
        "state": row[0],
        "cooldown_until": row[1],
        "last_signal_ts": row[2],
        "last_exit_reason": row[3],
    }


def save_state(
    conn: sqlite3.Connection,
    symbol: str,
    state: str,
    cooldown_until: Optional[str],
    last_signal_ts: Optional[str],
    last_exit_reason: Optional[str],
) -> None:
    conn.execute(
        """
        INSERT INTO state (symbol, state, cooldown_until, last_signal_ts, last_exit_reason)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            state=excluded.state,
            cooldown_until=excluded.cooldown_until,
            last_signal_ts=excluded.last_signal_ts,
            last_exit_reason=excluded.last_exit_reason
        """,
        (
            symbol,
            state,
            cooldown_until,
            last_signal_ts,
            last_exit_reason,
        ),
    )
    conn.commit()
