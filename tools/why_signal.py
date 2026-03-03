from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from storage.db import connect, init_schema


def _pretty(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def load_signal(conn, signal_id: Optional[int]) -> Dict[str, Any]:
    if signal_id is None:
        cur = conn.execute(
            """
            SELECT id, ts, symbol, tf, type, score, reason, payload_json
            FROM signal_events
            ORDER BY id DESC
            LIMIT 1
            """
        )
    else:
        cur = conn.execute(
            """
            SELECT id, ts, symbol, tf, type, score, reason, payload_json
            FROM signal_events
            WHERE id=?
            """,
            (signal_id,),
        )

    row = cur.fetchone()
    if not row:
        raise RuntimeError("No signal found (db empty or id not found).")

    payload = json.loads(row[7])
    return {
        "id": row[0],
        "ts": row[1],
        "symbol": row[2],
        "tf": row[3],
        "type": row[4],
        "score": row[5],
        "reason": row[6],
        "payload": payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain why a signal was emitted (Phase 6 tool).")
    ap.add_argument("--id", type=int, default=None, help="signal_events.id (default: latest)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    conn = connect(repo_root)
    init_schema(conn)

    s = load_signal(conn, args.id)

    print("=== SIGNAL ===")
    print(f"id:     {s['id']}")
    print(f"ts:     {s['ts']}")
    print(f"symbol: {s['symbol']}")
    print(f"tf:     {s['tf']}")
    print(f"type:   {s['type']}")
    print(f"score:  {_pretty(s['score'])}")
    print(f"reason: {_pretty(s['reason'])}")

    payload = s["payload"]

    # ENTRY payload: expect score_detail with points/metrics
    score_detail = payload.get("score_detail") or {}
    points = (score_detail.get("points") or {}) if isinstance(score_detail, dict) else {}
    metrics = (score_detail.get("metrics") or {}) if isinstance(score_detail, dict) else {}

    if points:
        print("\n=== SCORE BREAKDOWN ===")
        total = 0
        for k in sorted(points.keys()):
            v = points[k]
            total += int(v) if isinstance(v, int) else 0
            print(f"{k:22s}: {v}")
        print(f"{'TOTAL':22s}: {total}")

    if metrics:
        print("\n=== METRICS ===")
        for k in sorted(metrics.keys()):
            print(f"{k:22s}: {_pretty(metrics[k])}")

    # Gates / extra context may exist
    if "setup" in payload:
        print("\n=== CONTEXT ===")
        print(f"setup: {_pretty(payload.get('setup'))}")

    if not points and s["type"] == "ENTRY":
        print("\nNote: No score_detail.points in payload. (Old schema or payload missing?)")


if __name__ == "__main__":
    main()
