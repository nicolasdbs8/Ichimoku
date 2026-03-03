from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import yaml


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}: expected mapping at root")
    return data


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    settings_path = repo_root / "config" / "settings.yaml"
    universe_path = repo_root / "config" / "universe.yaml"

    settings = load_yaml(settings_path)
    universe = load_yaml(universe_path)

    symbols = universe.get("symbols", [])
    enabled = [s for s in symbols if isinstance(s, dict) and s.get("enabled", True)]
    exchange = (settings.get("exchange") or {}).get("name", "unknown")

    health = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "exchange": exchange,
        "env": settings.get("env", "unknown"),
        "timeframes": settings.get("timeframes", {}),
        "symbols_total": len(symbols),
        "symbols_enabled": len(enabled),
        "notes": [
            "Phase1 smoke scan only (no data fetch yet).",
            "Next: Phase2 data fetch + cache + resample.",
        ],
    }

    print("[scan_once] health=" + json.dumps(health, ensure_ascii=False))


if __name__ == "__main__":
    main()
