from __future__ import annotations

from pathlib import Path
import pandas as pd


def cache_path(repo_root: Path, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return repo_root / "data" / "cache" / f"{safe_symbol}__{timeframe}.csv"


def ensure_cache_dir(repo_root: Path) -> None:
    (repo_root / "data" / "cache").mkdir(parents=True, exist_ok=True)


def load_cache_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return df


def save_cache_csv(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out.to_csv(path, index=False)
