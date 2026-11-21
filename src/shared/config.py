"""Reading YAML config files.

Every experiment (one row of the results table) has its own YAML file in
configs/. It can say ``base: shared.yaml`` to inherit the shared settings and
only override what changes. Nothing should be hard-coded in notebooks.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, following its ``base:`` chain."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / "configs" / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(base, cfg)
    return cfg


def repo_path(relative: str | Path) -> Path:
    """Resolve a path from the config relative to the repo root."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_run_info(out_dir: str | Path, cfg: dict, seed: int | None, extra: dict | None = None) -> None:
    """Write run_info.json so every number can be traced to its config and commit."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": seed,
        "config": cfg,
    }
    info.update(extra or {})
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
