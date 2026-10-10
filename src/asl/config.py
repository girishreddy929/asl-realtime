"""Config loading helpers.

All thresholds and sizes live in configs/*.yaml. Code reads them through this module so a
missing or misspelled key fails loudly instead of silently using a wrong default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

ENV_ROOT = "ASL_PROJECT_ROOT"
_MISSING: Any = object()  # sentinel: "no default given" differs from "default=None"


def find_project_root() -> Path:
    """Return the repo root (folder with pyproject.toml and configs/).

    Order: ASL_PROJECT_ROOT env var, then walk up from this file, then from the cwd.
    Why not only the cwd: tests and scripts may be started from other folders.
    """
    env = os.environ.get(ENV_ROOT)
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{ENV_ROOT} points to a missing folder: {root}")
        return root
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (start, *start.parents):
            if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
                return parent
    raise FileNotFoundError(
        f"Project root not found. Run from the repo or set {ENV_ROOT}."
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping (an empty file gives {})."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data).__name__}: {p}")
    return data


def load_config(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load configs/<name>.yaml (for example name='runtime')."""
    directory = Path(config_dir) if config_dir else find_project_root() / "configs"
    return load_yaml(directory / f"{name}.yaml")


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path (relative to the repo root unless already absolute)."""
    p = Path(path)
    return p if p.is_absolute() else find_project_root() / p


def cfg_get(cfg: Mapping[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Read a nested value with a dotted key, e.g. cfg_get(cfg, 'camera.width').

    Raises KeyError naming the missing part unless a default is given.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif default is not _MISSING:
            return default
        else:
            raise KeyError(f"Missing config key '{dotted_key}' (failed at '{part}')")
    return node