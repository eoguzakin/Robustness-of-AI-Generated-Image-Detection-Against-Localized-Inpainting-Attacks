from __future__ import annotations
from pathlib import Path
import os
import yaml

def _expand(x):
    if isinstance(x, str):
        return os.path.expandvars(os.path.expanduser(x))
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    return _expand(yaml.safe_load(p.read_text(encoding="utf-8")))
