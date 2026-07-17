"""Shared helpers: logging, config loading, path resolution."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import yaml

LOG = logging.getLogger("cma_eeg")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a single console logger for the whole pipeline."""
    if not LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
        LOG.addHandler(h)
    LOG.setLevel(level)
    LOG.propagate = False
    return LOG


def load_config(path: str) -> dict[str, Any]:
    """Read the YAML config file into a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base`` (used for CLI overrides)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def banner(msg: str) -> None:
    """Log a visually distinct stage header."""
    LOG.info("=" * 70)
    LOG.info(msg)
    LOG.info("=" * 70)


@dataclass
class Paths:
    """Resolved output locations for one subject."""
    root: str
    subject: str

    @property
    def subject_dir(self) -> str:
        return ensure_dir(os.path.join(self.root, self.subject))

    @property
    def fig_dir(self) -> str:
        return ensure_dir(os.path.join(self.subject_dir, "figures"))

    def file(self, name: str) -> str:
        return os.path.join(self.subject_dir, f"{self.subject}_{name}")

    def fig(self, name: str) -> str:
        return os.path.join(self.fig_dir, f"{self.subject}_{name}")
