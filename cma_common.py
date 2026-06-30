#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cma_common.py — shared utilities for the cross-modal attention experiment.

Used by both build_stimuli.py (library preparation) and run_block.py
(experiment runner): configuration loading, path resolution, manifest I/O,
timestamped logging, and a couple of small audio helpers.

Deliberately lightweight — importing this module must NOT pull in PsychoPy,
PyAV, faster-whisper, or any heavy stack, so it is safe to import everywhere.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass

import yaml

# All paths are resolved relative to the project root (this file's directory).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.yaml")


def log(msg: str) -> None:
    """Timestamped progress logger shared by both scripts."""
    stamp = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load the YAML configuration file into a plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    """Resolved absolute paths derived from the config's `paths:` block."""
    stim_dir: str
    audio_dir: str
    behavior_dir: str
    manifest: str

    @classmethod
    def from_config(cls, cfg: dict) -> "Paths":
        p = cfg["paths"]
        stim_dir = _abs(p["stim_dir"])
        return cls(
            stim_dir=stim_dir,
            audio_dir=os.path.join(stim_dir, p["audio_subdir"]),
            behavior_dir=_abs(p["behavior_dir"]),
            manifest=_abs(p["manifest"]),
        )

    def ensure(self) -> "Paths":
        """Create the directories if they don't yet exist."""
        for d in (self.stim_dir, self.audio_dir, self.behavior_dir):
            os.makedirs(d, exist_ok=True)
        return self


def _abs(path: str) -> str:
    """Resolve a possibly-relative config path against the project root."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def abspath(path: str) -> str:
    """Public alias for resolving config-relative paths."""
    return _abs(path)


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------
def read_manifest(path: str) -> dict:
    """Load the stimulus manifest, or return an empty skeleton if absent."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"created": None, "audio": []}


def write_manifest(path: str, manifest: dict) -> None:
    """Persist the manifest as pretty JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest.setdefault("created", _dt.datetime.now().isoformat(timespec="seconds"))
    manifest["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# Small audio helper
# --------------------------------------------------------------------------
def audio_duration_seconds(wav_path: str) -> float:
    """Duration of a WAV in seconds (via soundfile; no heavy decoders)."""
    import soundfile as sf
    info = sf.info(wav_path)
    return info.frames / float(info.samplerate)
