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
    games_dir: str
    manifest: str

    @classmethod
    def from_config(cls, cfg: dict) -> "Paths":
        p = cfg["paths"]
        stim_dir = _abs(p["stim_dir"])
        return cls(
            stim_dir=stim_dir,
            audio_dir=os.path.join(stim_dir, p["audio_subdir"]),
            behavior_dir=_abs(p["behavior_dir"]),
            games_dir=_abs(p.get("games_dir", "games")),
            manifest=_abs(p["manifest"]),
        )

    def ensure(self) -> "Paths":
        """Create the directories if they don't yet exist."""
        for d in (self.stim_dir, self.audio_dir, self.behavior_dir, self.games_dir):
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


# --------------------------------------------------------------------------
# Probe word bank (shared by build_stimuli.py and run_block.py)
# --------------------------------------------------------------------------
# Common, concrete content words used as FALSE probe targets (words that are
# plausibly speakable but chosen to be ABSENT from a given transcript).
COMMON_WORDS = [
    "anchor", "purple", "engine", "harbor", "candle", "ladder", "silver",
    "meadow", "rocket", "pencil", "garden", "bottle", "thunder", "marble",
    "velvet", "copper", "lantern", "pillow", "saddle", "feather", "orchard",
    "cabinet", "blanket", "compass", "diamond", "kettle", "mirror", "ribbon",
    "tunnel", "wagon", "whistle", "balloon", "biscuit", "cottage", "dolphin",
    "glacier", "hammer", "jacket", "kingdom", "magnet",
]

# Function words excluded when picking TRUE probe targets from a transcript.
STOPWORDS = {
    "the", "and", "that", "with", "this", "from", "they", "have", "were",
    "their", "what", "when", "your", "which", "them", "then", "than", "into",
    "been", "more", "some", "such", "only", "would", "could", "should", "about",
    "there", "these", "those", "here", "very", "just", "also", "upon", "shall",
    "will", "him", "her", "his", "its", "our", "are", "was", "for", "not",
    "but", "you", "had", "has",
}


def content_words(transcript: str, min_len: int) -> list[str]:
    """Lower-cased content words from a transcript suitable as probe targets."""
    toks = [t.strip(".,!?;:\"'()").lower() for t in transcript.split()]
    return [t for t in toks
            if t.isalpha() and len(t) >= min_len
            and t not in STOPWORDS and t != "unk"]


def make_trial_audio_probe(words: list[dict], transcript: str, min_len: int,
                           rng, window_end_s: float | None = None) -> dict | None:
    """
    Build ONE yes/no 'was WORD spoken?' probe for a trial, correct answer
    balanced ~50/50. When `window_end_s` is given, TRUE targets are drawn only
    from words actually heard within that window (so the probe is valid even
    when only the first N seconds of a longer clip are played).
    """
    heard = [(w.get("word") or "").strip(".,!?;:\"'()").lower()
             for w in words
             if not w.get("is_unk")
             and (window_end_s is None or float(w.get("end_s", 0)) <= window_end_s)]
    present_pool = sorted({w for w in heard
                           if w.isalpha() and len(w) >= min_len
                           and w not in STOPWORDS})
    transcript_set = set(content_words(transcript, 1))
    absent_pool = [w for w in COMMON_WORDS
                   if w not in transcript_set and len(w) >= min_len]

    want_present = rng.random() < 0.5
    if want_present and present_pool:
        tw = rng.choice(present_pool)
        return {"target_word": tw, "present": True,
                "question": f'Was the word "{tw}" spoken?'}
    if absent_pool:
        fw = rng.choice(absent_pool)
        return {"target_word": fw, "present": False,
                "question": f'Was the word "{fw}" spoken?'}
    if present_pool:                       # fallback if no absent word available
        tw = rng.choice(present_pool)
        return {"target_word": tw, "present": True,
                "question": f'Was the word "{tw}" spoken?'}
    return None
