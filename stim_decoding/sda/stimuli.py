#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stimuli.py — build the stimulus time-courses that the EEG is regressed against.

  * Audio  : the broadband speech ENVELOPE (Hilbert magnitude, loudness-compressed).
  * Visual : a VISUAL EMBEDDING reconstructed from the deterministic Tetris game
             record — motion energy, luminance, line-clear impulses, and (shared)
             frame-PCA components. This is the visual analogue of the envelope.

Everything is produced at the model sample rate `fs`, band-limited to the EEG
band, and z-scored, so audio and visual streams are directly comparable.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import signal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tetris.tetris_game import TetrisGame  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase band-pass along axis 0."""
    ny = fs / 2.0
    b, a = signal.butter(4, [max(lo, 0.1) / ny, min(hi, ny * 0.99) / ny], "band")
    return signal.filtfilt(b, a, x, axis=0)


def _zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return (x - m) / np.where(s > 0, s, 1.0)


# --------------------------------------------------------------------------
# audio envelope
# --------------------------------------------------------------------------
def audio_envelope(wav_path: str, fs: float, n_samples: int, cfg: dict,
                   band=(1.0, 8.0)) -> np.ndarray:
    """Speech envelope at `fs`, band-limited + z-scored, length n_samples."""
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    env = np.abs(signal.hilbert(x)).astype(np.float64)
    p = float(cfg["stimuli"]["audio"].get("power_law", 0.3))
    if p and p > 0:
        env = np.power(np.maximum(env, 0), p)
    # resample to fs, then match the EEG band
    n_out = int(round(len(env) / sr * fs))
    env = signal.resample(env, n_out)
    env = _bandpass(env, fs, band[0], band[1])
    env = _fit_length(env, n_samples)
    return _zscore(env[:, None])[:, 0]


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), axis=0)])


# --------------------------------------------------------------------------
# F0 (pitch) contour — the cortical pitch-tracking analogue of the envelope
# --------------------------------------------------------------------------
def audio_f0_contour(wav_path: str, fs: float, n_samples: int, cfg: dict,
                     band=(1.0, 8.0)) -> np.ndarray:
    """
    Slowly-varying F0 (fundamental-frequency / pitch) contour at `fs`, band-limited
    + z-scored, length n_samples. The raw voiced-log-F0 track (from librosa.pyin) is
    cached to disk keyed by clip name (pyin is expensive), then interpolated across
    unvoiced gaps, resampled to `fs`, and matched to the EEG band so it is directly
    comparable to `audio_envelope`. This is the pitch-contour descriptor the cortex
    tracks in the delta-theta band, distinct from the amplitude envelope.
    """
    lf0, fr = _load_f0_track(wav_path, cfg)
    if lf0 is None:
        return np.zeros(n_samples)
    n_out = int(round(len(lf0) / fr * fs))
    lf0 = signal.resample(lf0, n_out)
    lf0 = _bandpass(lf0, fs, band[0], band[1])
    lf0 = _fit_length(lf0, n_samples)
    return _zscore(lf0[:, None])[:, 0]


def _load_f0_track(wav_path: str, cfg: dict):
    """(interpolated log-F0 at pyin frame-rate, frame_rate_hz) with a disk cache."""
    import soundfile as sf
    cache_dir = cfg.get("stimuli", {}).get("f0", {}).get(
        "cache_dir", os.path.join(ROOT, "records", "derivatives", "cca_decoding",
                                  "f0_cache"))
    cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(ROOT, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    key = os.path.splitext(os.path.basename(wav_path))[0]
    cache = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["lf0"], float(d["fr"])

    import librosa
    fcfg = cfg.get("stimuli", {}).get("f0", {})
    fmin = float(fcfg.get("fmin_hz", 75.0))
    fmax = float(fcfg.get("fmax_hz", 400.0))
    hop = int(fcfg.get("hop", 256))
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f0, _, _ = librosa.pyin(x, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop)
    lf0 = np.log(f0)
    idx = np.arange(len(lf0))
    good = np.isfinite(lf0)
    if good.sum() < 8:
        np.savez(cache, lf0=np.zeros(1), fr=float(sr / hop))
        return None, float(sr / hop)
    lf0 = np.interp(idx, idx[good], lf0[good])       # bridge unvoiced gaps
    fr = float(sr / hop)
    np.savez(cache, lf0=lf0.astype(np.float32), fr=fr)
    return lf0, fr


# --------------------------------------------------------------------------
# visual embedding (from the Tetris reconstruction record)
# --------------------------------------------------------------------------
def _reconstruct_frames(record: dict, fs: float, n_samples: int, cell_px: int,
                        down):
    """Deterministically replay the game at `fs` and return per-frame features.

    Returns (motion, luminance, clear_impulse, frames_ds) where frames_ds is
    (n_samples, down_r*down_c) downsampled grayscale for the shared PCA.
    """
    import cv2
    if "inputs" in record or str(record.get("engine", "")).startswith("tetris_game"):
        g = TetrisGame.from_record(record)             # replay the player's exact game
    else:                                              # legacy self-playing record
        g = TetrisGame(cols=int(record["cols"]), rows=int(record["rows"]),
                       seed=int(record["seed"]), mode="ai", on_top_out="reset")
    dt = 1.0 / fs
    dr, dc = int(down[0]), int(down[1])
    motion = np.zeros(n_samples)
    lum = np.zeros(n_samples)
    clears = np.zeros(n_samples)
    frames = np.zeros((n_samples, dr * dc), dtype=np.float32)
    prev_small = None
    prev_events = 0
    for i in range(n_samples):
        g.update(dt)
        img = g.to_image(cell_px)                      # (H, W) float [-1, 1]
        small = cv2.resize(img, (dc, dr), interpolation=cv2.INTER_AREA)
        lum[i] = float(small.mean())
        if prev_small is not None:
            motion[i] = float(np.abs(small - prev_small).mean())
        if g.line_clear_events > prev_events:          # a clear just started
            clears[i] = 1.0
            prev_events = g.line_clear_events
        frames[i] = small.reshape(-1)
        prev_small = small
    return motion, lum, clears, frames


def build_visual_embeddings(records: dict, fs: float, n_samples: int,
                            cfg: dict, band=(1.0, 8.0)) -> dict:
    """
    Build a comparable visual embedding for every trial. PCA (if requested) uses
    a SINGLE basis fit across all trials' frames, so components mean the same
    thing in every trial (required for a pooled encoder). No attention labels are
    involved, so this is leakage-free w.r.t. the decoding target.

    Returns {trial: embedding (n_samples, n_features)} and the feature names.
    """
    vc = cfg["stimuli"]["visual"]
    cell_px = int(vc.get("cell_px", 6))
    down = vc.get("downsample", [12, 6])
    feats = vc.get("features", ["motion", "luminance", "clear_impulse", "pca"])
    k = int(vc.get("pca_components", 5))

    raw = {}     # trial -> (motion, lum, clears, frames)
    for tr, rec in records.items():
        raw[tr] = _reconstruct_frames(rec, fs, n_samples, cell_px, down)

    # shared PCA basis across all trials' frames
    pca = None
    if "pca" in feats and k > 0:
        from sklearn.decomposition import PCA
        allframes = np.nan_to_num(np.concatenate([raw[tr][3] for tr in raw], axis=0))
        pca = PCA(n_components=k, random_state=0).fit(allframes)

    names, out = [], {}
    for tr, (motion, lum, clears, frames) in raw.items():
        cols = []
        if "motion" in feats:
            cols.append(motion[:, None])
        if "luminance" in feats:
            cols.append(lum[:, None])
        if "clear_impulse" in feats:
            cols.append(clears[:, None])
        if pca is not None:
            cols.append(pca.transform(np.nan_to_num(frames)))
        emb = np.concatenate(cols, axis=1)
        # band-limit the continuous channels to the EEG band; keep the impulse
        # column as an event train (band-pass then z-score is fine for it too).
        emb = _bandpass(emb, fs, band[0], band[1])
        out[tr] = np.nan_to_num(_zscore(emb))
    if "motion" in feats:
        names.append("motion")
    if "luminance" in feats:
        names.append("luminance")
    if "clear_impulse" in feats:
        names.append("clear_impulse")
    if pca is not None:
        names += [f"pca{i+1}" for i in range(k)]
    return out, names
