#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labram.py — use the LaBraM EEG foundation model (Jiang et al., ICLR 2024) as a
FROZEN feature extractor for the records cross-modal attention data.

We deliberately do NOT fine-tune: with a single subject and 60 trials, fine-tuning
even the 5.8M-parameter base model would overfit. Instead we take the pretrained
model's per-window [CLS] embedding and train a small, regularised linear probe on
top (see labram_attention.py). The foundation model is the same for every fold, so
it can introduce no label leakage.

Three things have to be exactly right for the pretrained weights to be meaningful:

  1. INPUT FORMAT. LaBraM was pretrained on EEG resampled to 200 Hz, broadband
     (~0.1-75 Hz) with the line notched, in units of 100 µV (i.e. µV / 100). We
     therefore build a SEPARATE preprocessing branch from the 1-8 Hz CCA pipeline
     (a narrow band would delete the alpha/beta activity that carries most of the
     audio-vs-visual attention effect).
  2. WINDOW LENGTH. braindecode's Labram derives its patch count from the model's
     `n_times`; the pretrained checkpoint has 15 one-second patches, so windows
     must be exactly 3000 samples (15 s) at 200 Hz. That is also the model's
     maximum (temporal_embedding has 16 slots).
  3. CHANNEL IDENTITY. LaBraM keys its learned spatial embeddings off standard
     10-20 names, so we hand it our (montage-renamed) channel names and it selects
     the matching position embeddings. Names outside its 128-channel vocabulary
     are dropped.
"""

from __future__ import annotations

import copy

import numpy as np

FS = 200                    # LaBraM sample rate (Hz)
PATCH = 200                 # one-second patch
WIN_SAMPLES = 3000          # 15 s = 15 patches = the pretrained checkpoint's n_times
LABRAM_HF = "braindecode/labram-pretrained"
UV_PER_UNIT = 100.0         # LaBraM input unit = 100 µV -> feed volts * 1e6 / 100


# --------------------------------------------------------------------------
# Preprocessing (broadband, 200 Hz) — reuses the project's montage / ICA / ref
# --------------------------------------------------------------------------
def preprocess_for_labram(raw, cfg, l_freq=0.3, h_freq=75.0, notch=50.0):
    """Clean the continuous raw into LaBraM's expected space.

    Reuses `preprocess.preprocess_continuous` (physical-number -> 10-20 rename via
    the ground-truth .bvef, session-level bad-channel interpolation, ocular/muscle
    ICA, Fz recovery + average reference) but overrides the band and sample rate to
    LaBraM's broadband/200 Hz, and notches the mains first. Returns (raw_model, prov).
    """
    from . import preprocess

    if notch:                                             # remove line noise before ICA
        raw.notch_filter(notch, verbose="ERROR")
    lab_cfg = copy.deepcopy(cfg)
    pc = lab_cfg["preprocess"]
    pc["l_freq"], pc["h_freq"], pc["resample_hz"] = l_freq, h_freq, float(FS)
    raw_model, prov = preprocess.preprocess_continuous(raw, lab_cfg)
    prov["labram_band"] = [l_freq, h_freq]
    prov["labram_notch"] = notch
    return raw_model, prov


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_labram(device="cpu"):
    """The pretrained braindecode LaBraM, frozen and in eval mode."""
    import torch  # noqa: F401
    from braindecode.models import Labram
    model = Labram.from_pretrained(LABRAM_HF).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def canonical_channels(ch_names):
    """(kept_names, keep_idx) for channels LaBraM knows (case-insensitive match to
    its 128-name vocabulary). Anything else (e.g. a non-10-20 label) is dropped."""
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER
    vocab = {c.upper() for c in LABRAM_CHANNEL_ORDER}
    keep = [(i, n) for i, n in enumerate(ch_names) if n.upper() in vocab]
    return [n for _, n in keep], [i for i, _ in keep]


def window_starts(n_times, win=WIN_SAMPLES, n_win=3):
    """Start samples for `n_win` (overlapping) windows spanning [0, n_times)."""
    if n_times <= win:
        return [0]
    last = n_times - win
    return sorted(set(int(round(s)) for s in np.linspace(0, last, n_win)))


def embed_trials(model, trials, ch_names, device="cpu", win=WIN_SAMPLES, n_win=3,
                 batch=16, scale_uv=True):
    """Frozen-LaBraM embeddings for every window of every trial.

    Parameters
    ----------
    model     : the frozen Labram.
    trials    : list of (n_ch, n_times) arrays at 200 Hz, IN VOLTS, channel order
                matching `ch_names`.
    ch_names  : the channel names (10-20) for the array's channel axis.

    Returns
    -------
    Zcls  : (n_windows, 200) the [CLS] token embedding.
    Zmean : (n_windows, 200) MEAN over the patch tokens (channels x time). Mean
            pooling exposes the per-patch spectral content better than [CLS] for a
            frozen linear probe, so it is the primary feature downstream.
    tid   : (n_windows,) source-trial index for each window (for group CV).
    """
    import torch
    keep_names, keep_idx = canonical_channels(ch_names)
    if len(keep_idx) < len(ch_names):
        dropped = sorted(set(ch_names) - set(keep_names))
        print(f"  LaBraM: dropping {len(dropped)} non-canonical channel(s): {dropped}")

    xs, tid = [], []
    for t, X in enumerate(trials):
        Xk = np.asarray(X, np.float32)[keep_idx]           # (n_keep, n_times)
        if scale_uv:
            Xk = Xk * 1e6 / UV_PER_UNIT                     # volts -> µV/100 (LaBraM units)
        for s in window_starts(Xk.shape[1], win, n_win):
            xs.append(Xk[:, s:s + win]); tid.append(t)

    Zcls = np.zeros((len(xs), model.embed_dim), np.float32)
    Zmean = np.zeros((len(xs), model.embed_dim), np.float32)
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = torch.from_numpy(np.stack(xs[i:i + batch])).to(device)
            out = model(xb, return_features=True, ch_names=keep_names)
            Zcls[i:i + batch] = out["cls_token"].float().cpu().numpy()
            Zmean[i:i + batch] = out["features"].mean(1).float().cpu().numpy()
    return Zcls, Zmean, np.asarray(tid, int)
