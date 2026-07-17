"""Cut labelled epochs time-locked to gap onset (instruction offset).

Timeline (t = 0 at gap onset):
    [-1.5, 0)   instruction/cue on screen ("attend to AUDIO / VISUAL")
    [0, +gap)   anticipatory gap — screen identical across conditions
    +gap        audiovisual stimulus onset  (>= 1.80 s for every usable trial)

``tmax`` is kept below the shortest gap so a stimulus-evoked response can never
leak into the "anticipatory" epoch. Downsampling is an integer ``decim`` (the
data is already low-passed, so no aliasing).
"""
from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from .utils import LOG


def make_epochs(raw: mne.io.BaseRaw, events: pd.DataFrame, cfg: dict,
                resample_hz: float | None = None) -> mne.Epochs:
    event_id = cfg["event_id"]
    codes = events.label.map(event_id).values
    mne_events = np.column_stack([
        events.gap_sample.values.astype(int),
        np.zeros(len(events), dtype=int),
        codes.astype(int),
    ])

    sfreq = raw.info["sfreq"]
    decim = 1
    if resample_hz and resample_hz < sfreq:
        decim = int(round(sfreq / resample_hz))

    reject = None
    if cfg.get("reject_uv"):
        reject = dict(eeg=float(cfg["reject_uv"]) * 1e-6)

    baseline = tuple(cfg["baseline"]) if cfg.get("baseline") else None
    epochs = mne.Epochs(
        raw, mne_events, event_id=event_id,
        tmin=cfg["tmin"], tmax=cfg["tmax"], baseline=baseline,
        reject=reject, decim=decim, preload=True,
        picks="eeg", reject_by_annotation=False, verbose="ERROR",
    )
    # Carry per-trial info through for later inspection.
    meta = events.reset_index(drop=True).copy()
    epochs.metadata = meta.iloc[epochs.selection].reset_index(drop=True)

    n_drop = len(events) - len(epochs)
    LOG.info("Epochs: %d kept (%d dropped by p2p) @ %.0f Hz | Audio %d / Visual %d",
             len(epochs), n_drop, epochs.info["sfreq"],
             len(epochs["Audio"]), len(epochs["Visual"]))
    return epochs
