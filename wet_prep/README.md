# Session-1 wet-EEG preprocessing (auditory oddball)

Preprocessing + analysis of `sub-P001_ses-S001_task-Default_run-001_eeg_old1.xdf`
(CGX Quick-32r, wet electrodes, `TrueDetective` auditory-oddball paradigm), following
the Tom/Arsalan benchmark notebook, adapted to this recording. One self-contained
script: `preprocess.py`.

## Pipeline — a figure at every level
`figures/L0…L4_*.png` — PSD + channel-variance topography + example traces after each:

| level | step |
|---|---|
| 0 | load XDF → 29 scalp channels (10-20) → montage |
| 1 | average reference |
| 2 | ICA (FastICA) + **ICLabel** auto-rejection of eye/heart/muscle ICs (> 0.8) — removed 1 eye-blink IC |
| 3 | band-pass 1–40 Hz |
| 4 | 60 Hz line notch |

**Filtering note:** the notebook filters *causally* (`phase='forward'`, for real-time
parity). On this drifty wet recording that leaves ~26× excess low-frequency drift
(320 vs 12 µV, verified), so this pipeline uses **zero-phase** filtering — the
offline-analysis standard — which yields clean ~12 µV EEG and undistorted ERPs. Switch
back to causal only if you need real-time parity.

## Comparative & analysis figures
- `compare_PSD_stages.png` — PSD overlaid across all levels (avg-ref → ICA → band-pass roll-off → 60 Hz notch).
- `analysis_oddball_MMN.png` — standard (n=160) vs deviant (n=20) ERP at Fz/Cz/Fpz → **mismatch negativity** (~120–220 ms) + P3a, with MMN topography.
- `analysis_resting_alpha.png` — eyes-closed vs eyes-open PSD + alpha topography (Berger effect).

## Saved data (`preprocessed/`)
- `sub-P001_ses-S001_cleaned_raw.fif` — cleaned continuous, 256 Hz.
- `*_sub01_v1_cgx_ica.mat` — segments: `eyes_closed`, `eyes_open`, `oddball_standard`,
  `oddball_deviant` (`{data, info}`; epochs are [channels, time, trials]).
- `provenance.json` — channels, ICA labels/exclusions, epoch counts, segment windows.

## Marker interpretation (from the timeline; `TrueDetective_Markers`)
`2` = standard tone (160×), `1` = deviant (20×), `3` = third event (20×), `11` = block
onset, resting = pre-tone period split eyes-closed/open at code `12`. The oddball
frequent/rare split is unambiguous; the **resting eyes-closed/open brackets are inferred**
(no marker spec available) — confirm the codes to finalize the resting analysis.

Run: `python wet_prep/preprocess.py`
