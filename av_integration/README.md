# AV-integration consolidated pipeline

End-to-end cross-modal (**Audio-vs-Visual attention**) decoding for the AudVis EEG,
built to be fully transparent: **a diagnostic figure is written at every step**, and
every number lands in a JSON. One command:

```bash
python av_integration/pipeline.py            # --no-star / --no-ica / --n-perm N / --fs
```

Outputs → `av_integration/derivatives/<subject>/`:
`figures/stepNN_*.png` (one per step) and `<subject>_av_integration.json`.

## Corrected channel order (critical)

The two coloured 32-channel cables were plugged in **reversed** for this recording
(assumed 1:32=green/33:64=yellow; actually 1:32=yellow/33:64=green), so a recorded
channel *n* sits at electrode *n±32*. This is **confirmed empirically** — spatial
smoothness (neighbour correlation of volume-conducted EEG) is **0.70 swapped vs 0.44
assumed** (`stim_decoding/channel_swap_validate.py`). Enabled via
`preprocess.cable_halves_swapped: true`. Consequence: **Fz is a normal data channel**
and the online reference (the absent recorded channel "2") is **AF3**, not Fz — the
"Fz reference" was itself an artefact of the wrong mapping. Average-referencing makes
this moot; AF3 is recovered as −average.

## Preprocessing (a plot per step)

| step | what | figure |
|---|---|---|
| 0 | load + **corrected (swapped) montage** | `step00_montage`, `step01_raw` |
| 1 | notch (50 Hz + harmonics) | `step02_notch` |
| 2 | band-pass 1–45 Hz + resample 200 Hz | `step03_bandpass_resample` |
| 3 | robust bad channels (low spatial coherence / railing, on **avg-ref** data so the frontal AF3 reference doesn't cause false flags) | `step04_bad_channels` |
| 4 | **ICA** (extended-infomax; ocular + muscle removed) — *before* STAR so it isn't swamped by blinks | `step*_ICA_*` |
| 5 | **STAR** — Sparse Time-Artifact Removal (meegkit / de Cheveigné): repairs outlier samples of a channel from its spatial neighbours | `step*_STAR` |
| 6 | interpolate bads + recover reference (AF3) + **average reference** | `step*_final_avgref`, `step06b_alpha_topo` |

## Decoding (models + features + every diagnostic)

- **Epoch**: the 19 s audiovisual block, Audio vs Visual (60 trials, 30/30).
- **Features**: band-power (θ/α/β/γ) per channel; spatial covariance; CSP.
  - `step07_bandpower_topographies` — per-band power by attended stream (Audio, Visual, difference).
  - `step07b_discriminability_AUC` — per-channel single-feature AUC per band.
- **Models** (trial-grouped stratified CV + label-permutation null):
  - band-power + shrinkage-LDA
  - Riemannian tangent-space + logistic regression (pyriemann)
  - CSP + LDA
  - `step08_model_comparison` (accuracy + null p95), `step08b_roc_confusion`,
    `step08c_haufe_patterns` (interpretable activation topographies of the LDA),
    `step08d_csp_patterns`.

## Requirements

In the project `.venv`: `mne`, `meegkit` (STAR), `pyriemann`, `scikit-learn`,
`scipy`, `numpy`, `matplotlib`. Reuses `stim_decoding/sda` (loading + alignment +
trial extraction) and the config `stim_decoding/config_stimdec.yaml`.
