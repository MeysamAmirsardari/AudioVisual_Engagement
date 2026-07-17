# EEG analysis — decoding the anticipatory gap

A reproducible pipeline that **cleans** the 64-channel recording and **decodes the
gap between the attention instruction and the audiovisual stimulus onset**.

During that gap the screen is identical across conditions (a fixation dot and the
photodiode square), so anything decodable there reflects the participant's
*internal attentional set* — not a stimulus difference. The decoding target is the
**attended modality (Audio vs Visual)**. Probe answers are irrelevant to this
question and are ignored by design.

```
python run_eeg_pipeline.py                 # runs on config_eeg.yaml
python run_eeg_pipeline.py --apply-montage # enable topographies (see caveat)
python run_eeg_pipeline.py --no-tempgen    # skip the slow generalisation matrix
```

Everything is driven by [`config_eeg.yaml`](config_eeg.yaml); any value can be
overridden on the command line.

## The paradigm, and what each trial looks like

```
   cue / instruction        anticipatory gap            audiovisual block
 ┌───────────────────┐┌──────────────────────────┐┌───────────────────────────┐
 │  "attend to        ││   fixation + photodiode   ││  speech + Tetris (20 s)   │
 │   AUDIO / VISUAL"   ││   square (identical!)     ││                           │
 └───────────────────┘└──────────────────────────┘└───────────────────────────┘
        1.5 s              ~2.0 s (jittered)                probe → ITI
 t = -1.5              t = 0 (gap onset)          t = +gap (stimulus onset)
                       photodiode RISES           photodiode FALLS
```

The photodiode square is drawn for the whole gap. Its **rising** edge (gap onset =
instruction offset) and **falling** edge (stimulus onset) are logged by the
amplifier. We time-lock every epoch to the **gap onset (t = 0)** and analyse:

* **cue window** `[-1.5, 0]` — the instruction is on screen. Decoding here is
  *expected* and partly a low-level visual confound (the words differ). It is the
  pipeline's **positive control**.
* **gap window** `[0, +1.5]` — the clean anticipatory period. `tmax` stays below
  the shortest gap (1.80 s), so a stimulus response can never leak in. **This is
  the headline "decode the gap" analysis.**

## Pipeline stages (`cma_eeg/`)

| stage | module | what it does |
|------|--------|--------------|
| load | `loading.py` | read BrainVision; parse photodiode edges (`S 15`) |
| align | `alignment.py` | pair edges → behaviour trials → **attended-modality labels** |
| clean | `preprocessing.py` | filter · montage · bad channels · reference · ICA |
| epoch | `epoching.py` | gap-onset-locked epochs, safe pre-stimulus window |
| decode | `decoding.py` | time-resolved · temporal-generalisation · whole-window · alpha |
| report | `reporting.py` | figures + a single self-contained HTML report |

### Marker ↔ behaviour alignment (the interesting part)

The amplifier and stimulus-PC clocks differ by a single constant offset. The
photodiode also **drops edges**, so markers cannot be assumed to alternate cleanly.
`alignment.py` therefore:

1. finds the clock offset that lands the most edges on a behaviour event
   (gap-onset / av-onset), then refines it on the matched inliers;
2. classifies every edge and keeps trials where **both** edges survived;
3. **cross-checks** each kept trial's EEG gap duration against the logged
   `delay_seconds` — a mis-pairing cannot pass silently.

For this dataset: clock offset ≈ 0.82 s, edge-match error ≈ 5 ms, gap-duration
cross-check < 9 ms, yielding **38 usable trials (17 Audio / 21 Visual)** out of 60
(the rest lost one or both photodiode edges).

### Decoding

* **Time-resolved** — a sliding shrinkage-LDA gives AUC(t). Significance is a
  **label-permutation test with max-statistic (family-wise-error) correction**
  across time, which respects the non-independent CV folds.
* **Temporal generalisation** — train-time × test-time AUC matrix (King & Dehaene).
* **Whole-window** — one AUC per window from compact per-channel features
  (sub-window ERP means + broadband + alpha power), with a label-permutation
  *p*-value and confusion matrix.
* **Anticipatory alpha** — 8–14 Hz power by condition and decoded from it (the
  classic anticipatory-attention signature).

Workhorse classifier: **shrinkage LDA** — the robust default for small-*n* M/EEG.

## Outputs (`AudVis/derivatives/eeg_pipeline/<subject>/`)

```
<sub>_report.html     one self-contained report — open this first
<sub>_events.csv      the aligned, labelled trial table
<sub>_metrics.json    headline decoding numbers
<sub>_clean_raw.fif   cleaned continuous data
<sub>_gap-epo.fif     epoched data
figures/              every panel as a PNG
```

## ⚠️ Montage caveat (read before enabling topographies)

The channels are recorded as **bare physical numbers** (1, 3, 4, …, 64; electrode
2 is the online reference and is absent), so scalp positions are **unknown** unless
a cap layout is asserted. `montage_acticap64.json` provides the standard
BrainProducts **actiCAP-64 "Standard-2"** ordering as a **template you must verify
against your own cap**. It is **off by default**.

* **Decoding never depends on the montage** — the classifiers use channels as
  anonymous features.
* Turn it on (`--apply-montage` / `preprocess.apply_montage: true`) only to get
  topographies, spherical-spline interpolation of bad channels, and (if
  `mne-icalabel` is installed) ICLabel. A wrong montage corrupts only those
  spatial views, never the decoding numbers.

## Dependencies

Core (already in the project `.venv`): `mne`, `scikit-learn`, `numpy`, `scipy`,
`matplotlib`, `pandas`, `pyyaml`, `joblib`. Optional enhancers are auto-detected
and the pipeline degrades cleanly without them — see
[`requirements-eeg.txt`](requirements-eeg.txt).

## Interpreting single-subject results

With one subject and 38 trials, whole-window point estimates are noisy (AUC 95%
CI ≈ ±0.18); the **permutation tests are the arbiter**, not the point estimate.
The time-resolved analysis with FWER correction is the primary inferential
result. Robust conclusions about anticipatory modality decoding need the group
level — run this per subject and aggregate the AUC(t) curves.
