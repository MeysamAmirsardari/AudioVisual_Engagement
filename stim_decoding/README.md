# stim_decoding — decoding attention from stimulus-response tracking

Decodes the **attended modality** (Audio vs Visual) during the 20 s audiovisual
**block** — not the anticipatory gap — by measuring how strongly the EEG tracks
each stream. This complements `eeg_analysis/` (which decodes the pre-stimulus
cue/gap): here the label comes from the *neural following of the stimuli
themselves*.

## Idea

| stream | model | direction | tracking index |
|--------|-------|-----------|----------------|
| **Auditory** | speech **envelope** | backward TRF (EEG → envelope) + **CCA** | reconstruction *r* |
| **Visual** | visual **embedding** | forward TRF (embedding → EEG) | encoding *r* (occipital ROI) |

On each trial the attended stream should be the one the EEG tracks better, so the
per-trial decision is `z(r_audio) − z(r_visual)`. Tracking indices are computed
**leave-one-trial-out** (the held-out trial is never in the model's training
set), and the decision is evaluated with a label-permutation null.

- **Audio envelope**: broadband Hilbert magnitude, loudness-compressed, band-
  limited to the EEG band.
- **Visual embedding**: reconstructed *deterministically* from each trial's Tetris
  game record (`data/<subj>/…/games/…_tNNN.json`) — motion energy, luminance,
  line-clear impulses, and shared frame-PCA components. (This is why the game is
  deterministic + saved: the exact visual stimulus is recoverable offline.)

## Outputs

`AudVis/derivatives/stim_decoding/<subject>/`:

- `figures/`
  - `accuracy.png`, `tracking_scatter.png` — per-trial decoding (direct + LDA,
    permutation p) and the two tracking indices.
  - `time_resolved.png` — accuracy vs decision-window length (AAD hallmark) +
    sliding-window time-course.
  - `spatiotemporal_patterns.png` — auditory/visual TRF & decoder **topographies**
    over lags; `trf_timecourses.png` — GFP time-courses.
  - `trf_heatmaps.png` — channel×lag weight images; `trf_butterfly.png` —
    all-channel TRF overlays.
  - `visual_trf_features.png` — per-embedding-feature visual TRF (GFP + peak topo).
  - `cca_details.png` — canonical correlations, comp-1 EEG spatial pattern,
    loading-vs-lag, per-trial CCA index by condition, canonical-variate scatter.
- `<subject>_models.joblib` — the **trained models** (backward decoder, forward
  encoders as `mne.decoding.ReceptiveField`, and the fitted `CCA` + details) for
  reuse/inspection.
- `<subject>_stimdec.json` (summary + metrics), `<subject>_patterns.npz`,
  `_cache.pkl` (cleaned trials + features for `--use-cache`).

## Continuous preprocessing and QC

The stimulus-response pipeline cleans once at the session level, then only crops
the audiovisual blocks. This is essential for cross-validated CCA: every block
must retain the same channel count, montage, and reference. It therefore never
interpolates or re-references individual trials.

1. Physical channel numbers are mapped to the verified `.bvef` cap montage.
2. The documented bad-channel schedule is combined with conservative continuous
   QC (flat/non-finite traces, variance outliers, weak common-mode correlation,
   and excess 20–45 Hz noise); selected channels are interpolated once before ICA.
3. ICA is fitted on a separate 1–40 Hz copy, then applied before the final 1–8 Hz
   model filter. This keeps ocular and muscle signatures available to ICA.
4. Incomplete or grossly artifacted audiovisual blocks are excluded rather than
   padded or spatially repaired. The run aborts if either attention condition has
   fewer than eight clean blocks.

Each new run writes `<subject>_preprocessing_qc.json`, including channel metrics,
automatic and scheduled decisions, ICA exclusions, and a keep/reject record for
every block. Review this ledger before interpreting decoding results.

## Run

```bash
python stim_decoding/run_stim_decoding.py           # uses config_stimdec.yaml
python stim_decoding/run_stim_decoding.py --no-ica  # skip ICA (debugging)
python stim_decoding/run_stim_decoding.py --preprocess-only  # save/inspect QC before decoding
```

Reuses `eeg_analysis/cma_eeg` for the BrainVision load + photodiode alignment
(av-onset per trial). All knobs live in `config_stimdec.yaml`.

## Kavin (sub001) specifics

- Time-varying noisy channels are interpolated **per trial** from a schedule
  (`preprocess.bad_channel_schedule`); see `notes/kavin_channels.md`.
- Topographies assume the actiCAP-64 montage (`preprocess.apply_montage: true`).
  Verify the montage against the real cap before over-interpreting topographies;
  the decoding itself does not depend on it.
- Single subject → treat effects as exploratory; the framework is built for a
  group extension.
