# Visual distraction × speech hierarchy — Kavin (sub001), session 1
## CCA2+ variant (all tracking/decoding uses the `cca2plus` model)

Same analysis and same corrected swapped-channel montage as `../av_distraction/`, but every
tracking/decoding metric uses the **`cca2plus`** preset from `cca_models.py` / aud_cca instead
of `cca2`:

- **Hierarchy tracking** (`pipeline.py`, figures `01`–`04`, `10`, `11`): the per-level
  EEG→speech tracking index is the top-*k* canonical correlation of **cca2plus**.
- **Per-trial decoding** (`cca_decoding.py`, `cca_decoding_topo.png`): the per-trial score is
  the **cca2plus** top canonical correlation. The `cca_models.png` / `cca_best.png` comparison
  figures plot cca1 / cca2 / **cca2plus** side by side.

### What `cca2plus` is
A richer-in-time cousin of `cca2`: EEG is PCA→80, given a 10-tap time-lag basis, whitened to
80 components; the stimulus is given a **wider 80-tap** lag basis (vs 40 in cca2) and whitened
to 80. So it models a longer stimulus–response window than cca2, while keeping cca2's two-stage
PCA regularisation — unlike `cca3`, it does **not** blow up the free-parameter count, so it is
far less prone to the N=30 overfitting that made cca3 unreliable.

### Preprocessing
Uses the updated preprocessing shared by all directories: corrected swapped montage,
**impedance-based** bad-channel rejection (≥ 60 kΩ → PO7, TP10), and **post-clean
spatial-outlier repair** that interpolates the FCz "bullseye" (see the top-level notes and the
`../av_distraction/` provenance).

### Results (cca2plus)
- **Hierarchy distraction effect** (Cohen's *d*, focused − distracted):
  envelope +0.08, word onset +0.37, lexical +0.30, contextual/GPT-2 +0.13.
  Trend **Spearman ρ = +0.20 (p = 0.80)** — cca2+ gives an **inverted-U** (mid-hierarchy
  peak at word onset / lexical), unlike cca2's monotonic envelope→linguistic climb
  (ρ = +0.80). The two models disagree on the shape; single subject, nothing significant.
- **Per-trial decoding** (cca2+ top canonical r, focused vs distracted):
  envelope 0.209 / 0.176, word onset 0.243 / 0.143, lexical 0.211 / 0.102,
  contextual 0.240 / 0.170 — focused > distracted at every level, largest at word
  onset / lexical (the inverted-U peak).

### Contents
- `figures/00`–`04`, `10`, `11` — preprocessing, per-level, and comparative hierarchy figures.
- `figures/cca_backward/models/best/decoding_topo.png` — de Cheveigné CCA/decoding figures.
- `results.json`, `report.md`, `cca_decoding.json`, `preprocessed/`.

Run: `python av_distraction_cca2plus/pipeline.py` then `python av_distraction_cca2plus/cca_decoding.py`
