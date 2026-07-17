# Visual distraction × speech hierarchy — Kavin (sub001), session 1

*2026-07-15T00:10:43*

**Hypothesis.** Visual distraction reduces speech tracking more at higher hierarchy levels.

**Design.** attend-Audio = focused on speech; attend-Visual (Tetris) = visual distraction. Per-level EEG→speech tracking via the aud_cca regularised CCA; the distraction effect is the focused−distracted difference (Cohen's d).

**Data.** 60 trials (30 focused / 30 distracted), corrected cable-swap montage, 1–8 Hz, 64 Hz.

## Preprocessing
![preprocessing](figures/00_preprocessing.png)

## Each level of the hierarchy

### Level 1: envelope (acoustic)
distraction effect d = -0.10 (p = 0.638); focused +0.131 vs distracted +0.135

![envelope](figures/01_envelope.png)

### Level 2: word_onset (lexical)
distraction effect d = +0.08 (p = 0.368); focused +0.096 vs distracted +0.094

![word_onset](figures/02_word_onset.png)

### Level 3: word_frequency (lexical)
distraction effect d = +0.06 (p = 0.391); focused +0.086 vs distracted +0.084

![word_frequency](figures/03_word_frequency.png)

### Level 4: gpt2_surprisal (linguistic)
distraction effect d = +0.21 (p = 0.197); focused +0.102 vs distracted +0.094

![gpt2_surprisal](figures/04_gpt2_surprisal.png)

## Comparative
![tracking](figures/10_tracking_by_level.png)

![hypothesis](figures/11_distraction_vs_hierarchy.png)

## Result

Trend of the distraction effect across the hierarchy: Spearman ρ = +0.80 (p = 0.200). Directionally consistent with the hypothesis (distraction grows up the hierarchy) — single subject, so treat as a pilot.
