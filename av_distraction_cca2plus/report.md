# Visual distraction × speech hierarchy — Kavin (sub001), session 1

*2026-07-15T12:28:13*

**Hypothesis.** Visual distraction reduces speech tracking more at higher hierarchy levels.

**Design.** attend-Audio = focused on speech; attend-Visual (Tetris) = visual distraction. Per-level EEG→speech tracking via the aud_cca regularised CCA; the distraction effect is the focused−distracted difference (Cohen's d).

**Data.** 60 trials (30 focused / 30 distracted), corrected cable-swap montage, 1–8 Hz, 64 Hz.

## Preprocessing
![preprocessing](figures/00_preprocessing.png)

## Each level of the hierarchy

### Level 1: envelope (acoustic)
distraction effect d = +0.08 (p = 0.373); focused +0.176 vs distracted +0.172

![envelope](figures/01_envelope.png)

### Level 2: word_onset (lexical)
distraction effect d = +0.37 (p = 0.070); focused +0.144 vs distracted +0.133

![word_onset](figures/02_word_onset.png)

### Level 3: word_frequency (lexical)
distraction effect d = +0.30 (p = 0.118); focused +0.130 vs distracted +0.121

![word_frequency](figures/03_word_frequency.png)

### Level 4: gpt2_surprisal (linguistic)
distraction effect d = +0.13 (p = 0.280); focused +0.142 vs distracted +0.137

![gpt2_surprisal](figures/04_gpt2_surprisal.png)

## Comparative
![tracking](figures/10_tracking_by_level.png)

![hypothesis](figures/11_distraction_vs_hierarchy.png)

## Result

Trend of the distraction effect across the hierarchy: Spearman ρ = +0.20 (p = 0.800). Directionally consistent with the hypothesis (distraction grows up the hierarchy) — single subject, so treat as a pilot.
