#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierarchy.py — a hierarchy of speech representations, from low-level acoustic to
high-level linguistic, each as a time series the EEG can be regressed/CCA'd against.

The levels (low -> high):

  1. envelope      broadband amplitude envelope           (acoustic)
  2. spectrogram   log-mel band envelopes                 (acoustic, spectrally detailed)
  3. word_onset    unit impulses at word onsets           (lexical segmentation)
  4. word_frequency  onset impulses scaled by lexical surprise (-log freq)   (lexical access)
  5. gpt2_surprisal  onset impulses scaled by contextual surprise (bits)     (linguistic prediction)

Word-level features are impulse trains placed at the aligned word onsets (from the
.words.json produced with the audio), so a lagged model recovers the word-evoked
response, optionally weighted by how surprising each word is. Everything is produced
at the model rate `fs`, band-limited to the EEG band, and returned as (n_samples,
n_channels) so it drops straight into the CCA machinery.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import stimuli

# the ordered hierarchy (name -> (level index, kind)); level 0 = lowest
LEVELS = ["envelope", "spectrogram", "word_onset", "word_frequency", "gpt2_surprisal"]
LEVEL_KIND = {"envelope": "acoustic", "spectrogram": "acoustic",
              "word_onset": "lexical", "word_frequency": "lexical",
              "gpt2_surprisal": "linguistic"}


# --------------------------------------------------------------------------
# acoustic
# --------------------------------------------------------------------------
def envelope(wav_path, fs, n_samples, cfg, band):
    return stimuli.audio_envelope(wav_path, fs, n_samples, cfg, band)[:, None]


def spectrogram(wav_path, fs, n_samples, band, n_mels=8):
    """Log-mel band envelopes at `fs`, each band band-limited + z-scored -> (n, n_mels)."""
    import librosa
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels,
                                         n_fft=1024, hop_length=256, fmin=50, fmax=8000)
    # per-band amplitude envelope, loudness-compressed exactly like the broadband
    # envelope (amplitude**0.3 = power**0.15) so the bands span the same envelope space
    comp = (mel ** 0.15).T                                 # (frames, n_mels)
    out = signal.resample(comp, n_samples, axis=0)         # -> model rate
    ny = fs / 2.0
    b, a = signal.butter(4, [max(band[0], 0.1) / ny, min(band[1], ny * 0.99) / ny], "band")
    out = signal.filtfilt(b, a, out, axis=0)
    # centre each band but keep the RELATIVE band amplitudes (do NOT z-score per band,
    # which would strip the broadband envelope shared across bands)
    return (out - out.mean(0)).astype(np.float64)


# --------------------------------------------------------------------------
# word-level (need the aligned word list)
# --------------------------------------------------------------------------
def _impulses(words, fs, n_samples, values):
    """A train of impulses at each word onset, each carrying its `values` entry."""
    x = np.zeros(n_samples)
    vals = np.asarray(values, dtype=float)
    for w, v in zip(words, vals):
        i = int(round(float(w["start_s"]) * fs))
        if 0 <= i < n_samples:
            x[i] += v
    return x[:, None]


def word_onset(words, fs, n_samples):
    return _impulses(words, fs, n_samples, np.ones(len(words)))


def word_frequency(words, fs, n_samples, lang="en"):
    """Lexical surprise = -Zipf frequency at each onset (rarer word -> larger impulse)."""
    from wordfreq import zipf_frequency
    surp = [-zipf_frequency(w["word"], lang) for w in words]   # ~ -log10 p
    return _impulses(words, fs, n_samples, surp)


def gpt2_surprisal(words, fs, n_samples, model=None, tokenizer=None):
    """Contextual surprise (bits) of each word under GPT-2 at its onset."""
    surp = gpt2_word_surprisal([w["word"] for w in words], model, tokenizer)
    return _impulses(words, fs, n_samples, surp)


# --------------------------------------------------------------------------
# GPT-2 helper (contextual surprisal per word, summing its sub-word tokens)
# --------------------------------------------------------------------------
def load_gpt2():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return lm, tok


def gpt2_word_surprisal(word_list, model, tokenizer):
    """Per-word surprisal in bits; a word's value is the summed surprisal of its BPE
    tokens given the preceding context (standard word-level LM surprisal)."""
    import math
    import torch
    text = " " + " ".join(word_list)
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"][0]
    with torch.no_grad():
        logp = torch.log_softmax(model(ids[None]).logits[0], -1)
    tok_surp = np.zeros(len(ids))
    for i in range(1, len(ids)):
        tok_surp[i] = -logp[i - 1, ids[i]].item() / math.log(2)
    # map BPE tokens back to words: a new word starts at a token whose text begins " "
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    per_word, cur, wi = [], 0.0, -1
    for t, s in zip(toks, tok_surp):
        if t.startswith("Ġ"):                              # GPT-2 space marker = new word
            if wi >= 0:
                per_word.append(cur)
            cur, wi = s, wi + 1
        else:
            cur += s
    per_word.append(cur)
    # align length to word_list (defensive)
    if len(per_word) < len(word_list):
        per_word += [np.mean(per_word or [0.0])] * (len(word_list) - len(per_word))
    return per_word[:len(word_list)]
