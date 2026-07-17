#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cca_paper.py - faithful re-implementation of the models in

    de Cheveigne, Wong, Di Liberto, Hjortkjaer, Slaney, Lalor (2018),
    "Decoding the auditory brain with canonical component analysis", NeuroImage.

For a single acoustic descriptor (envelope OR F0 contour) and a set of trials it
provides the paper's four stimulus-response models, each evaluated with
cross-validation as a function of a GLOBAL temporal shift L of the stimulus
relative to the EEG (the paper's Figs 2-4 abscissa):

  * backward  - reconstruct the descriptor from a spatio-temporal EEG filter
                (Fig 2): correlation vs shift + a scalp topography of the
                per-channel stimulus-EEG correlation at the best shift.
  * forward   - predict the best EEG channel from an FIR-filtered descriptor
                (Fig 3): correlation vs shift + the channel's impulse response
                and transfer function.
  * CCA 1/2/3 - jointly transform descriptor and EEG (Fig 4): canonical
                correlation of every CC pair vs shift.  Model 1 = EEG spatial
                (PCA) + FIR descriptor; model 2 = spatio-temporal EEG + FIR
                descriptor; model 3 = spatio-temporal EEG + descriptor filterbank.

Efficiency (the paper notes eigendecomposition is the bottleneck): the EEG-side
covariance does not depend on the shift or on which descriptor is used, so its
eigendecomposition is computed ONCE per cross-validation fold (`eeg_fold_cache`)
and reused across all shifts, descriptors, and models.  Only the small
descriptor-side covariances are recomputed per shift.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import models as M


# ==========================================================================
# building blocks
# ==========================================================================
def lag(x, lags):
    """models._lag on a (T,) or (T,C) array (zeros the wrapped edges)."""
    x = x[:, None] if x.ndim == 1 else x
    return M._lag(x, lags)


def shift_feat(feat, d):
    """Global shift: positive d delays the descriptor relative to the EEG."""
    if d == 0:
        return feat
    out = np.roll(feat, d)
    if d > 0:
        out[:d] = 0.0
    else:
        out[d:] = 0.0
    return out


def filterbank(feat, fs, bands):
    """Descriptor filterbank (paper model 3): (T,) -> (T, n_bands)."""
    ny = fs / 2.0
    out = np.empty((len(feat), len(bands)), np.float64)
    for i, (lo, hi) in enumerate(bands):
        b, a = signal.butter(3, [max(lo, 0.1) / ny, min(hi, ny * 0.99) / ny], "band")
        out[:, i] = signal.filtfilt(b, a, feat)
    return out


def default_bands(band=(1.0, 8.0), n=5):
    edges = np.geomspace(band[0], band[1], n + 1)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def _pool_xx(Xs):
    """(centered Cxx, mean, n) from a list of (T,p) arrays."""
    XtX = sum(X.T @ X for X in Xs).astype(np.float64)
    sx = sum(X.sum(0) for X in Xs).astype(np.float64)
    n = sum(len(X) for X in Xs)
    mx = sx / n
    return XtX / n - np.outer(mx, mx), mx, n


def _cross(Xs, Ys, mx, n):
    """centered Cxy, Cyy, my for lists aligned trial-by-trial (X mean = mx fixed)."""
    XtY = sum(X.T @ Y for X, Y in zip(Xs, Ys)).astype(np.float64)
    YtY = sum(Y.T @ Y for Y in Ys).astype(np.float64)
    sy = sum(Y.sum(0) for Y in Ys).astype(np.float64)
    my = sy / n
    Cxy = XtY / n - np.outer(mx, my)
    Cyy = YtY / n - np.outer(my, my)
    return Cxy, Cyy, my


# ==========================================================================
# cross-validation fold cache (EEG side: shift/descriptor independent)
# ==========================================================================
def eeg_fold_cache(Xlag_list, Xspa_list, folds=4, seed=0):
    """
    Per fold, the eigendecomposition of the pooled TRAIN EEG covariance for both the
    spatio-temporal (lagged) and the spatial (channels) representations. Reused for
    every shift, descriptor, and model -> the expensive eig runs once per fold.
    """
    n = len(Xlag_list)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = max(2, min(folds, n))
    parts = [set(p.tolist()) for p in np.array_split(idx, folds)]
    cache = []
    for held in parts:
        tr = [i for i in idx if i not in held]
        te = sorted(held)
        CxxL, mxL, _ = _pool_xx([Xlag_list[i] for i in tr])
        CxxS, mxS, _ = _pool_xx([Xspa_list[i] for i in tr])
        cache.append({"tr": tr, "te": te,
                      "evL": M._eig(CxxL), "mxL": mxL,
                      "evS": M._eig(CxxS), "mxS": mxS})
    return cache


# ==========================================================================
# CCA model - canonical correlation vs shift (Fig 4)
# ==========================================================================
def cca_curve(cache, Xlag_list, Xspa_list, feat_list, feat_lags, shifts, fs,
              k=6, n_pca_lag=120, n_pca_spa=40, shrink=0.1, model=2,
              bands=None):
    """
    Cross-validated canonical correlation of the top-k CC pairs as a function of the
    global shift, for one descriptor. `model`: 1 (EEG spatial + FIR descriptor),
    2 (EEG lagged + FIR descriptor), 3 (EEG lagged + descriptor filterbank).
    Returns an array (n_shifts, k) of pooled-test canonical correlations.
    """
    spatial = (model == 1)
    bands = bands or _dyadic_bank((1.0, 8.0), 20)[0]     # same model-3 bank as Fig 6

    def build_Y(feat, d):
        fs_ = shift_feat(feat, d)
        return filterbank(fs_, fs, bands) if model == 3 else lag(fs_, feat_lags)

    out = np.zeros((len(shifts), k))
    cnt = np.zeros(len(shifts))
    for fold in cache:
        tr, te = fold["tr"], fold["te"]
        ev, V = fold["evS"] if spatial else fold["evL"]
        mx = fold["mxS"] if spatial else fold["mxL"]
        Xs = Xspa_list if spatial else Xlag_list
        n_pca = n_pca_spa if spatial else n_pca_lag
        Wx = M._whiten_eig(ev, V, shrink, n_keep=n_pca)          # (p, r)
        for si, L in enumerate(shifts):
            d = int(round(L * fs))
            Ytr = [build_Y(feat_list[i], d) for i in tr]
            n = sum(len(Y) for Y in Ytr)
            Cxy, Cyy, my = _cross([Xs[i] for i in tr], Ytr, mx, n)
            evy, Vy = M._eig(Cyy)
            Wy = M._whiten_eig(evy, Vy, shrink)
            U, S, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
            kk = min(k, len(S))
            Ax, Ay = Wx @ U[:, :kk], Wy @ Vt.T[:, :kk]
            # pooled test canonical correlations
            Ute = np.concatenate([(Xs[i] - mx) @ Ax for i in te])
            Vte = np.concatenate([(build_Y(feat_list[i], d) - my) @ Ay for i in te])
            for c in range(kk):
                r = np.corrcoef(Ute[:, c], Vte[:, c])[0, 1]
                out[si, c] += r if np.isfinite(r) else 0.0
            cnt[si] += 1
    return out / np.maximum(cnt[:, None], 1)


# ==========================================================================
# backward model - envelope reconstruction (Fig 2)
# ==========================================================================
def backward_curve(cache, Xlag_list, feat_list, shifts, fs, n_pca=120):
    """
    Cross-validated correlation between the reconstructed and true descriptor as a
    function of the global shift (PCA-truncated spatio-temporal decoder). Returns
    train & test curves. The EEG eig is taken from the fold cache.
    """
    train = np.zeros(len(shifts)); test = np.zeros(len(shifts)); cnt = np.zeros(len(shifts))
    for fold in cache:
        tr, te = fold["tr"], fold["te"]
        ev, V = fold["evL"]; mx = fold["mxL"]
        idx = np.where(ev > 1e-6 * ev[0])[0][:n_pca]
        Vr, evr = V[:, idx], ev[idx]
        for si, L in enumerate(shifts):
            d = int(round(L * fs))
            ytr = [shift_feat(feat_list[i], d) for i in tr]
            my = np.concatenate(ytr).mean()
            n = sum(len(a) for a in ytr)
            Xty = sum((Xlag_list[i] - mx).T @ (ytr[j] - my)
                      for j, i in enumerate(tr))
            Cxy = Xty / n
            w = Vr @ ((Vr.T @ Cxy) / evr)                        # PCA-truncated LS
            # train corr
            ptr = np.concatenate([(Xlag_list[i] - mx) @ w for i in tr])
            ytr_c = np.concatenate([a - my for a in ytr])
            train[si] += _safe_corr(ptr, ytr_c)
            # test corr
            yte = [shift_feat(feat_list[i], d) for i in te]
            pte = np.concatenate([(Xlag_list[i] - mx) @ w for i in te])
            yte_c = np.concatenate([a - my for a in yte])
            test[si] += _safe_corr(pte, yte_c)
            cnt[si] += 1
    return train / np.maximum(cnt, 1), test / np.maximum(cnt, 1)


def backward_topography(eeg_list, feat_list, shift_L, fs):
    """Per-channel correlation between raw EEG and the descriptor at the best shift
    (Fig 2 right)."""
    d = int(round(shift_L * fs))
    E = np.concatenate([e for e in eeg_list])
    f = np.concatenate([shift_feat(feat_list[i], d) for i in range(len(feat_list))])
    return np.array([_safe_corr(E[:, c], f) for c in range(E.shape[1])])


# ==========================================================================
# forward model - encoding, best channel (Fig 3)
# ==========================================================================
def forward_curve(cache, Xspa_list, feat_list, feat_lags, shifts, fs, n_pca=None):
    """
    Cross-validated correlation for the best-predicted EEG channel as a function of
    the global shift (FIR descriptor -> each channel). Also returns the best
    channel's impulse response and transfer function from an all-data fit.
    """
    n_ch = Xspa_list[0].shape[1]
    test = np.zeros(len(shifts)); cnt = np.zeros(len(shifts))
    per_ch = np.zeros((len(shifts), n_ch))
    for fold in cache:
        tr, te = fold["tr"], fold["te"]
        for si, L in enumerate(shifts):
            d = int(round(L * fs))
            Ytr = [lag(shift_feat(feat_list[i], d), feat_lags) for i in tr]
            Y = np.vstack(Ytr); Ey = np.vstack([Xspa_list[i] for i in tr])
            my, me = Y.mean(0), Ey.mean(0)
            Cyy = (Y - my).T @ (Y - my) / len(Y)
            Cye = (Y - my).T @ (Ey - me) / len(Y)
            W = np.linalg.solve(Cyy + 1e-3 * np.eye(Cyy.shape[0]) * np.trace(Cyy)
                                / Cyy.shape[0], Cye)             # (n_lags, n_ch)
            Yte = np.vstack([lag(shift_feat(feat_list[i], d), feat_lags) for i in te])
            Ete = np.vstack([Xspa_list[i] for i in te])
            pred = (Yte - my) @ W
            for c in range(n_ch):
                per_ch[si, c] += _safe_corr(pred[:, c], Ete[:, c] - me[c])
            cnt[si] += 1
    per_ch /= np.maximum(cnt[:, None], 1)
    test = per_ch.max(1)
    best_ch = int(per_ch[per_ch.max(1).argmax()].argmax())
    # all-data impulse response for the best channel at its best shift
    L = shifts[int(test.argmax())]; d = int(round(L * fs))
    Y = np.vstack([lag(shift_feat(feat_list[i], d), feat_lags) for i in range(len(feat_list))])
    E = np.vstack([e for e in Xspa_list])
    my = Y.mean(0)
    Cyy = (Y - my).T @ (Y - my) / len(Y)
    Cye = (Y - my).T @ (E - E.mean(0)) / len(Y)
    W = np.linalg.solve(Cyy + 1e-3 * np.eye(Cyy.shape[0]) * np.trace(Cyy)
                        / Cyy.shape[0], Cye)
    irf = W[:, best_ch]
    tf = np.abs(np.fft.rfft(irf, n=256))
    freqs = np.fft.rfftfreq(256, 1.0 / fs)
    return test, best_ch, irf, tf, freqs


# ==========================================================================
# CCA components at the best shift on ALL trials (Fig 6)
# ==========================================================================
def cca_components(eeg_list, Xlag_list, feat_list, feat_lags, shift_L, fs,
                   k=12, n_pca=120, shrink=0.1):
    """
    Fit CCA model 2 on ALL trials at the best shift and return, per CC:
      * transfer function of the descriptor-side FIR filter (Fig 6 top),
      * EEG topography = cross-correlation of the CC's EEG component with each raw
        channel (Fig 6 bottom).
    """
    d = int(round(shift_L * fs))
    Xs, Ys = Xlag_list, [lag(shift_feat(feat_list[i], d), feat_lags)
                         for i in range(len(feat_list))]
    Cxx, mx, n = _pool_xx(Xs)
    Cxy, Cyy, my = _cross(Xs, Ys, mx, n)
    ev, V = M._eig(Cxx); evy, Vy = M._eig(Cyy)
    Wx = M._whiten_eig(ev, V, shrink, n_keep=n_pca)
    Wy = M._whiten_eig(evy, Vy, shrink)
    U, S, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
    kk = min(k, len(S))
    Ax, Ay = Wx @ U[:, :kk], Wy @ Vt.T[:, :kk]
    n_flag = len(feat_lags)
    # descriptor-side FIR per CC -> transfer function
    tf = np.zeros((kk, 129)); irf = np.zeros((kk, n_flag))
    for c in range(kk):
        h = Ay[:, c]                                             # weight per feat lag
        irf[c] = h
        tf[c] = np.abs(np.fft.rfft(h, n=256))
    freqs = np.fft.rfftfreq(256, 1.0 / fs)
    # EEG topography per CC = corr(component waveform, each raw channel)
    E = np.concatenate([e for e in eeg_list])
    comp = np.concatenate([(Xs[i] - mx) @ Ax for i in range(len(Xs))])  # (Nt, kk)
    n_ch = E.shape[1]
    topo = np.zeros((kk, n_ch))
    for c in range(kk):
        topo[c] = [_safe_corr(comp[:, c], E[:, ch]) for ch in range(n_ch)]
    return {"corrs": S[:kk], "tf": tf, "freqs": freqs, "irf": irf,
            "topo": topo, "feat_lags": feat_lags}


# ==========================================================================
# CCA model 3 components, exactly as in the paper's Fig 6 (filterbank)
# ==========================================================================
def _dyadic_bank(band=(1.0, 8.0), n=16, oct_half=0.42):
    """n overlapping constant-Q band-pass channels, log-spaced over `band`."""
    fcs = np.geomspace(band[0] * 1.05, band[1] * 0.95, n)
    m = 2.0 ** oct_half
    return [(max(fc / m, band[0] * 0.9), min(fc * m, band[1] * 1.02))
            for fc in fcs], fcs


def _design_bank(bands, fs):
    ny = fs / 2.0
    return [signal.butter(3, [max(lo, 0.1) / ny, min(hi, ny * 0.99) / ny], "band")
            for lo, hi in bands]


def _apply_bank(x, coeffs):
    return np.column_stack([signal.filtfilt(b, a, x) for b, a in coeffs])


def cca_components_model3(eeg_list, feat_list, shift_L, fs, k=12, n_pca=120,
                          shrink=0.1, n_bands=16):
    """
    Reproduce the paper's Fig 6 exactly, from CCA MODEL 3 (filterbank). The stimulus
    envelope AND the EEG are passed through a dyadic FIR filterbank; CCA is fit on all
    trials at the best shift. Per canonical component it returns:

      * tf   - amplitude transfer function of the CCA-derived FIR filter applied to
               the stimulus envelope: the combined impulse response
               h_c(t) = sum_k Ay[k,c] * fb_k(t), normalised later to peak 1.
      * topo - EEG topography = cross-correlation of the CC's EEG component waveform
               with each raw EEG channel (individual colour scale per component).
    """
    d = int(round(shift_L * fs))
    bands, fcs = _dyadic_bank((1.0, 8.0), n_bands)
    coeffs = _design_bank(bands, fs)
    # descriptor side: envelope through the filterbank
    Ys = [_apply_bank(shift_feat(feat_list[i], d), coeffs)
          for i in range(len(feat_list))]
    # EEG side: every channel through the same filterbank (spatio-spectral)
    Xs = [np.concatenate([_apply_bank(e[:, ch], coeffs)
                          for ch in range(e.shape[1])], axis=1) for e in eeg_list]
    Cxx, mx, n = _pool_xx(Xs)
    Cxy, Cyy, my = _cross(Xs, Ys, mx, n)
    ev, V = M._eig(Cxx); evy, Vy = M._eig(Cyy)
    Wx = M._whiten_eig(ev, V, shrink, n_keep=n_pca)
    Wy = M._whiten_eig(evy, Vy, shrink)
    U, S, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
    kk = min(k, len(S))
    Ax, Ay = Wx @ U[:, :kk], Wy @ Vt.T[:, :kk]
    # combined stimulus-side filter per component -> SMOOTH transfer function,
    # from the filterbank channels' analytic frequency responses (zero-phase ->
    # |H|^2), weighted by the canonical weights:  TF_c(f) = |sum_k Ay[k,c] |H_k(f)|^2|
    freqs = np.geomspace(0.2, 20.0, 400)
    w = 2 * np.pi * freqs / fs
    HB = np.vstack([np.abs(signal.freqz(b, a, worN=w)[1]) ** 2 for b, a in coeffs])
    tf = np.abs(Ay.T @ HB)                                   # (kk, F), smooth
    # EEG component topographies
    E = np.concatenate([e for e in eeg_list])
    comp = np.concatenate([(Xs[i] - mx) @ Ax for i in range(len(Xs))])
    topo = np.zeros((kk, E.shape[1]))
    for c in range(kk):
        topo[c] = [_safe_corr(comp[:, c], E[:, ch]) for ch in range(E.shape[1])]
    return {"corrs": S[:kk], "tf": tf, "freqs": freqs, "topo": topo,
            "bands": bands, "fcs": fcs, "model": 3}


def _safe_corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0
