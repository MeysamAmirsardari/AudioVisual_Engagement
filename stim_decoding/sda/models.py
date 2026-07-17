#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py — TRF (mTRF via MNE ReceptiveField) and CCA for stimulus-response.

  * forward encoder   : stimulus features -> EEG  (visual encoding; also the
                        auditory TRF used for spatiotemporal patterns).
  * backward decoder  : EEG -> stimulus            (envelope reconstruction).
  * CCA               : lagged EEG <-> envelope canonical correlation (robust
                        secondary auditory index).

All are fit on POOLED trials (ReceptiveField accepts (n_times, n_epochs, n_feat))
and evaluated on a held-out trial with Pearson correlation.
"""

from __future__ import annotations

import numpy as np
import mne
from mne.decoding import ReceptiveField

mne.set_log_level("ERROR")


def stack(trials: list[np.ndarray]) -> np.ndarray:
    """List of (n_times, n_feat) -> (n_times, n_epochs, n_feat)."""
    return np.stack(trials, axis=1)


def _rf(tmin, tmax, fs, alpha, patterns=False):
    return ReceptiveField(tmin, tmax, fs, estimator=float(alpha),
                          scoring="corrcoef", patterns=patterns, n_jobs=1)


def fit_trf(X_train: list, Y_train: list, tmin, tmax, fs, alpha, patterns=False):
    """Fit a (pooled) receptive field. X,Y are lists of per-trial (n_times, n*)."""
    rf = _rf(tmin, tmax, fs, alpha, patterns=patterns)
    rf.fit(stack(X_train), stack(Y_train))
    return rf


def score_trf(rf, X_test: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    """Per-output correlation on a single held-out trial."""
    return rf.score(X_test[:, None, :], Y_test[:, None, :])


def select_alpha(X: list, Y: list, tmin, tmax, fs, alphas, folds=4) -> float:
    """Pick the ridge alpha maximising mean held-out correlation (trial k-fold)."""
    idx = np.arange(len(X))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    splits = np.array_split(idx, folds)
    best_a, best_s = float(alphas[0]), -np.inf
    for a in alphas:
        scores = []
        for te in splits:
            tr = [i for i in idx if i not in set(te.tolist())]
            rf = fit_trf([X[i] for i in tr], [Y[i] for i in tr],
                         tmin, tmax, fs, a)
            for i in te:
                scores.append(np.nanmean(score_trf(rf, X[i], Y[i])))
        m = float(np.nanmean(scores))
        if m > best_s:
            best_s, best_a = m, float(a)
    return best_a


# --------------------------------------------------------------------------
# CCA (lagged EEG <-> envelope)
# --------------------------------------------------------------------------
def _lag(x: np.ndarray, lags: list[int]) -> np.ndarray:
    """Time-lag a (n_times, n_feat) signal -> (n_times, n_feat*len(lags))."""
    cols = []
    for L in lags:
        cols.append(np.roll(x, L, axis=0))
    out = np.concatenate(cols, axis=1)
    # zero the wrapped-around edges
    mmax = max(abs(min(lags)), abs(max(lags)))
    out[:mmax] = 0
    out[len(x) - mmax:] = 0
    return out


# Default envelope lag window (tight): the EEG carries the temporal response, so
# the envelope needs only a few lags to give CCA a small multi-column target.
_ENV_LAGS = [0, 1, 2, 3]                    # ~0-47 ms @64Hz


def _whiten(C: np.ndarray, gamma: float, n_keep=None, rcond: float = 1e-6):
    """
    PCA-truncated, shrinkage-regularized whitener W (p, k) for covariance C.

    Keep only the top eigen-directions (top `n_keep`, and above rcond*max) — this
    PCA reduction is the ESSENTIAL regularizer that makes high-dim lagged-EEG CCA
    generalize (de Cheveigne rCCA); without it the whitener amplifies noise
    directions and the canonical directions overfit. Within the kept subspace the
    eigenvalues are shrunk toward their mean by gamma, so W = V diag(1/sqrt(eig)).
    """
    evals, evecs = np.linalg.eigh(C)
    evals, evecs = evals[::-1], evecs[:, ::-1]           # descending
    evals = np.clip(evals, 0.0, None)
    idx = np.where(evals > rcond * evals[0])[0]
    if n_keep is not None:
        idx = idx[:int(n_keep)]                          # PCA (spatial) truncation
    evals, evecs = evals[idx], evecs[:, idx]
    evals = (1.0 - gamma) * evals + gamma * evals.mean()  # shrinkage on kept eigs
    return evecs / np.sqrt(evals)                        # (p, k)


def cca_fit(eeg_list: list, env_list: list, eeg_lags: list[int],
            env_lags: list[int] = None, k: int = 3, n_pca: int = 60,
            shrink: float = 0.15) -> dict:
    """
    Regularized spatio-temporal CCA between lagged EEG (X) and lagged envelope (Y).

    Fit ONLY on the trials given (no test data). PCA-reduce + shrinkage-whiten each
    view, then SVD the whitened cross-covariance K = Wx' Cxy Wy: singular values are
    the canonical correlations, and the singular vectors (un-whitened) are the
    canonical weights. The returned model carries its own lag definitions, so
    transform/score need only (model, eeg, env).
    """
    env_lags = _ENV_LAGS if env_lags is None else list(env_lags)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        X = np.concatenate([_lag(e, eeg_lags) for e in eeg_list], axis=0)
        Y = np.concatenate([_lag(v[:, None], env_lags) for v in env_list], axis=0)
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        n = len(Xc)
        Cxx = Xc.T @ Xc / n
        Cyy = Yc.T @ Yc / n
        Cxy = Xc.T @ Yc / n
        Wx = _whiten(Cxx, shrink, n_keep=n_pca)          # (p, kx)  EEG PCA-reduced
        Wy = _whiten(Cyy, shrink, n_keep=None)           # (q, ky)  envelope (small)
        K = Wx.T @ Cxy @ Wy                              # whitened cross-cov
        U, S, Vt = np.linalg.svd(K, full_matrices=False)
        n_comp = int(min(k, len(S)))
        Ax = Wx @ U[:, :n_comp]                          # (p, n_comp) canonical weights
        Ay = Wy @ Vt.T[:, :n_comp]                       # (q, n_comp)
        return {
            "Ax": Ax, "Ay": Ay, "mx": mx, "my": my,
            "corrs": S[:n_comp],                         # canonical correlations (train)
            "Px": Cxx @ Ax, "Py": Cyy @ Ay,              # Haufe forward patterns
            "n_comp": n_comp, "n_pca_kept": int(Wx.shape[1]),
            "eeg_lags": list(eeg_lags), "env_lags": list(env_lags),
        }


def cca_transform(model: dict, eeg: np.ndarray, env: np.ndarray):
    """Project one trial's EEG/envelope onto the fitted canonical directions."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        X = _lag(eeg, model["eeg_lags"]) - model["mx"]
        Y = _lag(env[:, None], model["env_lags"]) - model["my"]
        return X @ model["Ax"], Y @ model["Ay"]


def cca_score(model: dict, eeg: np.ndarray, env: np.ndarray,
              k: int | None = None) -> float:
    """
    Tracking accuracy for ONE held-out trial = mean Pearson correlation across the
    top-K canonical components (computed on that trial alone, no cross-trial
    normalisation).
    """
    U, V = cca_transform(model, eeg, env)
    k = min(k or U.shape[1], U.shape[1])
    corrs = []
    for i in range(k):
        c = np.corrcoef(U[:, i], V[:, i])[0, 1]
        corrs.append(c if np.isfinite(c) else 0.0)
    return float(np.mean(corrs)) if corrs else 0.0


# --------------------------------------------------------------------------
# Fast path on PRE-LAGGED, cached arrays (for the nested-CV LOO evaluator).
# Lagging each trial once and caching the covariance eigendecomposition per fold
# turns an otherwise multi-hour nested grid search into a few minutes: the
# eigendecomposition of Cxx depends only on the fold, not on (n_pca, shrink).
# --------------------------------------------------------------------------
def lag_trials(eeg_list, env_list, eeg_lags, env_lags):
    """Pre-lag each trial once -> (Xlist, Ylist) as float32 (n_times, n_feat)."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        X = [_lag(e, eeg_lags).astype(np.float32) for e in eeg_list]
        Y = [_lag(v[:, None], env_lags).astype(np.float32) for v in env_list]
    return X, Y


def _eig(C):
    ev, V = np.linalg.eigh(C)
    return ev[::-1], V[:, ::-1]                          # descending


def _whiten_eig(ev, V, gamma, n_keep=None, rcond=1e-6):
    ev = np.clip(ev, 0.0, None)
    idx = np.where(ev > rcond * ev[0])[0]
    if n_keep is not None:
        idx = idx[:int(n_keep)]
    ev2, V2 = ev[idx], V[:, idx]
    ev2 = (1.0 - gamma) * ev2 + gamma * ev2.mean()
    return V2 / np.sqrt(ev2)


def _pool(Xlist, Ylist):
    """Pooled, centered covariances from pre-lagged trial arrays."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Cxx = sum(X.T @ X for X in Xlist).astype(np.float64)
        Cyy = sum(Y.T @ Y for Y in Ylist).astype(np.float64)
        Cxy = sum(X.T @ Y for X, Y in zip(Xlist, Ylist)).astype(np.float64)
        sx = sum(X.sum(0) for X in Xlist).astype(np.float64)
        sy = sum(Y.sum(0) for Y in Ylist).astype(np.float64)
        n = sum(len(X) for X in Xlist)
    mx, my = sx / n, sy / n
    return (Cxx / n - np.outer(mx, mx), Cyy / n - np.outer(my, my),
            Cxy / n - np.outer(mx, my), mx, my)


def cca_fit_lagged(Xlist, Ylist, k, n_pca, shrink, eeg_lags, env_lags):
    """Fit rCCA from pre-lagged trial arrays (pool -> whiten -> SVD)."""
    Cxx, Cyy, Cxy, mx, my = _pool(Xlist, Ylist)
    evx, Vx = _eig(Cxx)
    evy, Vy = _eig(Cyy)
    return _model_from_eig(evx, Vx, evy, Vy, Cxx, Cyy, Cxy, mx, my,
                           k, n_pca, shrink, eeg_lags, env_lags)


def _model_from_eig(evx, Vx, evy, Vy, Cxx, Cyy, Cxy, mx, my,
                    k, n_pca, shrink, eeg_lags, env_lags):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Wx = _whiten_eig(evx, Vx, shrink, n_keep=n_pca)
        Wy = _whiten_eig(evy, Vy, shrink)
        U, S, Vt = np.linalg.svd(Wx.T @ Cxy @ Wy, full_matrices=False)
        n_comp = int(min(k, len(S)))
        Ax, Ay = Wx @ U[:, :n_comp], Wy @ Vt.T[:, :n_comp]
        return {"Ax": Ax, "Ay": Ay, "mx": mx, "my": my, "corrs": S[:n_comp],
                "Px": Cxx @ Ax, "Py": Cyy @ Ay, "n_comp": n_comp,
                "n_pca_kept": int(Wx.shape[1]),
                "eeg_lags": list(eeg_lags), "env_lags": list(env_lags)}


def cca_score_lagged(model, X, Y, k=None):
    """Top-K mean canonical correlation for one held-out trial's pre-lagged arrays."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        U = (X - model["mx"]) @ model["Ax"]
        V = (Y - model["my"]) @ model["Ay"]
    k = min(k or U.shape[1], U.shape[1])
    cs = [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)]
    cs = [c if np.isfinite(c) else 0.0 for c in cs]
    return float(np.mean(cs)) if cs else 0.0


def null_corrected_score(model, X, Y, Ynulls, k=None):
    """
    Genuine tracking for one trial = matched correlation MINUS the mean correlation
    of the same EEG against MISMATCHED envelopes (different speech clips). This
    removes the part of the raw correlation that any slow, autocorrelated signal
    would produce, so it does not reward model complexity for its own sake.
    """
    matched = cca_score_lagged(model, X, Y, k=k)
    null = np.mean([cca_score_lagged(model, X, Yn, k=k) for Yn in Ynulls]) \
        if Ynulls else 0.0
    return matched - float(null), matched, float(null)


def select_cca_params(Xlist, Ylist, k, n_pca_grid, shrink_grid,
                      eeg_lags, env_lags, folds=4, seed=0, n_null=3):
    """
    Choose (n_pca, shrink) by K-fold CV on the TRAINING trials ONLY (nested inside
    the outer LOO), maximising the NULL-CORRECTED tracking (matched minus
    mismatched-envelope). Null correction is what gives the grid an interior
    optimum: extreme dimensionality inflates the matched AND the null equally, so
    it stops looking good. Covariance eigendecompositions are cached per fold.
    Returns ((best_n_pca, best_shrink), best_cv_score).
    """
    idx = np.arange(len(Xlist))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = max(2, min(folds, len(idx)))
    fold_sets = [set(f.tolist()) for f in np.array_split(idx, folds)]

    cache = []
    for held in fold_sets:
        tr = [i for i in idx if i not in held]
        if not tr or not held:
            continue
        Cxx, Cyy, Cxy, mx, my = _pool([Xlist[i] for i in tr], [Ylist[i] for i in tr])
        # a fixed mismatch envelope pool (from the inner-train) for each held trial
        nulls = {i: [Ylist[j] for j in rng.choice(tr, size=min(n_null, len(tr)),
                                                   replace=False)] for i in held}
        cache.append((held, nulls, Cxx, Cyy, Cxy, mx, my, _eig(Cxx), _eig(Cyy)))

    best, best_score = (n_pca_grid[0], shrink_grid[0]), -np.inf
    for n_pca in n_pca_grid:
        for shrink in shrink_grid:
            scores = []
            for held, nulls, Cxx, Cyy, Cxy, mx, my, (evx, Vx), (evy, Vy) in cache:
                model = _model_from_eig(evx, Vx, evy, Vy, Cxx, Cyy, Cxy, mx, my,
                                        k, n_pca, shrink, eeg_lags, env_lags)
                for i in held:
                    corr, _, _ = null_corrected_score(model, Xlist[i], Ylist[i],
                                                       nulls[i], k=k)
                    scores.append(corr)
            s = float(np.mean(scores)) if scores else -np.inf
            if s > best_score:
                best_score, best = s, (int(n_pca), float(shrink))
    return best, best_score
