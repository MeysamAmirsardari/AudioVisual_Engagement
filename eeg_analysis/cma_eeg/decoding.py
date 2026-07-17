"""Decode attended modality (Audio vs Visual) from the anticipatory gap.

Analyses
--------
time_resolved            AUC as a function of time (sliding shrinkage-LDA) with a
                         label-permutation, max-statistic (FWER) test across time.
temporal_generalization  train-time x test-time AUC matrix (King & Dehaene 2014):
                         how stable/reactivated the attentional code is.
whole_window             one AUC for a whole window from compact per-channel
                         features (ERP mean + broadband + alpha power), with a
                         label-permutation p-value and a confusion matrix.
alpha_analysis           posterior alpha (8-14 Hz) power by condition + decoding,
                         the classic anticipatory-attention signature.

Workhorse classifier: shrinkage LDA — the robust default for small-n M/EEG.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mne.decoding import GeneralizingEstimator, SlidingEstimator, cross_val_multiscore

from .utils import LOG


# ---------------------------------------------------------------------------
def _clip8(x):
    """Bound standardized features: kills the rare tiny-variance blow-up
    (huge z-scores -> matmul overflow) while leaving normal features untouched.
    Module-level (not a lambda) so the pipeline stays picklable for n_jobs>1."""
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)


def make_classifier(cfg: dict):
    clip = FunctionTransformer(_clip8)
    if cfg.get("classifier", "lda_shrinkage") == "logreg":
        return make_pipeline(StandardScaler(), clip,
                             LogisticRegression(max_iter=2000, C=1.0))
    return make_pipeline(
        StandardScaler(), clip,
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def get_labels(epochs, audio_code: int) -> np.ndarray:
    """Binary target: 1 = Audio (attended), 0 = Visual."""
    return (epochs.events[:, 2] == audio_code).astype(int)


# ---------------------------------------------------------------------------
def _sliding_cv_auc(est, X, y, n_folds, seed) -> np.ndarray:
    """Mean-over-folds AUC(t) for one stratified CV."""
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return cross_val_multiscore(est, X, y, cv=cv, n_jobs=-1).mean(0)   # (times,)


def _find_runs(mask):
    """Yield (start, end) index pairs of contiguous True runs in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _max_cluster_mass(curve, thr):
    """Largest sum-of-(AUC-0.5) over contiguous timepoints exceeding thr(t)."""
    runs = _find_runs(curve > thr)
    return max(((curve[s:e] - 0.5).sum() for s, e in runs), default=0.0)


def time_resolved(epochs, y, cfg, target_hz: float = 100.0) -> dict:
    """Sliding-estimator AUC(t) with a cluster-based label-permutation test.

    The null is built by shuffling the labels and recomputing the whole CV
    AUC(t). Significance uses cluster mass (Maris & Oostenveld): timepoints whose
    AUC exceeds a pointwise cluster-forming threshold (the 95th percentile of the
    null) are grouped into contiguous clusters; each observed cluster's mass is
    compared to the null distribution of the *maximum* cluster mass. This is
    family-wise-error corrected across time, respects the non-independent CV
    folds, and — unlike a max-point test — is sensitive to sustained effects.
    """
    ep = epochs.copy()
    factor = max(1, int(round(ep.info["sfreq"] / target_hz)))
    if factor > 1:
        ep.decimate(factor)
    X, times = ep.get_data(picks="eeg"), ep.times
    est = SlidingEstimator(make_classifier(cfg), scoring="roc_auc", n_jobs=-1)
    seed = int(cfg["random_state"])
    n_folds = int(cfg.get("cv_folds", 5))

    observed = _sliding_cv_auc(est, X, y, n_folds, seed)

    n_perm = int(cfg.get("n_perm_time", 200))
    rng = np.random.RandomState(seed)
    null_curve = np.empty((n_perm, X.shape[-1]))
    for p in range(n_perm):
        null_curve[p] = _sliding_cv_auc(est, X, rng.permutation(y),
                                        n_folds, seed + 1 + p)

    thr = np.percentile(null_curve, 95, axis=0)            # cluster-forming thr(t)
    null_max_mass = np.array([_max_cluster_mass(null_curve[p], thr)
                              for p in range(n_perm)])

    sig = np.zeros(observed.size, bool)
    sig_clusters = []
    for s, e in _find_runs(observed > thr):
        mass = float((observed[s:e] - 0.5).sum())
        pval = float((1 + (null_max_mass >= mass).sum()) / (1 + n_perm))
        if pval < 0.05:
            sig[s:e] = True
        sig_clusters.append((float(times[s]), float(times[e - 1]), pval))

    n_sig = sum(1 for *_, p in sig_clusters if p < 0.05)
    LOG.info("Time-resolved: peak AUC %.3f at %.0f ms | %d cluster(s) formed, "
             "%d significant (p<.05, cluster-corrected)",
             observed.max(), times[observed.argmax()] * 1000,
             len(sig_clusters), n_sig)
    return dict(times=times, mean=observed, thr=thr, sig=sig,
                sig_clusters=sig_clusters,
                null_band=(np.percentile(null_curve, 2.5, 0),
                           np.percentile(null_curve, 97.5, 0)))


# ---------------------------------------------------------------------------
def temporal_generalization(epochs, y, cfg, max_hz: float = 64.0) -> dict:
    """Train-time x test-time AUC matrix on a time-decimated copy (for speed)."""
    ep = epochs.copy()
    factor = max(1, int(round(ep.info["sfreq"] / max_hz)))
    if factor > 1:
        ep.decimate(factor)
    X = ep.get_data(picks="eeg")
    est = GeneralizingEstimator(make_classifier(cfg), scoring="roc_auc", n_jobs=-1)
    cv = StratifiedKFold(int(cfg.get("cv_folds", 5)), shuffle=True,
                         random_state=int(cfg["random_state"]))
    scores = cross_val_multiscore(est, X, y, cv=cv, n_jobs=-1)   # (folds, tr, te)
    gat = scores.mean(0)
    LOG.info("Temporal generalisation: matrix %s, peak AUC %.3f",
             gat.shape, gat.max())
    return dict(times=ep.times, gat=gat)


# ---------------------------------------------------------------------------
def _window_features(epochs, tmin, tmax, alpha_band, n_bins: int = 4) -> np.ndarray:
    """Compact per-channel features over a window, sensitive to transients:
        * mean amplitude in each of ``n_bins`` equal sub-windows (uV)
        * log broadband power over the window
        * log alpha power over the window
    -> (n_epochs, (n_bins + 2) * n_channels). Low enough dimensionality for a
    fast label-permutation test, yet the sub-window means capture onset ERPs.
    """
    ep = epochs.copy().crop(tmin, tmax)
    X = ep.get_data(picks="eeg")                     # (n, ch, t), volts
    sf = ep.info["sfreq"]
    bins = [b * 1e6 for b in
            (arr.mean(axis=2) for arr in np.array_split(X, n_bins, axis=2))]
    broad = np.log(X.var(axis=2) + 1e-30)           # (n, ch)
    nper = min(X.shape[2], int(sf))
    freqs, psd = welch(X, fs=sf, nperseg=nper, axis=2)    # (n, ch, f)
    band = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    alpha = np.log(psd[:, :, band].mean(axis=2) + 1e-30)  # (n, ch)
    return np.concatenate(bins + [broad, alpha], axis=1)


def whole_window(epochs, y, cfg, window, name: str, alpha_band) -> dict:
    """Single-AUC decoder for a window, with a label-permutation p-value."""
    X = _window_features(epochs, window[0], window[1], alpha_band)
    clf = make_classifier(cfg)
    cv = StratifiedKFold(int(cfg.get("cv_folds", 5)), shuffle=True,
                         random_state=int(cfg["random_state"]))
    # cross-validated AUC via out-of-fold decision scores
    y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    _, perm_scores, pval = permutation_test_score(
        clf, X, y, scoring="roc_auc", cv=cv,
        n_permutations=int(cfg.get("n_permutations", 1000)),
        random_state=int(cfg["random_state"]), n_jobs=-1)
    LOG.info("Whole-window [%s] %.2f-%.2f s: AUC %.3f (p=%.3f, perm)",
             name, window[0], window[1], auc, pval)
    return dict(name=name, window=window, auc=float(auc), p=float(pval),
                confusion=cm, y=y, y_score=y_score,
                perm_null=perm_scores)


# ---------------------------------------------------------------------------
def alpha_analysis(epochs, y, cfg, window) -> dict:
    """Anticipatory alpha (8-14 Hz) power per channel + decoding from it."""
    band = cfg.get("alpha_band", [8.0, 14.0])
    ep = epochs.copy().crop(window[0], window[1])
    X = ep.get_data(picks="eeg")
    sf = ep.info["sfreq"]
    nper = min(X.shape[2], int(sf))
    freqs, psd = welch(X, fs=sf, nperseg=nper, axis=2)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    alpha = np.log(psd[:, :, sel].mean(axis=2) + 1e-30)      # (n, ch)

    audio_map = alpha[y == 1].mean(0)
    visual_map = alpha[y == 0].mean(0)

    clf = make_classifier(cfg)
    cv = StratifiedKFold(int(cfg.get("cv_folds", 5)), shuffle=True,
                         random_state=int(cfg["random_state"]))
    y_score = cross_val_predict(clf, alpha, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_score)
    LOG.info("Alpha-power decoding [%.0f-%.0f Hz]: AUC %.3f",
             band[0], band[1], auc)
    return dict(band=band, ch_names=ep.ch_names, info=ep.info,
                audio=audio_map, visual=visual_map, diff=audio_map - visual_map,
                auc=float(auc))
