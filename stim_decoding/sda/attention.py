#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attention.py — decode the attended modality from neural tracking of the streams.

Per trial (leave-one-trial-out):
  r_audio  = envelope reconstruction accuracy (backward TRF; + CCA cross-check)
  r_visual = visual-embedding encoding accuracy (forward TRF, occipital ROI)
Decision: the attended stream is the better-tracked one. We report a direct
z-scored comparison and a supervised LDA on the tracking indices, each with a
label-permutation null, plus time-resolved decoding (window-length curve +
sliding window) and the final spatiotemporal TRF patterns.
"""

from __future__ import annotations

import numpy as np

from . import models


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def _roi_idx(ch_names, roi):
    s = set(roi)
    idx = [i for i, c in enumerate(ch_names) if c in s]
    return idx if idx else list(range(len(ch_names)))


def run(trials: dict, envelopes: dict, visuals: dict, ch_names, cfg,
        fs: float, log=print) -> dict:
    m = cfg["model"]
    tr_ids = sorted(trials.keys())
    labels = np.array([trials[t]["label"] for t in tr_ids])
    y = (labels == "Audio").astype(int)                # 1 = Audio, 0 = Visual

    # per-trial model matrices (n_times, n_*)
    EEG = {t: trials[t]["eeg"].T for t in tr_ids}      # (n_times, n_ch)
    ENV = {t: envelopes[t] for t in tr_ids}            # (n_times,)
    VIS = {t: visuals[t] for t in tr_ids}              # (n_times, n_vfeat)
    n_times = EEG[tr_ids[0]].shape[0]
    roi = _roi_idx(ch_names, m["occipital_roi"])

    bt = (m["bwd_tmin_s"], m["bwd_tmax_s"])
    ft = (m["fwd_tmin_s"], m["fwd_tmax_s"])
    lags = list(range(int(round(m["cca_eeg_lags_s"][0] * fs)),
                      int(round(m["cca_eeg_lags_s"][1] * fs)) + 1))
    env_lags = list(range(int(round(m.get("cca_env_lags_s", [0.0, 0.05])[0] * fs)),
                          int(round(m.get("cca_env_lags_s", [0.0, 0.05])[1] * fs)) + 1))

    # ---- select ridge alpha once (audio backward, visual forward) -----------
    log("  selecting ridge alphas ...")
    a_aud = models.select_alpha([EEG[t] for t in tr_ids], [ENV[t][:, None] for t in tr_ids],
                                bt[0], bt[1], fs, m["ridge_alphas"])
    a_vis = models.select_alpha([VIS[t] for t in tr_ids], [EEG[t] for t in tr_ids],
                                ft[0], ft[1], fs, m["ridge_alphas"])
    log(f"  alpha audio(bwd)={a_aud:g}, visual(fwd)={a_vis:g}")

    # ---- leave-one-trial-out tracking indices -------------------------------
    r_aud = np.zeros(len(tr_ids)); r_vis = np.zeros(len(tr_ids))
    r_cca = np.zeros(len(tr_ids))
    recon_env = {}; pred_roi = {}                      # for time-resolved analysis
    for j, t in enumerate(tr_ids):
        tr = [u for u in tr_ids if u != t]
        # auditory backward decoder
        rf_a = models.fit_trf([EEG[u] for u in tr], [ENV[u][:, None] for u in tr],
                              bt[0], bt[1], fs, a_aud)
        r_aud[j] = float(np.nanmean(models.score_trf(rf_a, EEG[t], ENV[t][:, None])))
        recon_env[t] = rf_a.predict(EEG[t][:, None, :])[:, 0, 0]
        # visual forward encoder
        rf_v = models.fit_trf([VIS[u] for u in tr], [EEG[u] for u in tr],
                              ft[0], ft[1], fs, a_vis)
        sc = models.score_trf(rf_v, VIS[t], EEG[t])
        r_vis[j] = float(np.nanmean(np.asarray(sc)[roi]))
        pred_roi[t] = rf_v.predict(VIS[t][:, None, :])[:, 0, :][:, roi]
        # CCA cross-check (auditory) — regularized CCA, held-out top-K corr
        try:
            cca = models.cca_fit([EEG[u] for u in tr], [ENV[u] for u in tr],
                                 lags, env_lags, k=m["cca_components"],
                                 n_pca=m.get("cca_n_pca", 60),
                                 shrink=m.get("cca_shrink", 0.15))
            r_cca[j] = models.cca_score(cca, EEG[t], ENV[t], k=m["cca_components"])
        except Exception:
            r_cca[j] = np.nan
        if (j + 1) % 10 == 0:
            log(f"    LOO {j+1}/{len(tr_ids)}")

    # ---- decisions ----------------------------------------------------------
    s_direct = _z(r_aud) - _z(r_vis)                   # >0 -> Audio
    pred_direct = (s_direct > 0).astype(int)
    acc_direct = float((pred_direct == y).mean())
    auc_direct = _auc(y, s_direct)

    # supervised LDA on the tracking indices (LOO). Include CCA only if it is
    # mostly valid; otherwise fall back to [r_aud, r_vis].
    cols = [_z(r_aud), _z(r_vis)]
    if np.isfinite(r_cca).sum() >= max(4, len(r_cca) // 2):
        rc = np.array(r_cca, float)
        rc[~np.isfinite(rc)] = np.nanmean(rc[np.isfinite(rc)])
        cols.append(_z(rc))
    feats = np.column_stack(cols)
    pred_lda, score_lda = _loo_lda(feats, y, cfg["decode"].get("classifier", "lda"))
    acc_lda = float((pred_lda == y).mean())
    auc_lda = _auc(y, score_lda)

    # permutation nulls (label shuffle; tracking indices are label-independent)
    rng = np.random.RandomState(cfg["decode"]["random_state"])
    nperm = int(cfg["decode"]["n_permutations"])
    p_direct = _perm_p(acc_direct, lambda yy: float((pred_direct == yy).mean()), y, rng, nperm)
    p_lda = _perm_p(acc_lda,
                    lambda yy: float((_loo_lda(feats, yy, cfg["decode"].get("classifier", "lda"))[0] == yy).mean()),
                    y, rng, min(nperm, 500))

    # ---- time-resolved ------------------------------------------------------
    tr_res = _time_resolved(tr_ids, ENV, recon_env, pred_roi, EEG, roi, y, cfg, fs)

    # ---- final models fit on all trials (patterns + saved objects) ----------
    patterns, fitted_trf = _final_patterns(tr_ids, EEG, ENV, VIS, labels, ft, bt,
                                           fs, a_aud, a_vis)
    cca_all, cca_details = _final_cca(tr_ids, EEG, ENV, lags, env_lags, ch_names, fs,
                                      m["cca_components"], shrink=m.get("cca_shrink", 0.15),
                                      n_pca=m.get("cca_n_pca", 60))

    res = {
        "trial_ids": tr_ids, "labels": labels.tolist(),
        "r_audio": r_aud.tolist(), "r_visual": r_vis.tolist(), "r_cca": r_cca.tolist(),
        "decision_score": s_direct.tolist(),
        "alpha_audio": a_aud, "alpha_visual": a_vis, "roi_idx": roi,
        "metrics": {
            "n_trials": len(tr_ids),
            "n_audio": int(y.sum()), "n_visual": int((1 - y).sum()),
            "acc_direct": acc_direct, "auc_direct": auc_direct, "p_direct": p_direct,
            "acc_lda": acc_lda, "auc_lda": auc_lda, "p_lda": p_lda,
            "confusion": _confusion(y, pred_direct),
        },
        "time_resolved": tr_res,
        "patterns": patterns,
        "cca": {"canonical_corrs": cca_details["canonical_corrs"],
                "eeg_lags_s": cca_details["eeg_lags_s"]},
    }
    fitted = {"audio_fwd": fitted_trf["audio_fwd"],
              "audio_bwd": fitted_trf["audio_bwd"],
              "visual_fwd": fitted_trf["visual_fwd"], "cca": cca_all,
              "cca_details": cca_details, "ch_names": ch_names,
              "delays_fwd_s": patterns["delays_fwd_s"],
              "delays_bwd_s": patterns["delays_bwd_s"],
              "alpha_audio": a_aud, "alpha_visual": a_vis,
              "r_cca": r_cca.tolist(), "labels": labels.tolist()}
    return res, fitted


def _final_cca(tr_ids, EEG, ENV, lags, env_lags, ch_names, fs, k, shrink=0.15, n_pca=60):
    """Fit regularized CCA on all trials; return canonical corrs + Haufe patterns."""
    model = models.cca_fit([EEG[t] for t in tr_ids], [ENV[t] for t in tr_ids],
                           lags, env_lags, k=k, shrink=shrink, n_pca=n_pca)
    U = np.concatenate([models.cca_transform(model, EEG[t], ENV[t])[0]
                        for t in tr_ids], axis=0)
    V = np.concatenate([models.cca_transform(model, EEG[t], ENV[t])[1]
                        for t in tr_ids], axis=0)
    corrs = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(U.shape[1])]
    n_ch, n_lags = len(ch_names), len(lags)
    # Haufe forward patterns (interpretable spatiotemporal maps), reshaped per lag.
    xpat = model["Px"].reshape(n_lags, n_ch, -1)                    # (n_lags, n_ch, n_comp)
    ypat = np.asarray(model["Py"])                                 # (n_env_lags, n_comp)
    sub = slice(None, None, max(1, U.shape[0] // 2000))
    return model, {
        "canonical_corrs": corrs,
        "x_loadings": xpat.tolist(), "y_loadings": ypat.tolist(),
        "eeg_lags_s": [L / fs for L in lags],
        "env_lags_s": [L / fs for L in models._ENV_LAGS],
        "comp1_u": U[sub, 0].tolist(), "comp1_v": V[sub, 0].tolist(),
    }


# --------------------------------------------------------------------------
def _auc(y, score):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def _confusion(y, pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y, pred, labels=[1, 0]).tolist()   # [[AA,AV],[VA,VV]]


def _loo_lda(X, y, kind="lda"):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    n = len(y)
    pred = np.zeros(n, int); score = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        clf = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
               if kind == "lda" else LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        pred[i] = clf.predict(X[i:i + 1])[0]
        score[i] = clf.predict_proba(X[i:i + 1])[0, 1]
    return pred, score


def _perm_p(observed, acc_fn, y, rng, nperm):
    null = np.array([acc_fn(rng.permutation(y)) for _ in range(nperm)])
    return float((np.sum(null >= observed) + 1) / (nperm + 1))


def _windows(n, w, step):
    return [(s, s + w) for s in range(0, max(1, n - w + 1), step)]


def _grouped_lda_acc(feat, y, groups):
    """Leave-one-trial-out LDA accuracy on windowed tracking indices."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    feat = np.column_stack([_z(feat[:, c]) for c in range(feat.shape[1])])
    correct = total = 0
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(feat[tr], y[tr])
        correct += int((clf.predict(feat[te]) == y[te]).sum()); total += int(te.sum())
    return correct / total if total else float("nan")


def _window_indices(tr_ids, ENV, recon_env, pred_roi, EEG, roi, s, e):
    """Per-trial [audio_corr, visual_corr] within a window [s, e)."""
    a, v = [], []
    for t in tr_ids:
        a.append(_c(recon_env[t][s:e], ENV[t][s:e]))
        v.append(np.nanmean([_c(pred_roi[t][s:e, k], EEG[t][s:e, roi[k]])
                             for k in range(len(roi))]))
    return np.array(a), np.array(v)


def _time_resolved(tr_ids, ENV, recon_env, pred_roi, EEG, roi, y, cfg, fs):
    """Supervised (LOO-LDA) decoding vs window length and across block time."""
    dc = cfg["decode"]
    n = len(ENV[tr_ids[0]])
    # (a) accuracy vs decision-window length (windows grouped by trial for the LOO)
    wl_acc = {}
    for W in dc["window_lengths_s"]:
        w = int(round(W * fs))
        A, V, G, Y = [], [], [], []
        for j, t in enumerate(tr_ids):
            for (s, e) in _windows(n, w, w):
                aa, vv = _window_indices([t], ENV, recon_env, pred_roi, EEG, roi, s, e)
                A.append(aa[0]); V.append(vv[0]); G.append(t); Y.append(y[j])
        wl_acc[str(W)] = _grouped_lda_acc(np.column_stack([A, V]),
                                          np.array(Y), np.array(G))
    # (b) sliding-window accuracy over block time (one window per trial -> LOO)
    w = int(round(dc["sliding_window_s"] * fs)); step = int(round(dc["sliding_step_s"] * fs))
    centers, sw_acc = [], []
    for (s, e) in _windows(n, w, step):
        a, v = _window_indices(tr_ids, ENV, recon_env, pred_roi, EEG, roi, s, e)
        sw_acc.append(_grouped_lda_acc(np.column_stack([a, v]), y, np.array(tr_ids)))
        centers.append(((s + e) / 2) / fs)
    return {"window_lengths_s": dc["window_lengths_s"], "wl_accuracy": wl_acc,
            "sliding_centers_s": centers, "sliding_accuracy": sw_acc}


def _c(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _final_patterns(tr_ids, EEG, ENV, VIS, labels, ft, bt, fs, a_aud, a_vis):
    """Fit patterns on all trials (and attend-audio subset) for visualisation."""
    aud = [t for t in tr_ids if labels[tr_ids.index(t)] == "Audio"] or tr_ids
    vis = [t for t in tr_ids if labels[tr_ids.index(t)] == "Visual"] or tr_ids
    # auditory forward TRF (envelope -> EEG) on attend-audio trials
    rf_af = models.fit_trf([ENV[t][:, None] for t in aud], [EEG[t] for t in aud],
                           ft[0], ft[1], fs, a_vis, patterns=False)
    # auditory backward decoder pattern (Haufe forward pattern) on attend-audio
    rf_ab = models.fit_trf([EEG[t] for t in aud], [ENV[t][:, None] for t in aud],
                           bt[0], bt[1], fs, a_aud, patterns=True)
    # visual forward TRF (embedding -> EEG) on attend-visual trials
    rf_vf = models.fit_trf([VIS[t] for t in vis], [EEG[t] for t in vis],
                           ft[0], ft[1], fs, a_vis, patterns=False)
    patterns = {
        "delays_fwd_s": (np.asarray(rf_af.delays_) / fs).tolist(),
        "delays_bwd_s": (np.asarray(rf_ab.delays_) / fs).tolist(),
        "audio_fwd_trf": np.asarray(rf_af.coef_)[:, 0, :].tolist(),        # (n_ch, n_delays)
        "audio_bwd_weights": np.asarray(rf_ab.coef_)[0, :, :].tolist(),    # (n_ch, n_delays)
        "audio_bwd_pattern": np.asarray(rf_ab.patterns_)[0, :, :].tolist(), # (n_ch, n_delays)
        "visual_fwd_trf": np.asarray(rf_vf.coef_).tolist(),                # (n_ch, n_vfeat, n_delays)
    }
    return patterns, {"audio_fwd": rf_af, "audio_bwd": rf_ab, "visual_fwd": rf_vf}
