"""
Extended probe suite — supplements the linear+MLP probes in pipeline.py
with three additions that more directly measure what L2's GP grammar will
actually extract from the SSL encoder's output.

Additions:

1. Nonlinear-probe bracketing — TWO families with different hypothesis spaces:

   (A) Tree-ensemble family — axis-aligned thresholds on raw coordinates:
       - Random-forest (bagging over independently-trained deep trees)
       - Histogram gradient-boosted trees (boosting + shallow trees,
         sequential residual fitting)
     RF and GBT differ in AGGREGATION, not hypothesis space; they
     constitute one family with two aggregation regimes.

   (B) Smooth-kernel family — polynomial combinations in feature space:
       - Polynomial kernel ridge (degrees 2-3 swept)
     Opposite bias from tree-based — smooth in feature space, closer to
     the arithmetic-composition bias of L2's GP grammar.

   Neither family IS a GP proxy — GP has its own inductive bias (typed
   arithmetic + conditionals). But together they bracket the
   nonlinear-signal extractable from the embedding across two genuinely
   different hypothesis spaces.

   Reading the results:
   - all families > linear → nonlinear signal IS present
   - tree-ensemble > kernel → signal lives in thresholds / regime
     boundaries (regime-dependent structure)
   - kernel > tree-ensemble → signal lives in smooth combinations
     (arithmetic relationships, ratios, products)
   - both roughly equal → signal is bias-robust; L2 will likely find it

   Citations: Olson et al. 2018 (PMLB benchmark), Virgolin & Pissis 2022
   establish that no single model class proxies GP reliably; two-family
   bracketing is the defensible alternative.

2. Frozen-embedding reconstruction probe
   Fit a fresh Ridge regression from pooled emb_fine (3072-d) to each
   variate's time-averaged value across the 60-bar window. Directly
   measures how much input information survives the pooling step — i.e.
   performance on the encoder's own training task using only the summary
   embedding L2 will consume.

3. Embedding variance + effective-rank diagnostic
   Per-dimension variance summary + effective rank via SVD eigenvalue
   entropy. Detects collapse and tells us how many of the 3072 dims are
   actually being used.

None of these touch pipeline.py or the QC-deployed notebook — this is a
supplementary module invoked after training completes.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import warnings

import numpy as np


# ===========================================================================
# Probe 1: Random-forest fits (regression + classification)
# ===========================================================================

def fit_rf_regression_probe(X_train, y_train, X_test, y_test,
                              seed: int = 0,
                              n_estimators: int = 200,
                              max_depth: Optional[int] = 12,
                              max_features="sqrt",
                              min_samples_leaf: int = 20) -> Dict[str, Any]:
    """Fit RandomForestRegressor as a nonlinear-signal-presence probe.

    AI Engineer H1 regularization (Breiman 2001 defaults):
      - max_features='sqrt' → ~55 features per split on 3072-d input
        (vs. sklearn's 1.0 = all features, which memorizes)
      - max_depth=12 → bounded tree depth
      - min_samples_leaf=20 → bigger-than-default leaves, prevents singleton
        overfit on N ~ few thousand training windows

    Also reports OOB score alongside test R² for reviewer-visible
    regularization check (oob > test means overfit).

    StandardScaler is applied for parity with the linear path (RF is
    scale-invariant, so this is a no-op but keeps the call-site
    interchangeable).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        # n_jobs capped to 4 after 2026-04-23 crash: loky spawned
        # one worker per core on a 16 GB Mac, each fork-copying the
        # 3072-d input — drove RSS past the jetsam threshold.
        n_jobs=4,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_tr, y_train)
    r2 = float(model.score(X_te, y_test))
    oob = float(model.oob_score_) if hasattr(model, "oob_score_") else float("nan")

    return dict(
        r2=r2,
        oob_score=oob,
        n_estimators=n_estimators,
        max_depth=-1 if max_depth is None else int(max_depth),
        max_features=str(max_features),
        min_samples_leaf=min_samples_leaf,
    )


def fit_rf_classification_probe(X_train, y_train, X_test, y_test,
                                  seed: int = 0,
                                  n_estimators: int = 200,
                                  max_depth: Optional[int] = 12,
                                  max_features="sqrt",
                                  min_samples_leaf: int = 20) -> Dict[str, Any]:
    """Fit RandomForestClassifier as a nonlinear-signal-presence probe.

    Same Breiman-2001 regularization defaults as fit_rf_regression_probe
    (AI Engineer H1). See that function's docstring for rationale.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        oob_score=True,
        # n_jobs capped to 4 after 2026-04-23 crash (see regression probe).
        n_jobs=4,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_tr, y_train)

    bal_acc = float(balanced_accuracy_score(y_test, model.predict(X_te)))
    oob = float(model.oob_score_) if hasattr(model, "oob_score_") else float("nan")

    try:
        y_proba = model.predict_proba(X_te)
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        auc = float("nan")

    return dict(
        balanced_accuracy=bal_acc,
        auc_roc=auc,
        oob_score=oob,
        n_estimators=n_estimators,
        max_depth=-1 if max_depth is None else int(max_depth),
        max_features=str(max_features),
        min_samples_leaf=min_samples_leaf,
    )


# ===========================================================================
# Probe 1b: Histogram gradient-boosted trees (boosting bias)
# ===========================================================================

_GBT_EARLY_STOP_MIN_TRAIN = 1500  # below this, early-stop is too noisy


def fit_gbt_regression_probe(X_train, y_train, X_test, y_test,
                              seed: int = 0,
                              max_iter: int = 300,
                              max_depth: int = 6,
                              learning_rate: float = 0.05,
                              min_samples_leaf: int = 20) -> Dict[str, Any]:
    """Fit HistGradientBoostingRegressor as the boosting-bias nonlinear probe.

    Different inductive bias from RF:
      - RF averages many independently-trained deep trees (bagging)
      - GBT builds a sequence of shallow trees, each fitting residuals of
        the prior ensemble (boosting)

    Typical signal-quality relationship: GBT beats RF on smooth-nonlinear
    signal; RF beats GBT on strongly-interacting-threshold signal. Running
    both tells you which regime the representation lies in.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # AI Engineer M2 / Code Reviewer #3: disable early-stopping on small
    # folds. 15% holdout of a <1500-sample fold = <225 samples → noisy
    # early-stop decisions that depress GBT scores relative to its
    # bias-bracketing position.
    use_es = len(X_tr) >= _GBT_EARLY_STOP_MIN_TRAIN
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=use_es,
        # Pass valid int/float defaults unconditionally — newer sklearn rejects
        # None even when early_stopping=False (fold-0 crash 2026-04-23 run).
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_tr, y_train)
    r2 = float(model.score(X_te, y_test))

    return dict(
        r2=r2,
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        early_stopping=use_es,
        n_iter_actual=int(getattr(model, "n_iter_", max_iter)),
    )


def fit_gbt_classification_probe(X_train, y_train, X_test, y_test,
                                   seed: int = 0,
                                   max_iter: int = 300,
                                   max_depth: int = 6,
                                   learning_rate: float = 0.05,
                                   min_samples_leaf: int = 20) -> Dict[str, Any]:
    """Fit HistGradientBoostingClassifier as the boosting-bias classification probe."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    # Code Reviewer MUST-FIX #4: HistGradientBoostingClassifier doesn't accept
    # class_weight='balanced' like RF does; we use sample_weight instead.
    # Note for reviewers: sample_weight reweights the log-loss gradient,
    # class_weight reweights leaf-value predictions — semantically similar
    # for imbalance correction but not identical. Adequate for probe
    # comparison; flag in dissertation if the two-family imbalance handling
    # is a concern.
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y_train)

    use_es = len(X_tr) >= _GBT_EARLY_STOP_MIN_TRAIN
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=use_es,
        # Pass valid int/float defaults unconditionally — newer sklearn rejects
        # None even when early_stopping=False (fold-0 crash 2026-04-23 run).
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model.fit(X_tr, y_train, sample_weight=sw)

    bal_acc = float(balanced_accuracy_score(y_test, model.predict(X_te)))

    try:
        y_proba = model.predict_proba(X_te)
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        auc = float("nan")

    return dict(
        balanced_accuracy=bal_acc,
        auc_roc=auc,
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        early_stopping=use_es,
        n_iter_actual=int(getattr(model, "n_iter_", max_iter)),
    )


# ===========================================================================
# Probe 1c: Polynomial kernel ridge (smooth nonlinear bias)
# ===========================================================================

def fit_poly_kernel_ridge_probe(X_train, y_train, X_test, y_test,
                                  degrees=(2, 3),
                                  alpha_grid=None,
                                  subsample_for_kernel: int = 1000) -> Dict[str, Any]:
    """Polynomial kernel ridge regression — the smooth-nonlinear probe.

    Captures polynomial combinations of embedding dimensions (e.g. x_i × x_j,
    x_i² × x_k). Bias is opposite to tree-based probes: smooth in feature
    space, struggles with sharp thresholds.

    AI Engineer H2: sweeps degrees {2, 3} and reports the better. Degree 2
    alone systematically under-brackets GP (which builds depth-5 trees of
    Multiply/SafeDivide reaching implicit degree ≫ 2). Degree 4+ is
    numerically fragile on standardized embeddings.

    Code Reviewer MUST-FIX #2: subsample cap lowered from 3000 → 2000 and
    inner GridSearchCV n_jobs=2 (was -1) to avoid OOM when multiple folds
    of 3000×3000 Gram matrices are computed in parallel.

    Classification variant intentionally omitted: KernelRidge is regression-
    only; adapting via SVC-poly changes the loss function (hinge vs. squared)
    in ways that confuse the comparison. For classification the RF + GBT
    (tree-ensemble) side of the bracket stands alone.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score

    if alpha_grid is None:
        alpha_grid = np.logspace(-2, 3, 6)

    # Subsample training set if too large (O(N²) memory)
    rng = np.random.RandomState(0)
    if len(X_train) > subsample_for_kernel:
        idx = rng.choice(len(X_train), subsample_for_kernel, replace=False)
        X_tr_sub = X_train[idx]
        y_tr_sub = y_train[idx]
        subsampled = True
    else:
        X_tr_sub = X_train
        y_tr_sub = y_train
        subsampled = False

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_sub)
    X_te_s = scaler.transform(X_test)

    inner_cv = min(3, max(2, len(y_tr_sub) // 100))

    # Sweep degrees; keep the best test score
    best_r2 = -np.inf
    best_record = None
    for degree in degrees:
        grid = GridSearchCV(
            KernelRidge(kernel="poly", degree=int(degree), coef0=1.0),
            param_grid={"alpha": alpha_grid},
            cv=inner_cv,
            scoring="r2",
            # n_jobs=1 after 2026-04-23 crash — GridSearchCV forking
            # on top of the encoder-phase RSS was too aggressive.
            n_jobs=1,
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                grid.fit(X_tr_s, y_tr_sub)
            r2 = float(r2_score(y_test, grid.best_estimator_.predict(X_te_s)))
        except Exception as e:
            # Skip degrees that fail (e.g. Gram matrix ill-conditioning at d=3)
            continue
        if r2 > best_r2:
            best_r2 = r2
            best_record = dict(
                r2=r2,
                degree=int(degree),
                alpha=float(grid.best_params_["alpha"]),
                n_train_used=int(len(X_tr_s)),
                subsampled=subsampled,
                degrees_swept=list(degrees),
            )

    if best_record is None:
        return dict(r2=float("nan"), degree=None, error="all degrees failed")
    return best_record


# ===========================================================================
# Permutation-test p-value (Ojala & Garriga 2010)
# ===========================================================================

def permutation_test_probe_score(fit_probe_fn, X_train, y_train, X_test, y_test,
                                   score_key: str = "r2",
                                   n_permutations: int = 100,
                                   seed: int = 0,
                                   **probe_kwargs) -> Dict[str, Any]:
    """Test statistical significance of a probe's observed score by
    comparing against a null distribution produced by shuffling training
    labels n_permutations times and refitting.

    Reference: Ojala & Garriga 2010, "Permutation Tests for Studying
    Classifier Performance" (JMLR). Canonical citation for probe
    significance testing.

    Args:
        fit_probe_fn: one of the fit_*_probe functions here or in pipeline
        score_key:    which key in the probe's return dict carries the
                      test metric ("r2" for regression, "balanced_accuracy"
                      for classification)
        n_permutations: 100 is a sensible default; for p<0.01 resolution,
                      use 1000 (10× cost).
        probe_kwargs: forwarded to fit_probe_fn

    Returns:
        dict with observed score, null-distribution stats, and p-value.
    """
    rng = np.random.RandomState(seed)

    # Observed score on true labels
    observed = fit_probe_fn(X_train, y_train, X_test, y_test, **probe_kwargs)
    observed_score = float(observed.get(score_key, float("nan")))

    # Null distribution
    null_scores = np.empty(n_permutations, dtype=np.float64)
    y_shuffled = y_train.copy()
    for i in range(n_permutations):
        rng.shuffle(y_shuffled)
        r = fit_probe_fn(X_train, y_shuffled, X_test, y_test, **probe_kwargs)
        null_scores[i] = float(r.get(score_key, float("nan")))

    # p-value: fraction of null scores ≥ observed.
    # Code Reviewer MUST-FIX #1: previously filtered NaN null scores and shrunk
    # denominator → biased p upward. Now: replace NaN with -inf (they
    # COULDN'T have beaten observed anyway) and keep denominator=N.
    null_scores_safe = np.where(np.isfinite(null_scores), null_scores, -np.inf)
    n_ge = int(np.sum(null_scores_safe >= observed_score))
    p_value = (n_ge + 1) / (n_permutations + 1)    # Ojala & Garriga smoothing
    floor_p = 1.0 / (n_permutations + 1)            # lowest reportable p
    n_censored = int(np.sum(~np.isfinite(null_scores)))

    finite_null = null_scores[np.isfinite(null_scores)]
    return dict(
        observed_score=observed_score,
        null_mean=float(finite_null.mean()) if len(finite_null) else float("nan"),
        null_std=float(finite_null.std())   if len(finite_null) else float("nan"),
        null_p05=float(np.percentile(finite_null, 5)) if len(finite_null) else float("nan"),
        null_p95=float(np.percentile(finite_null, 95)) if len(finite_null) else float("nan"),
        null_max=float(finite_null.max())   if len(finite_null) else float("nan"),
        p_value=p_value,
        p_floor=floor_p,                             # print as "p ≤ floor_p" when n_ge==0
        is_p_at_floor=(n_ge == 0),
        n_permutations=int(n_permutations),
        n_null_nan=n_censored,                       # for transparency
        score_key=score_key,
    )


# ===========================================================================
# Probe 2: Frozen-embedding reconstruction
# ===========================================================================

def fit_frozen_reconstruction_probe(emb_fine: np.ndarray,
                                      X_windows: np.ndarray,
                                      train_idx: np.ndarray,
                                      test_idx: np.ndarray,
                                      alpha: float = 1.0,
                                      target: str = "last_bar") -> Dict[str, Any]:
    """Measure how well the pooled emb_fine summary reconstructs variates.

    For each variate v, fit a RidgeCV regression from the pooled embedding
    (3072-d) to a per-variate scalar target derived from the window. This
    tests whether pooling destroys variate-level information downstream
    consumers (L2, probes) need.

    Three target choices:
      - "last_bar":   X_windows[:, -1, v]  — value at end of window (default,
                      matches what the encoder sees as "current state")
      - "time_mean":  X_windows.mean(axis=1)[:, v]  — time-average in window
      - "time_std":   X_windows.std(axis=1)[:, v]   — realized-vol proxy

    A variate with high R² here means: given just the emb_fine summary,
    you can recover that variate's current value. Low R² = pooling step
    lost it.

    Args:
        emb_fine: (N_windows, 3072) pooled embeddings from frozen encoder
        X_windows: (N_windows, T=60, V=147) original windows
        train_idx, test_idx: integer index arrays for walk-forward splits
        alpha: Ridge regularization strength (RidgeCV sweeps around this)
        target: see above

    Returns:
        dict with mean_r2, median_r2, per_variate_r2, and counts.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    # Select per-variate target
    if target == "last_bar":
        targets = X_windows[:, -1, :].astype(np.float32)
    elif target == "time_mean":
        targets = X_windows.mean(axis=1).astype(np.float32)
    elif target == "time_std":
        targets = X_windows.std(axis=1).astype(np.float32)
    else:
        raise ValueError(f"unknown target {target!r}")
    _, V = targets.shape

    X_tr = emb_fine[train_idx]
    X_te = emb_fine[test_idx]
    y_tr = targets[train_idx]
    y_te = targets[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    per_var_r2 = np.full(V, np.nan, dtype=np.float32)
    skipped_variates = []
    # RidgeCV picks alpha via leave-one-out — removes the alpha-tuning
    # sensitivity we saw in the unit test with fixed alpha=1.0.
    alpha_grid = np.logspace(-3, 5, 20)  # #216 unified with pipeline.py RidgeCV
    for v in range(V):
        y_v_tr = y_tr[:, v]
        y_v_te = y_te[:, v]
        # Skip degenerate variates — NaN train, NaN test, or zero-variance train
        if not np.isfinite(y_v_tr).all():
            skipped_variates.append((v, "nan_train"))
            continue
        if not np.isfinite(y_v_te).all():
            skipped_variates.append((v, "nan_test"))
            continue
        if np.nanstd(y_v_tr) < 1e-6:
            skipped_variates.append((v, "zero_var_train"))
            continue
        try:
            model = RidgeCV(alphas=alpha_grid)
            model.fit(X_tr_s, y_v_tr)
            per_var_r2[v] = float(model.score(X_te_s, y_v_te))
        except Exception as e:
            skipped_variates.append((v, f"fit_error:{type(e).__name__}"))
            continue

    finite = np.isfinite(per_var_r2)
    mean_r2 = float(per_var_r2[finite].mean()) if finite.any() else float("nan")
    median_r2 = float(np.median(per_var_r2[finite])) if finite.any() else float("nan")
    n_beat_zero = int(np.sum(per_var_r2 > 0))
    n_beat_half = int(np.sum(per_var_r2 > 0.5))

    return dict(
        mean_r2=mean_r2,
        median_r2=median_r2,
        per_variate_r2=per_var_r2.tolist(),
        n_variates=int(V),
        n_variates_beating_zero=n_beat_zero,
        n_variates_beating_half=n_beat_half,
        target=target,
        skipped_variates=skipped_variates,
    )


# ===========================================================================
# Probe 3: Embedding variance + effective-rank diagnostic
# ===========================================================================

def compute_embedding_diagnostics(embeddings: np.ndarray,
                                    label: str = "") -> Dict[str, Any]:
    """Analyze a frozen embedding set for collapse / unused capacity.

    Two things matter:
      (a) per-dimension variance — if most dimensions have near-zero variance,
          the encoder is collapsing signal onto a narrow subspace
      (b) effective rank via eigenvalue entropy — H(λ/Σλ), exponentiated.
          Ranges from 1 (rank-1 collapse) to D (uniform spread). Telling
          indicator of how much of the nominal D-dim capacity is being used.

    Args:
        embeddings: (N, D) frozen embeddings
        label:      human-readable tag for the summary output (e.g. "emb_fine")

    Returns:
        dict with variance summary, effective rank, cumulative variance percentiles.
    """
    if embeddings.ndim != 2:
        return dict(error=f"expected 2D embeddings, got {embeddings.shape}")

    N, D = embeddings.shape
    if N < 2:
        return dict(error=f"too few samples ({N}) for meaningful diagnostic")

    # Per-dimension variance (across samples)
    per_dim_var = embeddings.var(axis=0)  # (D,)

    # Effective rank via SVD of centered embeddings
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    try:
        # Economy SVD — dramatically faster than full SVD for tall matrices
        _, S, _ = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    except np.linalg.LinAlgError as e:
        return dict(error=f"SVD failed: {e}", n_samples=int(N), n_dims=int(D))

    # Eigenvalues of the sample covariance matrix
    eigenvalues = (S ** 2) / max(N - 1, 1)
    total = float(eigenvalues.sum())
    if total < 1e-12:
        return dict(error="all eigenvalues zero",
                    n_samples=int(N), n_dims=int(D))

    probs = eigenvalues / total
    # Shannon entropy of eigenvalue distribution → exp gives effective rank.
    # Roy & Vetterli 2007, "The Effective Rank: A Measure of Effective
    # Dimensionality." Citable, widely-used.
    probs_nz = probs[probs > 1e-12]
    entropy = float(-(probs_nz * np.log(probs_nz)).sum())
    effective_rank = float(np.exp(entropy))

    # Stable rank (Rudelson & Vershynin 2007): ‖A‖_F² / ‖A‖_2² = Σσ² / σ²_max.
    # Disagrees with effective rank when eigenvalues are heavy-tailed; the
    # disagreement itself is informative (AI Engineer M1). Stable rank is
    # dominated by the top eigenvalue; entropy rank weights the tail more.
    stable_rank = float(eigenvalues.sum() / eigenvalues[0]) if eigenvalues[0] > 0 else float("nan")

    # Cumulative variance explained — how many dims account for 90%/99%?
    cumvar = np.cumsum(probs)
    n_dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n_dims_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    return dict(
        label=label,
        n_samples=int(N),
        n_dims=int(D),
        var_mean=float(per_dim_var.mean()),
        var_median=float(np.median(per_dim_var)),
        var_min=float(per_dim_var.min()),
        var_max=float(per_dim_var.max()),
        var_p5=float(np.percentile(per_dim_var, 5)),
        var_p95=float(np.percentile(per_dim_var, 95)),
        var_near_zero_frac=float((per_dim_var < 1e-4).mean()),
        effective_rank=effective_rank,               # Roy & Vetterli 2007
        effective_rank_frac=effective_rank / D,
        stable_rank=stable_rank,                     # Rudelson & Vershynin 2007
        n_dims_for_90pct_variance=n_dims_90,
        n_dims_for_99pct_variance=n_dims_99,
        top_eigenvalue_frac=float(probs[0]),
    )


def format_diagnostic_report(diag: Dict[str, Any]) -> str:
    """Human-readable one-screen summary of embedding diagnostics."""
    if "error" in diag:
        return f"[{diag.get('label', '?')}] ERROR: {diag['error']}"
    return "\n".join([
        f"=== Embedding diagnostics: {diag['label']} ===",
        f"  shape:             ({diag['n_samples']}, {diag['n_dims']})",
        f"  per-dim variance:  mean={diag['var_mean']:.4f}  "
        f"median={diag['var_median']:.4f}  "
        f"min={diag['var_min']:.4f}  max={diag['var_max']:.4f}",
        f"  variance p5/p95:   {diag['var_p5']:.4f} / {diag['var_p95']:.4f}",
        f"  fraction near-zero var (<1e-4): {diag['var_near_zero_frac']:.3f}",
        f"  effective rank:    {diag['effective_rank']:.1f} of {diag['n_dims']}  "
        f"(fraction={diag['effective_rank_frac']:.3f})  [entropy, Roy-Vetterli 2007]",
        f"  stable rank:       {diag.get('stable_rank', float('nan')):.1f}  "
        f"[Rudelson-Vershynin 2007]",
        f"  dims for 90% var:  {diag['n_dims_for_90pct_variance']}",
        f"  dims for 99% var:  {diag['n_dims_for_99pct_variance']}",
        f"  top eigenvalue:    {diag['top_eigenvalue_frac']:.3f} of total "
        f"({'COLLAPSE WARNING' if diag['top_eigenvalue_frac'] > 0.5 else 'OK'})",
    ])


__all__ = [
    "fit_rf_regression_probe", "fit_rf_classification_probe",
    "fit_gbt_regression_probe", "fit_gbt_classification_probe",
    "fit_poly_kernel_ridge_probe",
    "permutation_test_probe_score",
    "fit_frozen_reconstruction_probe",
    "compute_embedding_diagnostics",
    "format_diagnostic_report",
]
