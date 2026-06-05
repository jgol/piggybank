"""Per-typed-vector PCA basis fitting + persistence (v8).

Fits PCA(n_components=5, svd_solver='full', random_state=42) separately
on each of the 7 D84 typed vectors (EMB_SHARED, EMB_GRID, EMB_STRIKE,
EMB_SPX, EMB_VIX, EMB_FLOW_RAW, EMB_FLOW_AGG). Fit is on the training
window ONLY (rows where date ≤ pipeline.TRAIN_END); val/test rows are
NOT seen by PCA.fit.

Sign canonicalization: for each principal component, flip sign so the
largest-absolute-magnitude coefficient is positive. Eliminates the
arbitrary PCA sign ambiguity that otherwise breaks downstream GT/LT
grammar semantics across reruns.

Persistence: writes `layer2/pca_bases.npz` with:
    - One (n_components=5, 384) array per typed vector (7 arrays total)
    - A `_manifest` entry containing fit metadata:
        {group_name: {"n_fit_windows": int,
                      "train_end": str,
                      "explained_variance_ratio": [5 floats],
                      "pca_components_sha256": str}}
    - A `_global_manifest` entry binding the basis to the L1 checkpoint
      + Parquet + train window spec, hashed into a single
      pca_bases_sha256 recorded in provenance.

This module is intentionally side-effect-free (other than disk write).
Loading is done by `layer2/evaluator.py` lazily at first EmbProj
invocation.

Pre-reg v8 compliance:
    - svd_solver='full': no randomness from randomized SVD
    - random_state=42: pinned
    - hard assert on fit window count: prevents test-data leakage
    - sign canonicalization: deterministic across reruns
    - basis SHA bound to L1 checkpoint SHA in _global_manifest
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


# v8 configuration (pre-reg committed values)
N_COMPONENTS = 5                 # v8 legacy default — kept for unit-test fixtures
SVD_SOLVER = "full"              # no randomized-SVD stochasticity
RANDOM_STATE = 42                # pinned for reproducibility
TRAIN_END = "2024-09-30"         # matches pipeline.TRAIN_END
EXPECTED_N_TRAIN_WINDOWS = 6084  # hard-assert guard against test leak

EMB_COLS: List[str] = [
    "EMB_SHARED", "EMB_GRID", "EMB_STRIKE", "EMB_SPX", "EMB_VIX",
    "EMB_FLOW_RAW", "EMB_FLOW_AGG",
    # R2 per-variate embeddings (bypass mean-pool bottleneck)
    "EMB_VAR_ATM_IV", "EMB_VAR_ATM_SPREAD", "EMB_VAR_SPX_RET",
]

# v9: per-group K capped at 15 (= ceil(effective_rank), capped at 15). Locked
# by user 2026-04-25 via
# Effective ranks (eigenvalue-entropy formula on train slice, computed by the
# 2026-04-25 diagnostic agent): VIX=2.84, STRIKE=6.21, FLOW_AGG=13.48,
# SHARED=15.23, SPX=15.30, GRID=18.57, FLOW_RAW=29.88. Three groups (SHARED,
# SPX, GRID) round up beyond 15 and are clipped to 15 to keep the grammar
# tractable. Total EmbProj operators: 84 (was 35 at v8 K=5 flat).
PER_GROUP_K_V9: Dict[str, int] = {
    "EMB_VIX":      3,
    "EMB_STRIKE":   7,
    "EMB_FLOW_AGG": 14,
    "EMB_SHARED":   15,
    "EMB_SPX":      15,
    "EMB_GRID":     15,
    "EMB_FLOW_RAW": 15,
    # R2 per-variate EmbProj terminals (bypass mean-pool bottleneck).
    # K=5 per variate: single variates have less diversity than 88-variate groups.
    "EMB_VAR_ATM_IV":     5,
    "EMB_VAR_ATM_SPREAD": 5,
    "EMB_VAR_SPX_RET":    5,
}
K_SELECTION_RULE_V9: str = "min(15, ceil(effective_rank))"

# Where bases live on disk. Relative to the repo root.
_BASES_PATH = Path(__file__).resolve().parents[2] / "layer2" / "pca_bases.npz"
_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "layer2" / "pca_bases_manifest.json"


def canonicalize_signs(components: np.ndarray) -> np.ndarray:
    """Flip sign of each component so argmax-|coefficient| is positive.

    components: (n_components, n_features). Returns sign-flipped copy.
    Deterministic: for each PC, find the index of the largest-magnitude
    coefficient; if that coefficient is negative, flip the whole PC.
    """
    out = components.copy()
    for k in range(out.shape[0]):
        argmax_abs = int(np.argmax(np.abs(out[k])))
        if out[k, argmax_abs] < 0:
            out[k] = -out[k]
    return out


def _compute_effective_rank(arr: np.ndarray) -> float:
    """Eigenvalue-entropy effective rank of a centered data matrix.

    Defined as exp(H(p)) where p_k = lambda_k / sum(lambda) is the
    normalized eigenvalue spectrum of the sample covariance and H is
    Shannon entropy in nats. Roy & Vetterli (2007). For full PCA on
    arr (N, V), all V eigenvalues are observed, so the effective
    rank is well-defined without needing to choose a truncation K.

    Returns a float in [1.0, V].
    """
    centered = arr - arr.mean(axis=0, keepdims=True)
    # Use SVD to get singular values; squared singular values / (N-1)
    # are the eigenvalues of the covariance matrix.
    s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigs = (s ** 2) / max(centered.shape[0] - 1, 1)
    eigs = eigs[eigs > 0]
    if len(eigs) == 0:
        return 0.0
    p = eigs / eigs.sum()
    # Shannon entropy in nats; effective rank = exp(H).
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def fit_pca_bases(parquet_path: str,
                  n_components: int = N_COMPONENTS,
                  train_end: str = TRAIN_END,
                  require_n_train: int = EXPECTED_N_TRAIN_WINDOWS,
                  per_group_k: Optional[Dict[str, int]] = None,
                  ) -> Dict:
    """Fit per-typed-vector PCA on the L1 Parquet's train slice.

    Args:
        per_group_k: optional dict mapping group name -> K. If provided,
            REPLACES the flat `n_components` for each group. Per v9 spec,
            production callers MUST pass `PER_GROUP_K_V9`. The `n_components`
            arg is retained as a fallback for legacy unit tests that rely on
            the flat K=5 behavior.

    Returns a dict with, per typed vector:
        {"components": (K_group, 384),       # sign-canonicalized
         "explained_variance_ratio": (K_group,),
         "explained_variance_": (K_group,),
         "mean_": (384,),                    # for transform consistency
         "n_features_in_": int,
         "k_used": int,                      # K applied for this group
         "effective_rank": float}            # eigenvalue-entropy rank

    plus global keys:
        "_train_end": str
        "_n_fit_windows": int
        "_n_components": int                 # legacy flat K (or 'per_group')
        "_per_group_k": Dict[str, int] or None
        "_k_selection_rule": str
        "_svd_solver": str
        "_random_state": int
    """
    df = pd.read_parquet(parquet_path)
    train_mask = df["date"] <= train_end
    n_train = int(train_mask.sum())

    # HARD ASSERT: prevents accidental val/test leak via wrong mask.
    # Per pre-reg v8, fit window count MUST equal EXPECTED_N_TRAIN_WINDOWS.
    # If this fails, either TRAIN_END has shifted or the Parquet regeneration
    # introduced an off-by-N issue — both worth investigating before PCA fit.
    if n_train != require_n_train:
        raise RuntimeError(
            f"PCA fit would see n_train={n_train} windows, but pre-reg v8 "
            f"commits to {require_n_train} (train_end={train_end}). "
            f"Investigate: either the Parquet date coverage changed or the "
            f"train_end constant is stale. Refusing to fit on unexpected "
            f"train slice to prevent silent test-data leak."
        )

    # Only fit columns that exist in the Parquet. R2 per-variate columns
    # (EMB_VAR_*) may not be present in older Parquets.
    _cols_to_fit = [c for c in EMB_COLS if c in df.columns]

    if per_group_k is not None:
        missing = [c for c in _cols_to_fit if c not in per_group_k]
        if missing:
            raise ValueError(
                f"per_group_k missing keys: {missing} (got {sorted(per_group_k)})"
            )
        for c, k in per_group_k.items():
            if k > 15:
                raise ValueError(
                    f"per_group_k[{c!r}]={k} exceeds the v9-locked cap of 15"
                )
            if k < 1:
                raise ValueError(
                    f"per_group_k[{c!r}]={k} must be >= 1"
                )

    result: Dict = {
        "_train_end": train_end,
        "_n_fit_windows": n_train,
        "_n_components": "per_group" if per_group_k is not None else n_components,
        "_per_group_k": dict(per_group_k) if per_group_k is not None else None,
        "_k_selection_rule": K_SELECTION_RULE_V9 if per_group_k is not None else "flat",
        "_svd_solver": SVD_SOLVER,
        "_random_state": RANDOM_STATE,
    }

    for col in _cols_to_fit:
        # Stack list-of-floats column into (N, 384) array
        arr_all = np.stack(df[col].tolist()).astype(np.float64)
        arr_train = arr_all[train_mask.values]
        # Effective rank — recorded in manifest for audit.
        eff_rank = _compute_effective_rank(arr_train)
        k_for_group = (per_group_k[col] if per_group_k is not None
                       else n_components)
        pca = PCA(n_components=k_for_group,
                  svd_solver=SVD_SOLVER,
                  random_state=RANDOM_STATE)
        pca.fit(arr_train)
        components = canonicalize_signs(pca.components_)
        # v9 P0-B: persist explained_variance_ so the evaluator can
        # Mahalanobis-normalize EmbProj outputs (divide projection by
        # sqrt(explained_variance_[k])). This makes every EmbProj_*_k
        # operator unit-variance on the train slice, removing the
        # 5.8x–64.8x scale dominance over bypass scalars that broke
        # mixed `GT/LT(EmbProj_*, scalar)` comparators in v8.
        ev = pca.explained_variance_.astype(np.float64)
        # Loud failure: a zero/NaN explained_variance_ would make the
        # downstream sqrt-divide explode. Per project policy (RF-1 hard
        # lesson), we raise rather than silently clamp.
        if not np.all(np.isfinite(ev)) or np.any(ev <= 0.0):
            raise RuntimeError(
                f"PCA explained_variance_ for group {col!r} is invalid: "
                f"{ev.tolist()}. Refusing to fit a degenerate basis."
            )
        result[col] = {
            "components": components.astype(np.float32),   # (K, 384)
            "mean_": pca.mean_.astype(np.float32),          # (384,)
            # v9 P0-B addition — needed by evaluator for Mahalanobis scale.
            "explained_variance_": ev.astype(np.float32),   # (K,)
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_features_in_": int(pca.n_features_in_),
            # v9 per-group K addition: record the K used and the eigenvalue-
            # entropy effective rank for audit. Manifest writes both.
            "k_used": int(k_for_group),
            "effective_rank": float(eff_rank),
        }
    return result


# ---------------------------------------------------------------------------
# PLS2 + Residual PCA hybrid fitting (supervised basis extraction)
# ---------------------------------------------------------------------------

# Number of PLS directions per group. PLS2 fits against a multi-target Y
# of dimensionality 7: [next_bar_return, mfe_5, mae_5, iv_change_5, regime_oh(3)].
_N_PLS_DIRECTIONS = 7


def fit_supervised_bases(parquet_path: str,
                         targets_path: str,
                         train_end: str = TRAIN_END,
                         require_n_train: int = EXPECTED_N_TRAIN_WINDOWS,
                         per_group_k: Optional[Dict[str, int]] = None,
                         ) -> Dict:
    """Fit per-typed-vector PLS2 + residual PCA on the L1 Parquet's train slice.

    Supervised hybrid: PLS2 extracts directions maximally covariant with
    GP-relevant targets (next_bar_return, MFE-5, MAE-5, IV_change_5,
    regime 3-class one-hot), then standard PCA is applied to the PLS
    residual to capture remaining variance directions. The output format
    is identical to fit_pca_bases() so load_bases()/project() work
    without modification.

    Args:
        parquet_path: path to L1 typed-vector Parquet (same as fit_pca_bases)
        targets_path: path to a numpy .npz file containing:
            - 'targets': (N_total, 7) float32 array aligned row-by-row with
              the Parquet. Columns: [next_bar_return, mfe_5, mae_5,
              iv_change_5, regime_oh_0, regime_oh_1, regime_oh_2]
            - 'valid': (N_total,) bool — rows with valid targets (no NaN)
        train_end: training window cutoff date (YYYY-MM-DD)
        require_n_train: expected number of training windows (hard assertion)
        per_group_k: per-group component budget (MUST be PER_GROUP_K_V9 for
            production). Groups with K <= _N_PLS_DIRECTIONS use ALL K as PLS
            (no residual PCA). Groups with K > _N_PLS_DIRECTIONS use
            _N_PLS_DIRECTIONS PLS + (K - _N_PLS_DIRECTIONS) residual PCA.

    Returns:
        Dict with the SAME structure as fit_pca_bases() — drop-in replacement.
        The "components" for each group are (K, 384) orthogonalized directions:
            - First min(K, 7) rows: PLS x-weight directions (orthogonalized)
            - Remaining (K - 7) rows: top PCA directions of the PLS residual
        "explained_variance_" is replaced by per-direction std of training
        projections (same semantic role: Mahalanobis normalization in evaluator).
    """
    df = pd.read_parquet(parquet_path)
    train_mask = df["date"] <= train_end
    n_train = int(train_mask.sum())

    if n_train != require_n_train:
        raise RuntimeError(
            f"PLS fit would see n_train={n_train} windows, but pre-reg "
            f"commits to {require_n_train} (train_end={train_end}). "
            f"Investigate: either the Parquet date coverage changed or the "
            f"train_end constant is stale. Refusing to fit on unexpected "
            f"train slice to prevent silent test-data leak."
        )

    # Only fit columns present in the Parquet; R2 per-variate may be absent.
    _cols_to_fit_pls = [c for c in EMB_COLS if c in df.columns]

    if per_group_k is not None:
        missing = [c for c in _cols_to_fit_pls if c not in per_group_k]
        if missing:
            raise ValueError(
                f"per_group_k missing keys: {missing} (got {sorted(per_group_k)})"
            )
        for c, k in per_group_k.items():
            if k > 15:
                raise ValueError(
                    f"per_group_k[{c!r}]={k} exceeds the v9-locked cap of 15"
                )
            if k < 1:
                raise ValueError(
                    f"per_group_k[{c!r}]={k} must be >= 1"
                )

    # Load targets
    tgt_data = np.load(targets_path, allow_pickle=False)
    targets_all = tgt_data["targets"]  # (N_total, 7)
    targets_valid = tgt_data["valid"]  # (N_total,) bool

    if targets_all.shape[0] != len(df):
        raise RuntimeError(
            f"Targets array has {targets_all.shape[0]} rows but Parquet has "
            f"{len(df)} rows. They must be aligned row-by-row."
        )
    # Allow variable target count (4 without regime, 7 with regime).
    # PLS n_components is capped at min(K, n_targets) per group.
    _n_pls_actual = targets_all.shape[1]
    if _n_pls_actual < 4:
        raise RuntimeError(
            f"Targets array has {_n_pls_actual} columns, need at least 4 "
            f"(next_bar_ret, mfe_5, mae_5, iv_change_5)."
        )

    # Build joint training mask: in-train AND targets valid (no NaN rows)
    train_idx = train_mask.values & targets_valid
    n_pls_fit = int(train_idx.sum())
    print(f"  PLS fit rows: {n_pls_fit}/{n_train} "
          f"({100*n_pls_fit/max(n_train,1):.1f}% of train have valid targets)")
    if n_pls_fit < 100:
        raise RuntimeError(
            f"Only {n_pls_fit} valid PLS training rows — too few for stable fit. "
            f"Check targets_path for excessive NaN coverage."
        )

    result: Dict = {
        "_train_end": train_end,
        "_n_fit_windows": n_train,
        "_n_pls_fit_windows": n_pls_fit,
        "_n_components": "per_group" if per_group_k is not None else N_COMPONENTS,
        "_per_group_k": dict(per_group_k) if per_group_k is not None else None,
        "_k_selection_rule": K_SELECTION_RULE_V9 if per_group_k is not None else "flat",
        "_svd_solver": SVD_SOLVER,
        "_random_state": RANDOM_STATE,
        "_method": "pls",
        "_n_pls_directions": _n_pls_actual,
    }

    Y_train = targets_all[train_idx].astype(np.float64)  # (n_pls_fit, 7)

    for col in _cols_to_fit_pls:
        # Stack list-of-floats column into (N, 384) array
        arr_all = np.stack(df[col].tolist()).astype(np.float64)
        arr_train_full = arr_all[train_mask.values]       # full training slice
        arr_train_pls = arr_all[train_idx]                 # PLS-valid subset

        # NaN guard: encoder producing NaN embeddings indicates a catastrophic
        # training failure (gradient explosion, loss divergence, etc.)
        if np.any(np.isnan(arr_train_full)):
            raise RuntimeError(f"NaN detected in {col} training embeddings. Cannot fit PLS basis.")

        # Effective rank — recorded in manifest for audit.
        eff_rank = _compute_effective_rank(arr_train_full)

        k_for_group = (per_group_k[col] if per_group_k is not None
                       else N_COMPONENTS)

        # Determine PLS vs residual PCA split
        n_pls = min(k_for_group, _n_pls_actual)
        n_residual_pca = k_for_group - n_pls

        # --- Step 1: Center embedding using the FULL training slice mean ---
        # (Consistent with PCA path: mean is computed on all training rows,
        # not just the PLS-valid subset, so that project() uses the same mean.)
        train_mean = arr_train_full.mean(axis=0)
        X_centered_pls = arr_train_pls - train_mean  # PLS fitting input

        # Diagnostic: double-centering check. The PLS subset (target-valid
        # training rows) may differ systematically from the full training
        # slice. A large centroid shift means PLS directions are biased toward
        # the valid-target subpopulation. Warn if shift exceeds 0.1 L2 norm.
        subset_mean_norm = np.linalg.norm(X_centered_pls.mean(axis=0))
        if subset_mean_norm > 0.1:
            print(f"  WARNING: {col} PLS subset centroid shift = {subset_mean_norm:.4f} "
                  f"(target-valid subset mean differs from full training mean)")

        # --- Step 2: Fit PLS2 ---
        # sklearn PLSRegression n_components = number of latent directions.
        # We request n_pls directions from the multi-target regression.
        pls = PLSRegression(n_components=n_pls, scale=False, max_iter=500)
        pls.fit(X_centered_pls, Y_train)

        # PLS x-weights (directions in X-space): pls.x_rotations_ is (384, n_pls)
        # Each column is a direction that maximizes covariance with Y.
        # We want them as row vectors for consistency with PCA components format.
        pls_directions = pls.x_rotations_.T.copy()  # (n_pls, 384)

        # --- Step 3: Orthogonalize PLS directions via QR ---
        # PLS x_rotations are NOT necessarily orthogonal. QR decomposition
        # gives us an orthonormal basis spanning the same subspace.
        Q, R = np.linalg.qr(pls_directions.T, mode="reduced")  # Q: (384, n_pls)
        pls_ortho = Q.T  # (n_pls, 384) — orthonormal rows

        # --- Step 4: Residual PCA (if K > n_pls) ---
        if n_residual_pca > 0:
            # Project out the PLS subspace from the full training data
            X_centered_full = arr_train_full - train_mean
            # Projection onto PLS subspace: X @ Q @ Q.T
            # Residual: X - X @ Q @ Q.T
            projection_onto_pls = X_centered_full @ Q @ Q.T  # (N_train, 384)
            residual = X_centered_full - projection_onto_pls  # (N_train, 384)

            # Fit PCA on residual
            pca_resid = PCA(n_components=n_residual_pca,
                            svd_solver=SVD_SOLVER,
                            random_state=RANDOM_STATE)
            pca_resid.fit(residual)
            resid_components = pca_resid.components_  # (n_residual_pca, 384)

            # Stack: PLS directions first, then residual PCA
            combined_components = np.vstack([pls_ortho, resid_components])
        else:
            combined_components = pls_ortho

        # --- Step 5: Sign canonicalization ---
        combined_components = canonicalize_signs(combined_components)

        # --- Step 6: Compute per-direction std on training projections ---
        # This replaces explained_variance_ for Mahalanobis normalization.
        # Project ALL training rows (full train slice) onto the combined basis.
        X_centered_full = arr_train_full - train_mean
        train_projections = X_centered_full @ combined_components.T  # (N_train, K)
        direction_std = train_projections.std(axis=0)  # (K,)

        # Validate: no degenerate (zero/NaN) std values
        if not np.all(np.isfinite(direction_std)) or np.any(direction_std <= 0.0):
            raise RuntimeError(
                f"PLS+PCA direction std for group {col!r} is invalid: "
                f"{direction_std.tolist()}. Refusing to fit a degenerate basis."
            )

        # Near-zero std floor check: catches rank-deficient embeddings that
        # produce near-constant projections on some direction. These would
        # cause extreme Mahalanobis scaling (dividing by ~0) downstream.
        if np.any(direction_std < 1e-6):
            degenerate = np.where(direction_std < 1e-6)[0]
            raise RuntimeError(
                f"PLS basis for {col} has near-zero std on directions {degenerate.tolist()}. "
                f"This indicates a rank-deficient embedding. Check encoder quality."
            )

        # The evaluator uses `pc_std_` which is sqrt(explained_variance_ + eps).
        # We store direction_std**2 as "explained_variance_" so that the
        # existing load_bases() path computes sqrt(std^2 + eps) ~ std, which
        # is exactly what we want for Mahalanobis normalization.
        pseudo_explained_variance = (direction_std ** 2).astype(np.float64)

        result[col] = {
            "components": combined_components.astype(np.float32),  # (K, 384)
            "mean_": train_mean.astype(np.float32),                # (384,)
            "explained_variance_": pseudo_explained_variance.astype(np.float32),  # (K,)
            "explained_variance_ratio": [],  # not meaningful for PLS; empty list
            "n_features_in_": 384,
            "k_used": int(k_for_group),
            "effective_rank": float(eff_rank),
            "n_pls_directions": int(n_pls),
            "n_residual_pca": int(n_residual_pca),
        }

    return result


def save_bases(bases: Dict, out_path: Path = _BASES_PATH,
               manifest_path: Path = _MANIFEST_PATH,
               l1_checkpoint_path: Optional[str] = None) -> Tuple[str, str]:
    """Persist bases to npz + sidecar manifest. Returns (npz_sha256, manifest_json).

    The npz file contains per-group arrays (components + mean) but NO manifest
    metadata (npz doesn't play well with dict-of-dict structure). The manifest
    JSON carries the metadata, referencing the npz's SHA.

    If `l1_checkpoint_path` is provided, the manifest records the L1 checkpoint
    SHA256 so grammar evaluators can verify the basis was fit against the
    current encoder (drift guard).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build arrays for npz: per-group × 3 arrays (components, mean, explained_variance)
    # v9 P0-B: explained_variance_ added so the evaluator can Mahalanobis-
    # normalize EmbProj outputs.
    # Only save groups that were actually fit (R2 per-variate groups may be absent).
    _cols_present = [c for c in EMB_COLS if c in bases]
    np_save: Dict[str, np.ndarray] = {}
    for col in _cols_present:
        g = bases[col]
        np_save[f"{col}_components"] = g["components"]
        np_save[f"{col}_mean"] = g["mean_"]
        np_save[f"{col}_explained_variance"] = g["explained_variance_"]

    import os
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    # np.savez auto-appends .npz to string paths; pass a file handle
    # to avoid the double-suffix surprise.
    with tmp.open("wb") as _fp:
        np.savez(_fp, **np_save)
    os.replace(tmp, out_path)

    # Compute npz SHA
    with out_path.open("rb") as f:
        npz_sha = hashlib.sha256(f.read()).hexdigest()

    # Build manifest
    method = bases.get("_method", "pca")
    manifest = {
        "pca_bases_sha256":        npz_sha,
        "pca_bases_path":          str(out_path.relative_to(out_path.parents[1])),
        "method":                  method,
        "n_components":            bases["_n_components"],
        # v9 per-group K: explicit serialization for audit + downstream
        # cache-key keying. None for legacy flat-K bases.
        "per_group_k":             bases.get("_per_group_k"),
        "k_selection_rule":        bases.get("_k_selection_rule", "flat"),
        "svd_solver":              bases["_svd_solver"],
        "random_state":            bases["_random_state"],
        "train_end":               bases["_train_end"],
        "n_fit_windows":           bases["_n_fit_windows"],
        "emb_cols":                list(_cols_present),
        "per_group": {
            col: {
                "explained_variance_ratio": bases[col]["explained_variance_ratio"],
                "n_features_in":            bases[col]["n_features_in_"],
                "components_shape":          list(bases[col]["components"].shape),
                # v9: per-group K used + effective rank used for K choice.
                "k_used":                   bases[col].get("k_used"),
                "effective_rank":           bases[col].get("effective_rank"),
                # PLS-specific: number of supervised vs unsupervised directions
                "n_pls_directions":         bases[col].get("n_pls_directions"),
                "n_residual_pca":           bases[col].get("n_residual_pca"),
            }
            for col in _cols_present
        },
        "effective_rank_per_group": {
            col: bases[col].get("effective_rank") for col in _cols_present
        },
    }
    # PLS-specific manifest entries
    if method == "pls":
        manifest["n_pls_directions"] = bases.get("_n_pls_directions", _N_PLS_DIRECTIONS)
        manifest["n_pls_fit_windows"] = bases.get("_n_pls_fit_windows")
    if l1_checkpoint_path is not None:
        try:
            with Path(l1_checkpoint_path).open("rb") as f:
                manifest["l1_checkpoint_sha256"] = hashlib.sha256(f.read()).hexdigest()
            manifest["l1_checkpoint_path"] = str(l1_checkpoint_path)
        except Exception as exc:
            manifest["l1_checkpoint_read_error"] = repr(exc)

    m_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    m_tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(m_tmp, manifest_path)

    return npz_sha, json.dumps(manifest, indent=2)


def load_bases(path: Path = _BASES_PATH,
               manifest_path: Path = _MANIFEST_PATH) -> Dict:
    """Load bases from disk + verify manifest integrity. Returns
    {group_name: {"components": (K, 384), "mean_": (384,)}}
    plus manifest dict at key "_manifest".
    """
    data = np.load(path, allow_pickle=False)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Verify npz sha matches manifest (detects disk bit-flip / wrong-version
    # basis being loaded).
    import hashlib as _hashlib
    with path.open("rb") as f:
        actual_sha = _hashlib.sha256(f.read()).hexdigest()
    expected_sha = manifest.get("pca_bases_sha256", "")
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"PCA basis SHA mismatch:\n  manifest: {expected_sha}\n  actual:   {actual_sha}\n"
            f"The pca_bases.npz file has changed since manifest was written. "
            f"Re-fit via layer2.inference.pca_bases.main() before continuing."
        )
    out: Dict[str, Dict] = {"_manifest": manifest}
    # R2 per-variate types are optional — their PCA bases are fit separately
    # and may not exist yet in the NPZ. The evaluator's EmbProj dispatch
    # returns 0.0 for groups not in `bases`, so missing per-variate entries
    # are safe to skip. The 7 original group-pooled types remain mandatory.
    _OPTIONAL_GROUPS = frozenset([
        "EMB_VAR_ATM_IV", "EMB_VAR_ATM_SPREAD", "EMB_VAR_SPX_RET",
    ])
    for col in EMB_COLS:
        # v9 P0-B: explained_variance_ is required by the evaluator's
        # Mahalanobis-normalized EmbProj path. If a legacy npz lacks the
        # key, refuse to load — silent zero-fill would re-introduce the
        # 5.8x-64.8x scale-mismatch pathology v9 was meant to fix.
        ev_key = f"{col}_explained_variance"
        if ev_key not in data.files:
            if col in _OPTIONAL_GROUPS:
                # Per-variate bases not yet fit — skip gracefully.
                # The evaluator returns 0.0 for missing groups.
                continue
            raise RuntimeError(
                f"pca_bases.npz at {path} is missing key {ev_key!r}. "
                f"This npz was fit by a pre-v9 generator (no Mahalanobis "
                f"normalization). Re-fit via "
                f"`python -m layer2.inference.pca_bases` to regenerate "
                f"with v9 P0-B explained_variance_ included."
            )
        ev = data[ev_key]
        if not np.all(np.isfinite(ev)) or np.any(ev <= 0.0):
            raise RuntimeError(
                f"explained_variance_ for group {col!r} is invalid: "
                f"{ev.tolist()}. Refusing to load a degenerate basis."
            )
        out[col] = {
            "components": data[f"{col}_components"],     # (K, 384)
            "mean_":      data[f"{col}_mean"],            # (384,)
            "explained_variance_": ev,                    # (K,)
            # v9 P0-B: pre-computed scaling factor for EmbProj
            # Mahalanobis normalization. Caching here (not in evaluator)
            # avoids a per-call sqrt + add. epsilon is for numerical
            # safety only — the load-side validation above already
            # rejects ev <= 0.
            "pc_std_": np.sqrt(ev + 1e-8).astype(np.float32),
        }
    return out


def project(emb_vector: np.ndarray, bases_group: Dict) -> np.ndarray:
    """Project a single 384-d typed vector onto the group's PC basis.

    emb_vector: (384,)
    bases_group: {"components": (K, 384), "mean_": (384,)}
    Returns: (K,) scalar projections.

    Mirrors sklearn.PCA.transform: subtract mean, then project onto components.
    """
    centered = emb_vector - bases_group["mean_"]
    return centered @ bases_group["components"].T  # (K,)


def main() -> int:
    """CLI entry: fit projection bases on raw_data/local_store/l1_pilot.parquet
    and persist to layer2/pca_bases.npz.

    Modes:
        --mode pca   (default) Pure PCA, v9 per-group K.
        --mode pls   PLS2 + residual PCA hybrid (supervised directions).

    Examples:
        python -m layer2.inference.pca_bases                         # v9 PCA (default)
        python -m layer2.inference.pca_bases --mode pca              # explicit PCA
        python -m layer2.inference.pca_bases --mode pls --targets targets.npz
        python -m layer2.inference.pca_bases --legacy-flat-k         # v8-style flat K=5
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pca", "pls"], default="pca",
                    help="Projection method: 'pca' (default) or 'pls' (PLS2+residual PCA)")
    ap.add_argument("--parquet",
                    default="raw_data/local_store/l1_pilot.parquet")
    ap.add_argument("--checkpoint",
                    default="raw_data/local_store/ssl_model_v3")
    ap.add_argument("--targets",
                    default="raw_data/local_store/pls_targets.npz",
                    help="Path to targets .npz for PLS mode (ignored in PCA mode)")
    ap.add_argument("--legacy-flat-k", action="store_true",
                    help="Fit with flat K=5 (v8 default) instead of v9 per-group K.")
    args = ap.parse_args()

    if args.legacy_flat_k:
        per_group_k: Optional[Dict[str, int]] = None
    else:
        per_group_k = dict(PER_GROUP_K_V9)

    if args.mode == "pls":
        # --- PLS2 + Residual PCA path ---
        print(f"Fitting per-typed-vector PLS2+ResidualPCA "
              f"(per_group_k={per_group_k}, n_pls_directions={_N_PLS_DIRECTIONS}) [SUPERVISED]")
        print(f"  L1 Parquet: {args.parquet}")
        print(f"  Targets:    {args.targets}")
        print(f"  Train end:  {TRAIN_END}")
        print(f"  Expected N train: {EXPECTED_N_TRAIN_WINDOWS}")

        bases = fit_supervised_bases(
            args.parquet,
            targets_path=args.targets,
            per_group_k=per_group_k,
        )
        print()
        print("Per-typed-vector PLS+PCA basis summary:")
        header = f"{'group':<15}  {'K':>3}  {'n_pls':>5}  {'n_resid':>7}  {'eff_rank':>9}"
        print(header)
        print("-" * len(header))
        for col in EMB_COLS:
            if col not in bases:
                continue
            k = bases[col].get("k_used", 0)
            n_pls = bases[col].get("n_pls_directions", 0)
            n_resid = bases[col].get("n_residual_pca", 0)
            eff = bases[col].get("effective_rank", float("nan"))
            print(f"{col:<15}  {k:>3}  {n_pls:>5}  {n_resid:>7}  {eff:>9.2f}")
        print()

    else:
        # --- Pure PCA path (unchanged behavior) ---
        if args.legacy_flat_k:
            print(f"Fitting per-typed-vector PCA(n={N_COMPONENTS}, svd_solver={SVD_SOLVER}, "
                  f"random_state={RANDOM_STATE}) [LEGACY FLAT-K]")
        else:
            print(f"Fitting per-typed-vector PCA(per_group_k={per_group_k}, "
                  f"svd_solver={SVD_SOLVER}, random_state={RANDOM_STATE}) [v9]")
        print(f"  L1 Parquet: {args.parquet}")
        print(f"  Train end:  {TRAIN_END}")
        print(f"  Expected N train: {EXPECTED_N_TRAIN_WINDOWS}")

        bases = fit_pca_bases(args.parquet, per_group_k=per_group_k)
        print()
        print("Per-typed-vector cumulative explained variance ratios:")
        header = f"{'group':<15}  {'K':>3}  {'eff_rank':>9}  {'cum_var':>8}"
        print(header)
        print("-" * len(header))
        for col in EMB_COLS:
            if col not in bases:
                continue
            ev = bases[col]["explained_variance_ratio"]
            k = bases[col].get("k_used", len(ev))
            eff = bases[col].get("effective_rank", float("nan"))
            print(f"{col:<15}  {k:>3}  {eff:>9.2f}  {sum(ev):>7.1%}")
        print()

    npz_sha, _ = save_bases(bases, l1_checkpoint_path=args.checkpoint)
    total_components = sum(int(bases[c]["components"].shape[0]) for c in EMB_COLS if c in bases)
    print(f"Saved pca_bases.npz  sha256: {npz_sha[:16]}...  "
          f"total_components={total_components}")
    print(f"Manifest: layer2/pca_bases_manifest.json")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
