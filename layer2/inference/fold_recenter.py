"""v9 walk-forward fold-fixed per-PC re-centering for EmbProj outputs.

Purpose
-------
The L1 SSL encoder is frozen on the train window. Typed-vector centroids
drift substantially from train→val and train→test (RF-A: per-group PC0
mean shifts of 2.85-6.19 sigma, see (internal doc) v8.1). The v9
P0-B Mahalanobis normalization fixes the *scale* component of that drift
on the train slice, but leaves the *mean* component unaddressed on
val/test. Result: post-Mahalanobis EmbProj outputs on val/test no longer
have unit mean, breaking `GT/LT(EmbProj_*, scalar)` saturation as
strategies cross from train into val/test windows.

This module fits a per-fold per-(group, PC) StandardScaler over the
post-Mahalanobis projection space and persists the stats so the
evaluator can apply `(proj - fold_mean) / fold_std` at evaluation time.

No-leak guarantee
-----------------
Walk-forward fold-fixed: for each non-train fold F, the StandardScaler
is fit on rows from ALL folds STRICTLY PRECEDING F. Train fold gets
identity (mean=0 std=1) by PCA construction. Val fold uses train-only.
Test fold uses train+val. The fold-membership of each fit row is
asserted at fit time; a violation raises RuntimeError loudly.

Persisted artifacts
-------------------
- layer2/fold_recenter_stats.npz — keys "{fold_id}/{group}/mean" and
  "{fold_id}/{group}/std", each a (K_group,) array.
- layer2/fold_recenter_manifest.json — fold definitions, no-leak protocol,
  sha-tied to L1 parquet sha + PCA bases sha.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from layer2.inference.pca_bases import EMB_COLS, _BASES_PATH, _MANIFEST_PATH


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = _REPO_ROOT / "layer2" / "fold_recenter_stats.npz"
STATS_MANIFEST_PATH = _REPO_ROOT / "layer2" / "fold_recenter_manifest.json"


# -----------------------------------------------------------------------------
# Fold definitions (matches H2 pre-reg + experiment.py defaults)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldDef:
    """Inclusive [date_start, date_end] window for a fold."""
    fold_id: str
    date_start: str   # YYYY-MM-DD
    date_end: str     # YYYY-MM-DD


# ⚠ IH1 (2026-06-01 holistic review): this 3-fold train/val/test scheme (single
# 2024-09-30 split) does NOT match the GP's 4-fold ROLLING walk-forward
# (experiment.py WALK_FORWARD_FOLDS: train_ends 2024-03-29 / 2024-09-27 / 2025-03-31
# / 2025-09-30, keyed by INTEGER fold id "1".."4"). Under walk-forward, a fold's
# evaluation looks up its integer fold_id in the recenter stats; when the stats are
# keyed "train"/"val"/"test" the key is ABSENT, so the per-PC drift correction was
# silently skipped (now FAIL-LOUD in evaluator_vectorized.py). BEFORE running
# Conditions C/D under walk-forward, regenerate fold_recenter_stats.npz keyed to the
# SAME integer walk-forward folds. This 3-fold default is valid ONLY for the legacy
# single-split C/D evaluation.
DEFAULT_FOLD_DEFINITIONS: Tuple[FoldDef, ...] = (
    FoldDef(fold_id="train", date_start="2022-09-19", date_end="2024-09-30"),
    FoldDef(fold_id="val",   date_start="2024-10-01", date_end="2025-01-31"),
    FoldDef(fold_id="test",  date_start="2025-02-01", date_end="2026-04-09"),
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def assign_fold_id(date: str, folds: Tuple[FoldDef, ...]) -> Optional[str]:
    """Return fold_id for `date` (YYYY-MM-DD), or None if outside all folds.

    The fold ranges are inclusive on both ends; embargo days between folds
    fall in NEITHER fold and return None — but the H2 pre-registered
    fold definitions are CONTIGUOUS (train ends 2024-09-30, val starts
    2024-10-01) so no gap exists in default operation.
    """
    for fd in folds:
        if fd.date_start <= date <= fd.date_end:
            return fd.fold_id
    return None


def _project_post_mahalanobis(arr: np.ndarray,
                               group_basis: Dict[str, np.ndarray]
                               ) -> np.ndarray:
    """Compute (N, K) post-Mahalanobis projections.

    arr: (N, V) typed-vector matrix (V=384).
    group_basis: dict with "components" (K, V), "mean_" (V,), "pc_std_" (K,).

    Returns (N, K) array of post-Mahalanobis projections (matches the
    evaluator's EmbProj output before any fold-recenter).
    """
    centered = arr - group_basis["mean_"][None, :]
    raw = centered @ group_basis["components"].T          # (N, K)
    return raw / group_basis["pc_std_"][None, :]


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------

def fit_fold_recenter_stats(
    parquet_path: str,
    pca_bases_path: Path = _BASES_PATH,
    fold_definitions: Tuple[FoldDef, ...] = DEFAULT_FOLD_DEFINITIONS,
    min_fit_rows: int = 200,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Fit walk-forward fold-fixed per-PC StandardScaler stats.

    For each non-train fold F:
        Use rows strictly preceding F (i.e. rows belonging to folds whose
        date ranges all end before F's date_start). For F=val, this is
        train. For F=test, this is train + val.

    For train fold:
        Identity transform (mean=0, std=1) by PCA construction. Recorded
        explicitly so downstream cache keys are uniform across folds.

    Returns dict[fold_id][group] = {"mean": (K,), "std": (K,)}.
    """
    from layer2.inference.pca_bases import load_bases

    df = pd.read_parquet(parquet_path)
    if "date" not in df.columns:
        raise RuntimeError(
            f"L1 parquet at {parquet_path} missing 'date' column — cannot "
            f"assign fold IDs."
        )
    df["date"] = df["date"].astype(str)
    # Assign fold_id per row.
    df["__fold_id"] = df["date"].map(
        lambda d: assign_fold_id(d, fold_definitions)
    )
    n_unassigned = int(df["__fold_id"].isna().sum())
    if n_unassigned > 0:
        # Fail loudly: silent drops would break the no-leak audit.
        sample = df.loc[df["__fold_id"].isna(), "date"].head(5).tolist()
        raise RuntimeError(
            f"{n_unassigned} parquet rows have dates outside all fold "
            f"definitions: examples={sample}. Either widen fold_definitions "
            f"or amend the parquet."
        )

    bases = load_bases(pca_bases_path, _MANIFEST_PATH)
    fold_ids = [fd.fold_id for fd in fold_definitions]
    train_id = fold_ids[0]  # train is the first fold by convention

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    # Train fold: identity transform.
    out[train_id] = {}
    for col in EMB_COLS:
        K = int(bases[col]["components"].shape[0])
        out[train_id][col] = {
            "mean": np.zeros(K, dtype=np.float64),
            "std":  np.ones(K, dtype=np.float64),
        }

    # Non-train folds: fit on STRICTLY-PRECEDING folds.
    for i, fd in enumerate(fold_definitions):
        if fd.fold_id == train_id:
            continue
        preceding_ids = [f.fold_id for f in fold_definitions[:i]]
        # Loud failure: must have at least one preceding fold.
        if not preceding_ids:
            raise RuntimeError(
                f"fold {fd.fold_id!r} has no preceding folds; cannot fit "
                f"recenter stats walk-forward."
            )
        fit_mask = df["__fold_id"].isin(preceding_ids)
        n_fit = int(fit_mask.sum())
        if n_fit < min_fit_rows:
            raise RuntimeError(
                f"fold {fd.fold_id!r} has only {n_fit} preceding-fold rows "
                f"(< {min_fit_rows} required) for fit. Investigate parquet "
                f"date coverage."
            )
        # No-leak audit: assert every fit row's date is strictly before F.start.
        bad = df.loc[fit_mask & (df["date"] >= fd.date_start), "date"]
        if len(bad) > 0:
            raise RuntimeError(
                f"no-leak violation: {len(bad)} fit rows for fold "
                f"{fd.fold_id!r} have dates >= {fd.date_start} (its start). "
                f"Examples: {bad.head(3).tolist()}"
            )
        out[fd.fold_id] = {}
        for col in EMB_COLS:
            arr = np.stack(df.loc[fit_mask, col].tolist()).astype(np.float64)
            proj = _project_post_mahalanobis(arr, bases[col])  # (n_fit, K)
            mean = proj.mean(axis=0).astype(np.float64)
            std = proj.std(axis=0, ddof=1).astype(np.float64)
            # Numerical floor on std: post-Mahalanobis std should be ~1.0
            # on train; on later folds it lands ~0.7-1.3. Anything below
            # 1e-6 is degenerate and should fail loudly.
            if np.any(std < 1e-6):
                bad_pcs = np.where(std < 1e-6)[0].tolist()
                raise RuntimeError(
                    f"degenerate std (<1e-6) on fold={fd.fold_id!r} "
                    f"group={col!r} PCs={bad_pcs}: {std.tolist()}"
                )
            out[fd.fold_id][col] = {"mean": mean, "std": std}

    return out


# -----------------------------------------------------------------------------
# Persist
# -----------------------------------------------------------------------------

def save_fold_recenter_stats(
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    parquet_path: str,
    pca_bases_path: Path = _BASES_PATH,
    pca_manifest_path: Path = _MANIFEST_PATH,
    fold_definitions: Tuple[FoldDef, ...] = DEFAULT_FOLD_DEFINITIONS,
    out_path: Path = STATS_PATH,
    manifest_path: Path = STATS_MANIFEST_PATH,
) -> Tuple[str, str]:
    """Persist stats to npz + manifest. Returns (npz_sha256, manifest_json)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np_save: Dict[str, np.ndarray] = {}
    for fold_id, by_group in stats.items():
        for group, params in by_group.items():
            np_save[f"{fold_id}/{group}/mean"] = params["mean"].astype(np.float64)
            np_save[f"{fold_id}/{group}/std"] = params["std"].astype(np.float64)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **np_save)
    os.replace(tmp, out_path)

    npz_sha = _file_sha256(out_path)
    pca_sha = _file_sha256(pca_bases_path)
    parquet_sha = _file_sha256(Path(parquet_path))

    pca_manifest = json.loads(pca_manifest_path.read_text())

    manifest = {
        "fold_recenter_stats_sha256": npz_sha,
        "fold_recenter_stats_path": str(out_path.relative_to(out_path.parents[1])),
        "pca_bases_sha256": pca_sha,
        "l1_parquet_sha256": parquet_sha,
        "l1_parquet_path": str(Path(parquet_path).resolve()),
        "fold_definitions": [
            {"fold_id": fd.fold_id, "date_start": fd.date_start,
             "date_end": fd.date_end}
            for fd in fold_definitions
        ],
        "fit_protocol": "per_fold_recenter",
        "fit_window": "all rows in folds strictly preceding evaluation row's fold",
        "scaler": "StandardScaler (mean + std)",
        "granularity": "per (group, PC)",
        "causality": (
            "Walk-forward fold-fixed — fold[k] stats fit only on fold[<k] "
            "data. train fold uses identity (mean=0 std=1)."
        ),
        "per_group_k": pca_manifest.get("per_group_k"),
        "emb_cols": list(EMB_COLS),
    }

    m_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    m_tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(m_tmp, manifest_path)

    return npz_sha, json.dumps(manifest, indent=2)


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

def load_fold_recenter_stats(
    path: Path = STATS_PATH,
    manifest_path: Path = STATS_MANIFEST_PATH,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load fold-recenter stats from disk + verify manifest sha.

    Returns dict[fold_id][group] = {"mean": (K,), "std": (K,)}.
    """
    if not path.exists():
        raise RuntimeError(
            f"fold_recenter stats not found at {path}. Run\n"
            f"    python -m layer2.inference.fold_recenter\n"
            f"to fit + persist."
        )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    actual_sha = _file_sha256(path)
    expected_sha = manifest.get("fold_recenter_stats_sha256", "")
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"fold_recenter_stats SHA mismatch:\n"
            f"  manifest: {expected_sha}\n  actual:   {actual_sha}\n"
            f"Re-fit via `python -m layer2.inference.fold_recenter`."
        )

    data = np.load(path, allow_pickle=False)
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for key in data.files:
        # Expected key form: f"{fold_id}/{group}/{stat}"
        parts = key.split("/")
        if len(parts) != 3:
            continue
        fold_id, group, stat = parts
        out.setdefault(fold_id, {}).setdefault(group, {})[stat] = data[key]
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    """CLI entry: fit fold-recenter stats and persist.

        python -m layer2.inference.fold_recenter
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="raw_data/local_store/l1_pilot.parquet")
    args = ap.parse_args()

    print(f"Fitting walk-forward fold-fixed re-center stats")
    print(f"  L1 Parquet: {args.parquet}")
    print(f"  PCA bases:  {_BASES_PATH}")
    print(f"  Folds: {DEFAULT_FOLD_DEFINITIONS}")
    print()

    stats = fit_fold_recenter_stats(args.parquet)

    # Diagnostic print: per (fold, group) summary of mean/std distributions.
    print("Fold-recenter stats summary (post-fit):")
    print(f"{'fold':<6}  {'group':<14}  {'K':>3}  "
          f"{'mean[range]':>20}  {'std[range]':>18}")
    for fold_id, by_group in stats.items():
        for group in EMB_COLS:
            params = by_group[group]
            mean = params["mean"]
            std = params["std"]
            K = len(mean)
            print(f"{fold_id:<6}  {group:<14}  {K:>3}  "
                  f"[{mean.min():+.3f}, {mean.max():+.3f}]   "
                  f"[{std.min():.3f}, {std.max():.3f}]")
    print()

    npz_sha, _ = save_fold_recenter_stats(stats, args.parquet)
    print(f"Saved fold_recenter_stats.npz  sha256: {npz_sha[:16]}...")
    print(f"Manifest: {STATS_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
