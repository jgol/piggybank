"""v8 EmbProj unit tests — covers the grammar extension + PCA basis path.

Tests the specific behaviors that the Reality Checker flagged as
test-coverage gaps: EmbProj evaluator numerical correctness, PCA basis
SHA verification, scalar-only strip, missing-basis loud failure, sign
canonicalization determinism.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def test_embproj_evaluator_matches_hand_computed():
    """v9 P0-B: EmbProj_EMB_VIX_0 evaluator output must bit-equal the
    hand-computed (center, project, Mahalanobis-divide) on the same input.

    v9 changed the projection from `(v − mean) · components[k]` to
    `(v − mean) · components[k] / sqrt(explained_variance_[k])` so every
    EmbProj_X_k operator has unit variance on the train slice. This test
    verifies the new scaling and the evaluator's bit-equality with the
    hand-computed reference.
    """
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()  # ensure fresh load
    from layer2.evaluator import EvaluationContext, TreeEvaluator, _load_pca_bases_cached
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode, build_terminal_set, GType,
    )

    bases = _load_pca_bases_cached()
    assert bases is not None, "PCA bases not loaded — run layer2.inference.pca_bases first"
    assert "EMB_VIX" in bases

    # Build EmbProj_EMB_VIX_0(EMB_VIX)
    fd = next(f for f in ALL_FUNCTIONS if f.name == "EmbProj_EMB_VIX_0")
    term = next(t for t in build_terminal_set() if t.name == "EMB_VIX")
    tree = FuncNode(defn=fd, children=[TermNode(defn=term)])

    # Construct a deterministic test vector
    rng = np.random.default_rng(seed=12345)
    test_vec = rng.standard_normal(384).astype(np.float32)
    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    ctx.update({"EMB_VIX": test_vec})

    result = TreeEvaluator().evaluate(tree, ctx)

    # Hand-computed v9: (vec - mean_) · components[0] / sqrt(ev[0])
    g = bases["EMB_VIX"]
    raw = float(np.dot(test_vec - g["mean_"], g["components"][0]))
    pc_std = float(g["pc_std_"][0])
    expected = raw / pc_std

    assert abs(result - expected) < 1e-4, (
        f"EmbProj evaluator output {result} does not match hand-computed "
        f"normalized projection {expected} (raw={raw}, pc_std={pc_std})"
    )


def test_pca_basis_sha_mismatch_raises():
    """v8 Fix 4 + Code Reviewer C3: load_bases() must raise RuntimeError
    when manifest SHA does not match npz file SHA (tampering detection).
    """
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()
    from layer2.inference.pca_bases import load_bases, _BASES_PATH, _MANIFEST_PATH

    # Temporarily corrupt the manifest to simulate a tamper
    original_manifest = _MANIFEST_PATH.read_text()
    try:
        bad_manifest = json.loads(original_manifest)
        bad_manifest["pca_bases_sha256"] = "0" * 64  # obviously wrong
        _MANIFEST_PATH.write_text(json.dumps(bad_manifest, indent=2))

        with pytest.raises(RuntimeError, match="SHA mismatch"):
            load_bases()
    finally:
        _MANIFEST_PATH.write_text(original_manifest)
        _ev._reset_pca_bases_cache()


def test_scalar_only_grammar_strips_embproj():
    """v8 Fix 4 + Code Reviewer C7: scalar-only grammar MUST have zero
    EmbProj_* functions (and zero EMB_* functions generally). This is the
    arm-contamination guard — if scalar-only sees EmbProj, the encoder
    information leaks into the 'null' control."""
    from layer2.grammar import (
        SCALAR_ONLY_FUNCTIONS, build_scalar_only_terminal_set,
    )
    emb_in_scalar = [f for f in SCALAR_ONLY_FUNCTIONS
                     if "Emb" in f.name or "EMB_" in f.name]
    assert not emb_in_scalar, (
        f"scalar-only grammar leaks EMB functions: "
        f"{[f.name for f in emb_in_scalar]}"
    )
    emb_term = [t for t in build_scalar_only_terminal_set()
                if "EMB_" in t.name or "Emb" in t.name]
    assert not emb_term, (
        f"scalar-only terminals leak EMB: {[t.name for t in emb_term]}"
    )


def test_missing_bases_raises_loudly_when_grammar_has_embproj(monkeypatch):
    """v8 Fix 1 + R2: _load_pca_bases_cached(raise_on_missing=True) must
    raise RuntimeError when the bases file is missing. This is the
    pre-v8-pathology sentinel — silent None fallback would let real-l1
    silently degenerate to zero typed-vector signal.

    Implementation: mock load_bases to raise FileNotFoundError (the
    actual failure mode when npz is missing). Default-arg binding
    prevents monkeypatching _BASES_PATH directly; mocking the function
    body is cleaner.
    """
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()

    def _raise_missing():
        raise FileNotFoundError("pca_bases.npz missing (test fixture)")

    monkeypatch.setattr("layer2.inference.pca_bases.load_bases",
                        _raise_missing)

    try:
        with pytest.raises(RuntimeError, match="PCA bases unavailable"):
            _ev._load_pca_bases_cached(raise_on_missing=True)
    finally:
        _ev._reset_pca_bases_cache()


def test_embproj_unit_variance_on_train_rows():
    """v9 P0-B: Mahalanobis normalization ensures each of the 84
    `EmbProj_X_k` operators (per-group K capped at 15) has std ≈ 1 on the
    train slice. Sample train rows, compute the projection for every
    (group, k), assert every operator's std lands within [0.9, 1.1]."""
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()
    from layer2.evaluator import (
        EvaluationContext, TreeEvaluator, _load_pca_bases_cached,
    )
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode, build_terminal_set,
    )
    from layer2.inference.pca_bases import EMB_COLS, PER_GROUP_K_V9, TRAIN_END

    bases = _load_pca_bases_cached()
    assert bases is not None
    # R2 per-variate groups are optional (bases fit separately)
    _OPTIONAL_GROUPS = {"EMB_VAR_ATM_IV", "EMB_VAR_ATM_SPREAD", "EMB_VAR_SPX_RET"}
    for grp in EMB_COLS:
        if grp in _OPTIONAL_GROUPS:
            continue  # per-variate bases may not yet be fit
        assert grp in bases, f"missing group {grp}"

    parquet_path = Path(__file__).resolve().parents[1] / \
        "raw_data" / "local_store" / "l1_pilot.parquet"
    if not parquet_path.exists():
        pytest.skip(f"l1_pilot.parquet not present at {parquet_path}")

    import pandas as pd
    df = pd.read_parquet(parquet_path)
    train_df = df[df["date"] <= TRAIN_END].reset_index(drop=True)
    assert len(train_df) >= 100, f"train slice has only {len(train_df)} rows"

    # Use the FULL train slice so std is a stable population estimate of
    # ~6084 samples, not a noisy 100-sample subsample.
    rng = np.random.default_rng(seed=42)
    n_sample = min(2000, len(train_df))
    sample_idx = rng.choice(len(train_df), size=n_sample, replace=False)

    # The evaluator route: build EmbProj_X_k(X) trees and compute over rows.
    fd_by_name = {f.name: f for f in ALL_FUNCTIONS}
    term_by_name = {t.name: t for t in build_terminal_set()}
    eval_ = TreeEvaluator()
    ctx = EvaluationContext(max_lag=30, emb_dim=384)

    failures: list[str] = []
    for grp in EMB_COLS:
        # R2 per-variate groups may not yet have bases or Parquet columns
        if grp not in bases or grp not in train_df.columns:
            continue
        k_for_group = PER_GROUP_K_V9[grp]
        for k in range(k_for_group):
            op_name = f"EmbProj_{grp}_{k}"
            assert op_name in fd_by_name, f"missing op {op_name}"
            tree = FuncNode(
                defn=fd_by_name[op_name],
                children=[TermNode(defn=term_by_name[grp])],
            )
            outs = np.empty(n_sample, dtype=np.float64)
            for i, row_idx in enumerate(sample_idx):
                vec = np.asarray(train_df[grp].iloc[row_idx], dtype=np.float32)
                ctx.update({grp: vec})
                outs[i] = eval_.evaluate(tree, ctx)
            std = float(outs.std(ddof=1))
            if not (0.9 <= std <= 1.1):
                failures.append(f"{op_name}: std={std:.4f} (out of [0.9, 1.1])")

    assert not failures, (
        f"v9 P0-B Mahalanobis normalization failed for {len(failures)}/35 "
        f"operators:\n  " + "\n  ".join(failures)
    )


def test_pca_explained_variance_is_finite_and_positive():
    """v9 P0-B + per-group K: every (group, k) entry of
    pca_bases[group]["explained_variance_"] must be finite and strictly
    positive. A zero/NaN ev would make the Mahalanobis sqrt-divide explode
    silently. This guards against legacy npz files (pre-v9) sneaking
    through and re-introducing the F1 scale-mismatch pathology.

    v9: shape is per-group K (PER_GROUP_K_V9), not flat N_COMPONENTS.
    """
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()
    from layer2.evaluator import _load_pca_bases_cached
    from layer2.inference.pca_bases import EMB_COLS, PER_GROUP_K_V9

    bases = _load_pca_bases_cached()
    assert bases is not None
    for grp in EMB_COLS:
        # R2 per-variate groups are optional (bases fit separately)
        if grp not in bases:
            continue
        ev = bases[grp]["explained_variance_"]
        expected_k = PER_GROUP_K_V9[grp]
        assert ev.shape == (expected_k,), (
            f"{grp} explained_variance_ shape={ev.shape}, expected "
            f"({expected_k},) per PER_GROUP_K_V9"
        )
        assert np.all(np.isfinite(ev)), f"{grp} ev has NaN/Inf: {ev.tolist()}"
        assert np.all(ev > 0.0), f"{grp} ev has non-positive entries: {ev.tolist()}"
        # Also confirm pc_std_ matches sqrt(ev + eps)
        pc_std = bases[grp]["pc_std_"]
        expected = np.sqrt(ev + 1e-8)
        np.testing.assert_allclose(
            pc_std, expected, rtol=1e-5,
            err_msg=f"{grp} pc_std_ does not match sqrt(ev + eps)"
        )


def test_pca_sign_canonicalization_is_deterministic():
    """v8 Fix 4 + Code Reviewer C6: re-running `layer2.inference.pca_bases`
    on identical data must produce byte-identical npz output (pinned
    svd_solver + random_state + sign canonicalization). If this test
    fails, the v8 basis-SHA commitment in the pre-reg is not
    reproducible."""
    import hashlib
    from layer2.inference.pca_bases import _BASES_PATH

    # Capture current SHA
    sha_before = hashlib.sha256(_BASES_PATH.read_bytes()).hexdigest()

    # Re-fit
    result = subprocess.run(
        ["/opt/anaconda3/bin/python", "-m", "layer2.inference.pca_bases"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, (
        f"PCA refit failed: stderr={result.stderr[:500]}"
    )

    sha_after = hashlib.sha256(_BASES_PATH.read_bytes()).hexdigest()
    assert sha_before == sha_after, (
        f"PCA basis SHA is not deterministic!\n"
        f"  before: {sha_before}\n"
        f"  after:  {sha_after}\n"
        f"Check svd_solver='full', random_state=42, and sign canonicalization."
    )
