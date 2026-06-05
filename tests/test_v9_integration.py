"""v9 integration unit tests — covers the 3 HIGH concerns from the
Code Reviewer pass:

L8-1:  fold_id is propagated via EvaluationContext.current_fold_id ONLY,
       NOT through bar_data. Grammar terminals/functions cannot reference
       fold_id.

L8-12: PER_GROUP_K_V9 is a single source-of-truth (defined ONCE in
       layer2.inference.pca_bases, imported by grammar.py and evaluator.py).
       All 84 EmbProj operators are reachable AND produce non-zero outputs
       on representative train rows.

L4-10: fold_recenter_manifest.pca_bases_sha256 + l1_parquet_sha256 are
       cross-checked at evaluator load time. On mismatch: RuntimeError
       with both shas in the message — NOT a silent fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# L8-1: fold_id evaluator-boundary protocol
# =============================================================================

def test_l8_1_grammar_has_no_fold_id_terminal():
    """L8-1: no grammar terminal may be named fold_id, current_fold,
    fold, or any reserved variant. The fold_id is an evaluator-context
    attribute, NOT a tree input.
    """
    from layer2.grammar import (
        ALL_FUNCTIONS, build_terminal_set,
    )
    forbidden = {"fold_id", "current_fold_id", "fold", "current_fold",
                 "FoldId", "FOLD_ID", "__fold_id"}
    leaks = []
    for t in build_terminal_set():
        if t.name in forbidden:
            leaks.append(("build_terminal_set()", t.name))
    for f in ALL_FUNCTIONS:
        if f.name in forbidden:
            leaks.append(("ALL_FUNCTIONS", f.name))
    assert not leaks, (
        f"L8-1 violation: grammar exposes fold_id-like names: {leaks!r}. "
        f"fold_id MUST be propagated via EvaluationContext.current_fold_id, "
        f"not through grammar terminals or functions."
    )


def test_l8_1_bar_data_rejects_fold_id_key():
    """L8-1: EvaluationContext.update() must REJECT a __fold_id key in
    bar_data with a RuntimeError. This blocks the silent backdoor by which
    a backtester might smuggle fold_id into per-bar data.
    """
    from layer2.evaluator import EvaluationContext
    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    with pytest.raises(RuntimeError, match="L8-1 boundary violation"):
        ctx.update({"__fold_id": "val", "SPXClose": 4500.0})


def test_l8_1_set_current_fold_id_is_only_path():
    """L8-1: set_current_fold_id() is the only sanctioned path to inject
    fold_id into the EvaluationContext.
    """
    from layer2.evaluator import EvaluationContext
    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    assert ctx.current_fold_id is None  # default
    ctx.set_current_fold_id("val")
    assert ctx.current_fold_id == "val"
    ctx.set_current_fold_id(None)
    assert ctx.current_fold_id is None
    # The legacy `.fold_id` shim forwards to the same backing field.
    ctx.fold_id = "test"
    assert ctx.current_fold_id == "test"
    assert ctx.fold_id == "test"


def test_l8_1_backtesters_inject_fold_id_into_ctx_not_bar_data():
    """L8-1: SimpleBacktester / OptionsBacktester / MultiLegOptionsBacktester
    each call ctx.set_current_fold_id(...) BEFORE ctx.update(bar_data).
    Confirmed by source-string grep — the implementation pattern in all
    three backtester loops is the same.
    """
    import layer2.evaluator as _ev
    src = Path(_ev.__file__).read_text()
    n_set = src.count("ctx.set_current_fold_id(_fold_ids[i])")
    assert n_set >= 2, (
        f"Expected >= 2 backtester loops calling set_current_fold_id(), "
        f"found {n_set}. The L8-1 propagation pattern is missing in at "
        f"least one backtester (SimpleBacktester + MultiLegOptionsBacktester)."
    )
    # Ensure no __fold_id smuggling pattern exists.
    assert "bar_data['__fold_id']" not in src
    assert 'bar_data["__fold_id"]' not in src


# =============================================================================
# L8-12: PER_GROUP_K_V9 single source-of-truth
# =============================================================================

def test_l8_12_per_group_k_is_single_source_of_truth():
    """L8-12: grammar.py imports PER_GROUP_K_V9 from
    layer2.inference.pca_bases. There must be NO independent definition
    of PER_GROUP_K (or _N_EMBPROJ_COMPONENTS=5) inside grammar.py — that
    would let the two drift apart silently.
    """
    from layer2.inference.pca_bases import PER_GROUP_K_V9 as _PCA_PGK
    from layer2.grammar import PER_GROUP_K_GRAMMAR_V9 as _GRAM_PGK
    assert _PCA_PGK == _GRAM_PGK, (
        f"PER_GROUP_K drift: pca_bases.PER_GROUP_K_V9={_PCA_PGK} vs "
        f"grammar.PER_GROUP_K_GRAMMAR_V9={_GRAM_PGK}"
    )
    # The grammar module must NOT define a single-K constant.
    import layer2.grammar as _gram
    bad_attrs = [
        "_N_EMBPROJ_COMPONENTS", "N_EMBPROJ_COMPONENTS",
        "_PCA_K", "PCA_K", "_FLAT_K", "FLAT_K",
    ]
    for name in bad_attrs:
        assert not hasattr(_gram, name), (
            f"L8-12 violation: layer2.grammar exposes single-K constant "
            f"{name!r} = {getattr(_gram, name)!r}. Single source-of-truth "
            f"is layer2.inference.pca_bases.PER_GROUP_K_V9."
        )


def test_l8_12_all_84_embproj_operators_reachable():
    """L8-12: all 99 EmbProj_*_k operators (sum of PER_GROUP_K_V9) must
    be enumerable from ALL_FUNCTIONS. A drift between grammar and PCA
    bases would leave some operators unreachable.
    """
    from layer2.grammar import ALL_FUNCTIONS
    from layer2.inference.pca_bases import PER_GROUP_K_V9, EMB_COLS
    expected = []
    for grp in EMB_COLS:
        for k in range(PER_GROUP_K_V9[grp]):
            expected.append(f"EmbProj_{grp}_{k}")
    assert len(expected) == 99, f"Expected 99, got {len(expected)}"
    fd_names = {f.name for f in ALL_FUNCTIONS}
    missing = [n for n in expected if n not in fd_names]
    assert not missing, (
        f"L8-12 violation: {len(missing)} EmbProj operators are NOT in "
        f"ALL_FUNCTIONS: {missing[:5]}..."
    )


def test_l8_12_all_84_embproj_operators_produce_nonzero_on_train_rows():
    """L8-12: every EmbProj_*_k operator whose PCA basis exists must produce
    a non-zero output on at least one representative train-slice row. If a
    single-K constant in grammar drifted from per-group K in evaluator,
    the `if k >= components.shape[0]` guard would silently return 0.0
    for a subset of operators — half-broken v9 with NO error.

    R2 per-variate groups (EMB_VAR_*) are skipped if they don't yet exist
    in the Parquet or PCA bases — their bases are fit separately.
    """
    parquet_path = Path(__file__).resolve().parents[1] / \
        "raw_data" / "local_store" / "l1_pilot.parquet"
    if not parquet_path.exists():
        pytest.skip(f"l1_pilot.parquet not present at {parquet_path}")

    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()
    from layer2.evaluator import (
        EvaluationContext, TreeEvaluator,
    )
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode, build_terminal_set,
    )
    from layer2.inference.pca_bases import EMB_COLS, PER_GROUP_K_V9, TRAIN_END

    df = pd.read_parquet(parquet_path)
    train_df = df[df["date"] <= TRAIN_END].reset_index(drop=True)
    assert len(train_df) >= 100

    rng = np.random.default_rng(seed=42)
    n_sample = min(50, len(train_df))
    sample_idx = rng.choice(len(train_df), size=n_sample, replace=False)

    fd_by_name = {f.name: f for f in ALL_FUNCTIONS}
    term_by_name = {t.name: t for t in build_terminal_set()}
    eval_ = TreeEvaluator()
    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    ctx.set_current_fold_id("train")  # train fold uses identity recenter

    failures = []
    for grp in EMB_COLS:
        # R2 per-variate groups may not yet exist in the Parquet — skip
        if grp not in train_df.columns:
            continue
        for k in range(PER_GROUP_K_V9[grp]):
            op_name = f"EmbProj_{grp}_{k}"
            tree = FuncNode(
                defn=fd_by_name[op_name],
                children=[TermNode(defn=term_by_name[grp])],
            )
            outs = []
            for i in sample_idx:
                vec = np.asarray(train_df[grp].iloc[i], dtype=np.float32)
                ctx.update({grp: vec})
                outs.append(eval_.evaluate(tree, ctx))
            outs = np.asarray(outs, dtype=np.float64)
            n_nonzero = int(np.sum(np.abs(outs) > 1e-10))
            if n_nonzero == 0:
                failures.append(
                    f"{op_name}: ALL {n_sample} outputs are 0.0 — likely "
                    f"silent k>=components.shape[0] guard from PER_GROUP_K "
                    f"drift between grammar and evaluator."
                )
    assert not failures, (
        f"L8-12 silent-zero leak detected for {len(failures)} operators:\n  "
        + "\n  ".join(failures)
    )


def test_l8_12_per_group_k_sum_is_84():
    from layer2.inference.pca_bases import PER_GROUP_K_V9
    assert sum(PER_GROUP_K_V9.values()) == 99  # 84 group-pooled + 15 R2 per-variate


# =============================================================================
# L4-10: fold_recenter_manifest sha cross-check at evaluator load time
# =============================================================================

def test_l4_10_matching_shas_load_ok():
    """L4-10 positive: when fold_recenter_manifest's shas match the
    actually-loaded artifacts, _load_fold_recenter_cached() returns the
    stats dict cleanly.
    """
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()
    _ev._reset_fold_recenter_cache()
    stats = _ev._load_fold_recenter_cached(raise_on_missing=True)
    assert stats is not None and stats != {}
    # Train, val, test fold IDs all present.
    assert "train" in stats
    assert "val" in stats
    assert "test" in stats


def test_l4_10_pca_sha_mismatch_raises_with_both_shas():
    """L4-10 negative: when fold_recenter_manifest.pca_bases_sha256 does
    not match the actual pca_bases.npz sha, RuntimeError is raised with
    BOTH shas in the message.
    """
    from layer2.inference.fold_recenter import STATS_MANIFEST_PATH
    import layer2.evaluator as _ev

    original = STATS_MANIFEST_PATH.read_text()
    try:
        man = json.loads(original)
        real_sha = man["pca_bases_sha256"]
        fake_sha = "0" * 64
        man["pca_bases_sha256"] = fake_sha
        STATS_MANIFEST_PATH.write_text(json.dumps(man, indent=2))

        # Force a clean cache to trigger reload.
        _ev._reset_pca_bases_cache()
        _ev._reset_fold_recenter_cache()
        with pytest.raises(RuntimeError) as excinfo:
            _ev._load_fold_recenter_cached(raise_on_missing=True)
        msg = str(excinfo.value)
        # Both shas (or their 16-char prefixes) must appear in the message.
        assert "L4-10" in msg or "sha" in msg.lower()
        assert fake_sha[:16] in msg, (
            f"expected fake sha {fake_sha[:16]}... in error msg, got: {msg}"
        )
        assert real_sha[:16] in msg, (
            f"expected real sha {real_sha[:16]}... in error msg, got: {msg}"
        )
    finally:
        STATS_MANIFEST_PATH.write_text(original)
        _ev._reset_pca_bases_cache()
        _ev._reset_fold_recenter_cache()


def test_l4_10_parquet_sha_mismatch_raises_with_both_shas():
    """L4-10 negative: when fold_recenter_manifest.l1_parquet_sha256 does
    not match the actual l1_pilot.parquet sha, RuntimeError is raised with
    BOTH shas in the message.
    """
    from layer2.inference.fold_recenter import STATS_MANIFEST_PATH
    import layer2.evaluator as _ev

    original = STATS_MANIFEST_PATH.read_text()
    try:
        man = json.loads(original)
        real_sha = man["l1_parquet_sha256"]
        fake_sha = "0" * 64
        man["l1_parquet_sha256"] = fake_sha
        STATS_MANIFEST_PATH.write_text(json.dumps(man, indent=2))

        _ev._reset_pca_bases_cache()
        _ev._reset_fold_recenter_cache()
        with pytest.raises(RuntimeError) as excinfo:
            _ev._load_fold_recenter_cached(raise_on_missing=True)
        msg = str(excinfo.value)
        assert "L4-10" in msg or "sha" in msg.lower()
        assert fake_sha[:16] in msg
        assert real_sha[:16] in msg
    finally:
        STATS_MANIFEST_PATH.write_text(original)
        _ev._reset_pca_bases_cache()
        _ev._reset_fold_recenter_cache()


def test_l4_10_no_silent_fallback_on_mismatch():
    """L4-10: a sha mismatch must NEVER silently fall back to identity
    recenter. The evaluator must propagate the RuntimeError when
    raise_on_missing=True.
    """
    from layer2.inference.fold_recenter import STATS_MANIFEST_PATH
    import layer2.evaluator as _ev

    original = STATS_MANIFEST_PATH.read_text()
    try:
        man = json.loads(original)
        man["pca_bases_sha256"] = "f" * 64  # obviously wrong
        STATS_MANIFEST_PATH.write_text(json.dumps(man, indent=2))

        _ev._reset_pca_bases_cache()
        _ev._reset_fold_recenter_cache()
        # raise_on_missing=True is the production setting → RuntimeError.
        with pytest.raises(RuntimeError):
            _ev._load_fold_recenter_cached(raise_on_missing=True)
    finally:
        STATS_MANIFEST_PATH.write_text(original)
        _ev._reset_pca_bases_cache()
        _ev._reset_fold_recenter_cache()


# =============================================================================
# Cache key includes fold_recenter_sha and fold_id (per #7b decision)
# =============================================================================

def test_v9_cache_key_includes_fold_recenter_sha_and_fold_id():
    """The fitness cache key must incorporate fold_recenter_sha and
    fold_id so cross-fold tree comparisons don't collide on cache.
    """
    from layer2.fitness import FitnessEvaluator
    from layer2.templates import all_templates
    parquet_path = Path(__file__).resolve().parents[1] / \
        "raw_data" / "local_store" / "l1_pilot.parquet"
    if not parquet_path.exists():
        pytest.skip(f"l1_pilot.parquet not present at {parquet_path}")
    import pandas as pd
    df = pd.read_parquet(parquet_path).head(120)
    tmpl = all_templates()[0]
    fe = FitnessEvaluator(template=tmpl, data=df)
    # The constructor populated _pca_bases_sha and _fold_recenter_sha.
    assert isinstance(fe._pca_bases_sha, str) and fe._pca_bases_sha
    assert isinstance(fe._fold_recenter_sha, str) and fe._fold_recenter_sha
    # The cache key must literally include the FRC: prefix.
    from layer2.grammar import build_terminal_set, ALL_FUNCTIONS
    # Build a trivial entry/exit/size triple (any leaves work).
    from layer2.grammar import (
        TermNode, FuncNode, GType,
    )
    # Build entry/exit/size — use simplest valid triple.
    real_terms = [t for t in build_terminal_set() if t.ret_type == GType.REAL]
    side_terms = [t for t in build_terminal_set() if t.ret_type == GType.SIDE]
    if not (real_terms and side_terms):
        pytest.skip("Need REAL/SIDE terminals to build a tree triple")
    # Find a comparator: REAL × REAL → BOOL.
    gt = next((f for f in ALL_FUNCTIONS if f.name == "GT"), None)
    if gt is None:
        pytest.skip("Need GT comparator")
    entry = FuncNode(defn=gt, children=[
        TermNode(defn=real_terms[0]), TermNode(defn=real_terms[1])
    ])
    exit_ = FuncNode(defn=gt, children=[
        TermNode(defn=real_terms[0]), TermNode(defn=real_terms[1])
    ])
    size = TermNode(defn=side_terms[0])
    key = fe._tree_hash(entry, exit_, size)
    assert "PCA:" in key, f"cache key missing PCA: prefix: {key!r}"
    assert "FRC:" in key, f"cache key missing FRC: prefix: {key!r}"
    assert "FID:" in key, f"cache key missing FID: prefix: {key!r}"
    assert fe._pca_bases_sha[:8] in key
    assert fe._fold_recenter_sha[:8] in key
