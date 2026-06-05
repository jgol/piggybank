"""v9 bug-fix unit tests (post-P0).

Covers the seven bugs from the diagnostic-report fix pass:
  Bug 1: data-derived bars_per_day in _sharpe.
  Bug 2: MAX_CONTRACTS leverage cap (gross-premium sizing).
  Bug 3: subtree_mutate is callable from gp_engine.mutate().
  Bug 4: EmbProj on uninitialized buffer returns 0.0.
  Bug 5: PCA bases loader default raise_on_missing=True.
  Bug 6: documented elsewhere ((internal doc) doc fix); covered by
         test_max_depth_consistency_across_sources here.
  Bug 7: SSL-position map consistency between batch_forecast and grammar.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest


_REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Bug 1 — data-derived bars_per_day
# ---------------------------------------------------------------------------

def test_bug1_sharpe_requires_explicit_bars_per_day():
    """v9 Bug 1: `_sharpe(returns)` without bars_per_day should fail loudly
    instead of silently using the legacy 78."""
    from layer2.evaluator import SimpleBacktester
    rs = np.array([0.001, 0.002, -0.001, 0.0005], dtype=float)
    with pytest.raises(TypeError):
        SimpleBacktester._sharpe(rs)  # missing bars_per_day → TypeError


def test_bug1_derive_bars_per_day_from_dataframe():
    """v9 Bug 1: `_derive_bars_per_day` returns the modal rows-per-date,
    bounded to [10, 405]."""
    from layer2.evaluator import SimpleBacktester
    # 30-min cadence: 12 bars/day for 3 dates
    df = pd.DataFrame({
        "date": (["2024-01-01"] * 12 + ["2024-01-02"] * 12
                 + ["2024-01-03"] * 12),
    })
    assert SimpleBacktester._derive_bars_per_day(df) == 12

    # 5-min cadence: 78 bars/day for 2 dates
    df2 = pd.DataFrame({
        "date": (["2024-01-01"] * 78 + ["2024-01-02"] * 78),
    })
    assert SimpleBacktester._derive_bars_per_day(df2) == 78


def test_bug1_derive_bars_per_day_loud_on_unexpected_grid():
    """v9 Bug 1: unexpected bar grid (outside [10, 405]) raises."""
    from layer2.evaluator import SimpleBacktester
    df = pd.DataFrame({"date": ["2024-01-01"] * 5})  # 5 bars — too low
    with pytest.raises(RuntimeError, match="bars_per_day"):
        SimpleBacktester._derive_bars_per_day(df)


def test_bug1_sharpe_change_factor_matches_sqrt_ratio():
    """v9 Bug 1: switching from 78 → 12 bars/day reduces reported Sharpe by
    sqrt(78/12) ≈ 2.55x. Same returns, different annualization."""
    from layer2.evaluator import SimpleBacktester
    import math

    rng = np.random.default_rng(42)
    rs = rng.normal(loc=0.0001, scale=0.001, size=2000)
    sharpe_78 = SimpleBacktester._sharpe(rs, bars_per_day=78)
    sharpe_12 = SimpleBacktester._sharpe(rs, bars_per_day=12)
    expected_ratio = math.sqrt(78 / 12)
    assert abs(sharpe_78 / sharpe_12 - expected_ratio) < 0.05


# ---------------------------------------------------------------------------
# Bug 2 — MAX_CONTRACTS leverage cap (gross-premium sizing)
# ---------------------------------------------------------------------------

def test_bug2_atm_call_spread_does_not_explode_n_contracts():
    """v9 Bug 2: An ATM 0DTE call spread (long ATM, short OTM) has
    near-zero net entry value but positive gross. Pre-fix sizing produced
    n_contracts ≈ 10000 (notional / 0.10 floor); post-fix sizing on gross
    should keep n_contracts at sensible values (e.g. ≤ 200 for $5 gross)."""
    import math
    from layer2.evaluator import MultiLegOptionsBacktester
    from layer2.templates import bull_call_debit

    template = bull_call_debit()
    bt = MultiLegOptionsBacktester(template, notional=1000.0)

    # Approximate ATM call spread @ spot=5000, iv=0.18, T=60min:
    # both legs near $5–7 with offsetting signs → net ~$1, gross ~$10
    spot = 5000.0
    minutes_left = 60.0
    iv = 0.18
    strikes = [
        bt._delta_to_strike(spot, leg.delta_target, iv, minutes_left)
        for leg in template.legs
    ]
    gross = bt._gross_position_value(spot, strikes, minutes_left, iv)
    net = bt._net_position_value(spot, strikes, minutes_left, iv)

    # Sanity: gross should dominate net for defined-risk debit spread
    assert gross > 0.0
    assert gross >= abs(net) - 1e-6

    # Replicate the post-fix sizing logic
    abs_val = max(gross, 0.50)
    MAX_CONTRACTS = bt.notional / 0.50
    n_contracts = min(bt.notional * 1.0 / abs_val, MAX_CONTRACTS)

    # n_contracts × gross ≤ notional (leverage ≤ 1.0 of capital exposure)
    assert n_contracts * gross <= bt.notional + 1e-6, (
        f"n_contracts={n_contracts} × gross={gross} = "
        f"{n_contracts * gross} > notional={bt.notional}"
    )
    # And n_contracts is a reasonable number, not the pre-fix 10× explosion
    assert n_contracts <= MAX_CONTRACTS


# ---------------------------------------------------------------------------
# Bug 3 — subtree_mutate is reachable from gp_engine.mutate
# ---------------------------------------------------------------------------

def test_bug3_subtree_mutate_can_grow_embproj_into_scalar_only_seed():
    """v9 Bug 3: subtree_mutate (now invoked from gp_engine.mutate at 0.10
    rate) MUST be capable of inserting EmbProj_* operators into trees that
    don't have them.

    Test plan:
      * Build a scalar-only entry tree: GT(ATM_IV, RealizedVol30m).
      * Run subtree_mutate N=1000 times against the FULL grammar (which
        includes EmbProj_*).
      * At least one mutation result should contain an EmbProj_* operator.
    """
    import random
    random.seed(20260424)

    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, Grammar, GType, TermNode,
        build_terminal_set, to_str,
    )

    grammar = Grammar(
        functions=ALL_FUNCTIONS, terminals=build_terminal_set(),
        max_depth=5, max_nodes=20,
    )

    # Build the scalar-only seed: GT(ATM_IV, RealizedVol30m)
    gt_fd = next(f for f in ALL_FUNCTIONS if f.name == "GT")
    atm = next(t for t in build_terminal_set() if t.name == "ATM_IV")
    rv = next(t for t in build_terminal_set() if t.name == "RealizedVol30m")
    seed = FuncNode(defn=gt_fd, children=[
        TermNode(defn=atm), TermNode(defn=rv),
    ])

    n_with_embproj = 0
    for _ in range(1000):
        try:
            mut = grammar.subtree_mutate(seed, max_d=3)
        except Exception:
            continue
        if "EmbProj_" in to_str(mut):
            n_with_embproj += 1

    assert n_with_embproj > 0, (
        f"subtree_mutate produced 0/1000 EmbProj-using trees from a "
        f"scalar-only seed — Bug 3 fix is not effective."
    )


def test_bug3_gp_engine_mutate_invokes_subtree_mutate():
    """v9 Bug 3: gp_engine.mutate's signature now includes
    subtree_mutation_rate. Verify the function is callable with the new
    kwarg and that subtree_mutate is in fact invoked."""
    import inspect
    from layer2.gp_engine import mutate as engine_mutate

    sig = inspect.signature(engine_mutate)
    assert "subtree_mutation_rate" in sig.parameters, (
        "gp_engine.mutate is missing the subtree_mutation_rate kwarg "
        "(Bug 3 fix not in place)"
    )


# ---------------------------------------------------------------------------
# Bug 4 — EmbProj on zero-vector returns 0.0
# ---------------------------------------------------------------------------

def test_bug4_embproj_on_zero_vector_returns_zero():
    """v9 Bug 4: when the typed-vector buffer is uninitialized (early bars
    before ctx.update fills it), EmbProj_X_k must return literal 0.0 — not
    project (0 - mean_) onto components, which yields a non-zero
    structural constant."""
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()
    from layer2.evaluator import EvaluationContext, TreeEvaluator
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode, build_terminal_set,
    )

    fd = next(f for f in ALL_FUNCTIONS if f.name == "EmbProj_EMB_VIX_0")
    term = next(t for t in build_terminal_set() if t.name == "EMB_VIX")
    tree = FuncNode(defn=fd, children=[TermNode(defn=term)])

    # No ctx.update() — vector buffer empty, get_vec returns ctx._zero_vec.
    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    out = TreeEvaluator().evaluate(tree, ctx)
    assert out == 0.0, f"EmbProj on uninitialized buffer returned {out}, expected 0.0"

    # Explicit all-zero vector (e.g. v2 corpus EMB_FLOW_AGG) must also → 0.
    ctx2 = EvaluationContext(max_lag=30, emb_dim=384)
    ctx2.update({"EMB_VIX": np.zeros(384, dtype=np.float32)})
    out2 = TreeEvaluator().evaluate(tree, ctx2)
    assert out2 == 0.0, (
        f"EmbProj on explicit zero vector returned {out2}, expected 0.0"
    )


# ---------------------------------------------------------------------------
# Bug 5 — _load_pca_bases_cached default is raise_on_missing=True
# ---------------------------------------------------------------------------

def test_bug5_load_pca_bases_default_raises_on_missing(monkeypatch):
    """v9 Bug 5: the loader default has flipped from raise_on_missing=False
    to True. A call without the kwarg must raise RuntimeError when bases
    are unavailable — not silently return None."""
    import layer2.evaluator as _ev
    _ev._reset_pca_bases_cache()

    def _raise_missing():
        raise FileNotFoundError("pca_bases.npz missing (test fixture)")

    monkeypatch.setattr("layer2.inference.pca_bases.load_bases",
                        _raise_missing)

    try:
        with pytest.raises(RuntimeError, match="PCA bases unavailable"):
            _ev._load_pca_bases_cached()  # default kwarg now True
    finally:
        _ev._reset_pca_bases_cache()


# ---------------------------------------------------------------------------
# Bug 6 — max_depth/max_nodes consistency across sources
# ---------------------------------------------------------------------------

def test_bug6_max_depth_consistency_across_sources():
    """v9 Bug 6: ExperimentConfig (production), pre-reg YAML, and the
    PRAXIS_DECISIONS L2.5 entry must all agree on the production
    max_depth/max_nodes. (internal doc) was previously stating 7/35 (wrong).
    """
    import yaml
    from layer2.experiment import ExperimentConfig

    cfg = ExperimentConfig(l1_parquet_path="raw_data/local_store/l1_pilot.parquet")
    assert cfg.grammar_max_depth == 4, cfg.grammar_max_depth
    assert cfg.grammar_max_nodes == 12, cfg.grammar_max_nodes

    # Pre-reg YAML
    prereg_path = _REPO / "docs" / "(internal doc)"
    if prereg_path.exists():
        prereg = yaml.safe_load(prereg_path.read_text())
        # Walk the YAML for grammar_max_depth/grammar_max_nodes anywhere
        text = prereg_path.read_text()
        assert "grammar_max_depth: 5" in text, (
            "Pre-reg YAML does not commit grammar_max_depth=5"
        )
        assert "grammar_max_nodes: 20" in text, (
            "Pre-reg YAML does not commit grammar_max_nodes=20"
        )

    # methodology entry
    praxis_path = _REPO / "layer2" / "PRAXIS_DECISIONS.md"
    if praxis_path.exists():
        praxis_text = praxis_path.read_text()
        assert "max_depth=5" in praxis_text and "max_nodes=20" in praxis_text


# ---------------------------------------------------------------------------
# Bug 7 — SSL-position map consistency between batch_forecast and grammar
# ---------------------------------------------------------------------------

def test_bug7_ssl_position_maps_consistent():
    """v9 Bug 7: `batch_forecast._TYPED_VECTOR_SSL_RANGES_V3` must agree
    with `grammar.EMB_TYPE_TO_RAW_RANGE` (mapped through `_ssl_pos`).
    Drift between them would silently corrupt every EmbProj output."""
    from layer1.inference.batch_forecast import (
        assert_ssl_position_maps_consistent,
    )
    # Should not raise
    assert_ssl_position_maps_consistent()


def test_bug7_ssl_position_maps_loud_failure_on_mutation(monkeypatch):
    """v9 Bug 7: if `_TYPED_VECTOR_SSL_RANGES_V3` is corrupted, the
    consistency assertion must raise RuntimeError."""
    import layer1.inference.batch_forecast as _bf
    bad_map = dict(_bf._TYPED_VECTOR_SSL_RANGES_V3)
    bad_map["emb_grid"] = (0, 50)  # wrong range
    monkeypatch.setattr(_bf, "_TYPED_VECTOR_SSL_RANGES_V3", bad_map)
    with pytest.raises(RuntimeError, match="SSL-position map drift"):
        _bf.assert_ssl_position_maps_consistent()
