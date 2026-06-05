"""Regression test for the BLOCKER (2026-06-03 code review): the VRP compound
seed's thresholds are authored in (fold-)normalized space and must NOT be
double-transformed by recalibrate_seed_thresholds (which assumes frozen-authored
seeds). Pre-fix, recalibration gutted ATM_IV θ 0.81->2.36 (~25% -> ~7.6% of bars
pass), silently changing the seeded fitness landscape. The fix tags the VRP EphReal
nodes `fold_normalized=True` so recalibration skips exactly them.
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from layer2.templates import (  # noqa: E402
    _vrp_compound_entry, _gt, _real_term, _ephemeral_real,
)
from layer2.terminal_stats import (  # noqa: E402
    recalibrate_seed_thresholds, normalize_threshold,
)

# fold robust stats with ATM_IV scale ~3x smaller than frozen (the corruption
# amplifier that #4's robust normalization introduces)
_FOLD = {"ATM_IV": (0.13, 0.0511, "robust"), "IVRVGap5d": (0.0, 0.50, "robust")}


def _threshold_of(node, term_name):
    """Return the EphReal value paired with `term_name` in the first GT/LT found."""
    if getattr(node, "name", None) in ("GT", "LT") and len(node.children) == 2:
        a, b = node.children
        if a.name == term_name and b.name == "EphReal":
            return b.value
        if b.name == term_name and a.name == "EphReal":
            return a.value
    for ch in getattr(node, "children", []):
        r = _threshold_of(ch, term_name)
        if r is not None:
            return r
    return None


def test_vrp_seed_thresholds_survive_recalibration_unchanged():
    seed = _vrp_compound_entry(0.8125, 0.6227)  # 75th-pct normalized thresholds
    s2 = copy.deepcopy(seed)
    recalibrate_seed_thresholds(s2, _FOLD)
    assert _threshold_of(s2, "ATM_IV") == pytest.approx(0.8125, abs=1e-9), (
        "VRP ATM_IV threshold must be preserved (not double-transformed)")
    assert _threshold_of(s2, "IVRVGap5d") == pytest.approx(0.6227, abs=1e-9)


def test_vrp_tag_survives_deepcopy():
    seed = _vrp_compound_entry(0.8125, 0.6227)
    s2 = copy.deepcopy(seed)
    # find an EphReal under a GT and confirm the tag survived the deepcopy
    gt = next(c for c in s2.children if c.name == "GT")
    eph = next(ch for ch in gt.children if ch.name == "EphReal")
    assert getattr(eph, "fold_normalized", False) is True


def test_legacy_frozen_seed_still_recalibrates():
    # a frozen-authored threshold (the legacy path) must STILL be transformed —
    # the fix must be surgical, not a blanket skip.
    legacy = _gt(_real_term("ATM_IV"),
                 _ephemeral_real(normalize_threshold("ATM_IV", 0.18)))
    before = legacy.children[1].value
    recalibrate_seed_thresholds(legacy, _FOLD)
    after = legacy.children[1].value
    assert abs(after - before) > 1e-6, (
        "legacy frozen-authored seed must still recalibrate (not skipped)")


@pytest.mark.parametrize("q", [0.66, 0.75])
def test_post_recalibration_selectivity_preserved_on_real_data(q):
    """The whole point: post-recalibration the VRP gate still selects ~the intended
    quantile of bars (was gutted to ~7.6% pre-fix)."""
    import numpy as np, pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    from layer2.terminal_stats import compute_norm_stats_from_arrays
    from layer2.experiment import derive_vrp_compound_seed
    P = "raw_data/local_store/l1_minute_scalars.parquet"
    if not os.path.exists(P):
        pytest.skip("production parquet absent")
    df = pd.read_parquet(P, filters=[("date", "<", "2024-03-30")])
    tr = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                       embargo_days=0)["train"]
    fold = compute_norm_stats_from_arrays(
        prepare_terminal_data(tr, normalize_terminals=False), robust=True)
    td = prepare_terminal_data(tr, norm_stats_override=fold)
    seed = derive_vrp_compound_seed(td, q=q)
    assert seed is not None
    recalibrate_seed_thresholds(seed, fold)  # the step that used to corrupt it
    iv_thr = _threshold_of(seed, "ATM_IV")
    iv_norm = td["ATM_IV"]
    pass_rate = float(np.mean(iv_norm >= iv_thr))
    # intended ~ (1 - q); pre-fix this collapsed to ~0.076 for q=0.75
    assert abs(pass_rate - (1.0 - q)) < 0.08, (
        f"post-recalibration ATM_IV pass-rate {pass_rate:.3f} should be ~{1-q:.2f}, "
        f"not the pre-fix-gutted ~0.076")


# ---------------------------------------------------------------------------
# BLOCKER end-to-end (review 2026-06-03): in the production conditions the seed
# is SUBSTITUTED before it is recalibrated (experiment._run_one_template), and the
# substitution walkers rebuild every TermNode — which dropped the fold_normalized
# tag, defeating the fix for scalar-only / probes-only / emb-only (real-l1 was the
# only condition that worked). These tests route through the real production order.
# ---------------------------------------------------------------------------
def _all_ephs(node, out):
    if getattr(node, "name", None) == "EphReal":
        out.append((round(node.value, 6), getattr(node, "fold_normalized", False)))
    for ch in getattr(node, "children", []):
        _all_ephs(ch, out)
    return out


@pytest.mark.parametrize("subst_name", [
    "substitute_stripped_terminals_for_scalar_only",   # condition A (production pilot)
    "substitute_for_probes_pure",                      # condition B
    "substitute_for_emb_pure",                         # condition C
])
def test_vrp_tag_survives_substitution_then_recalibration(subst_name):
    import layer2.grammar as g
    subst = getattr(g, subst_name)
    seed = _vrp_compound_entry(1.0, 0.6)   # ATM_IV θ=1.0σ (the amplifier case)
    seed = subst(seed)                     # SUBSTITUTE (production: before recalib)
    recalibrate_seed_thresholds(seed, _FOLD)  # then RECALIBRATE
    ephs = _all_ephs(seed, [])
    # Both VRP thresholds present, still tagged, and UNCHANGED (not double-transformed).
    assert sorted(v for v, _ in ephs) == [0.6, 1.0], (
        f"{subst_name}: thresholds changed -> {ephs} (tag stripped by substitution?)")
    assert all(tag for _, tag in ephs), (
        f"{subst_name}: fold_normalized tag did not survive substitution -> {ephs}")
