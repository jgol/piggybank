"""ITEM #5 — VRP-anchored compound entry seeds (replace the leaked, all-negative
grid-search seeds).

Locks: (1) the 4 CREDIT base templates' entry_seeds are the compound AND form
AND(GT(ATM_IV, .), GT(IVRVGap5d, .)); (2) exit = hold-to-expiry, stop_seed=0.0;
(3) the compound seed is condition-agnostic (ATM_IV + IVRVGap5d are in every
condition's terminal set); (4) per-fold θ derivation produces finite thresholds on
real fold-1 data; (5) seeds parse / round-trip (structurally valid); (6) the
debit RPB template deliberately keeps a debit-appropriate entry (documented).
"""
import os

import numpy as np
import pytest

from layer2.grammar import (
    GType, to_str, from_sexpr, GT, AND,
    build_scalar_only_terminal_set, build_probes_only_terminal_set,
    build_emb_only_terminal_set,
)
from layer2.templates import (
    BASE_TEMPLATE_FACTORIES, base_template_by_name,
    _vrp_compound_entry, _exit_hold_to_expiry,
    iron_condor_base, iron_butterfly_base,
    bull_put_credit_base, bear_call_credit_base, ratio_put_backspread_base,
)

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"
_skip_no_parquet = pytest.mark.skipif(
    not os.path.exists(_PARQUET), reason="production minute parquet not present")

# The 4 credit templates that carry the VRP compound entry (RPB is debit -> excluded).
_CREDIT_BASES = (iron_condor_base, iron_butterfly_base,
                 bull_put_credit_base, bear_call_credit_base)


# ---------------------------------------------------------------------------
# (1) credit base templates use the compound VRP entry form
# ---------------------------------------------------------------------------

def _is_vrp_compound(tree) -> bool:
    """Structurally check tree == AND(GT(ATM_IV, EphReal), GT(IVRVGap5d, EphReal))."""
    if not (hasattr(tree, "defn") and tree.defn is AND and len(tree.children) == 2):
        return False
    terms = set()
    for child in tree.children:
        if not (hasattr(child, "defn") and child.defn is GT and len(child.children) == 2):
            return False
        lhs, rhs = child.children
        if not (hasattr(lhs, "defn") and lhs.defn.name in ("ATM_IV", "IVRVGap5d")):
            return False
        if not (hasattr(rhs, "defn") and rhs.defn.name == "EphReal"):
            return False
        terms.add(lhs.defn.name)
    return terms == {"ATM_IV", "IVRVGap5d"}


def test_credit_base_templates_use_compound_vrp_entry():
    for factory in _CREDIT_BASES:
        t = factory()
        assert _is_vrp_compound(t.entry_seed), (
            f"{t.name} entry_seed is not the compound VRP form: {to_str(t.entry_seed)}")
        s = to_str(t.entry_seed)
        assert "ATM_IV" in s and "IVRVGap5d" in s and "AND" in s


def test_vrp_entry_references_both_required_terminals():
    """The seed references ATM_IV AND IVRVGap5d (the two-condition VRP gate)."""
    s = to_str(_vrp_compound_entry())
    assert s.count("GT") == 2
    assert "ATM_IV" in s and "IVRVGap5d" in s


# ---------------------------------------------------------------------------
# (2) exit = hold-to-expiry, stop_seed = 0.0
# ---------------------------------------------------------------------------

def test_credit_base_templates_exit_is_hold_to_expiry_and_stop_zero():
    for factory in _CREDIT_BASES:
        t = factory()
        # Hold-to-expiry seed: LT(MinutesToClose, EphReal(-5.0)) — never fires.
        s = to_str(t.exit_seed)
        assert "MinutesToClose" in s and "LT" in s, f"{t.name} exit not hold-to-expiry: {s}"
        assert t.stop_seed == 0.0, f"{t.name} stop_seed != 0.0 (got {t.stop_seed})"


def test_hold_to_expiry_exit_threshold_is_below_normalized_range():
    """The hold-to-expiry exit compares MinutesToClose < -5.0; robust-normalized
    MinutesToClose lives in [-1.349, +1.349], so the seed NEVER fires (true
    hold-to-expiry). We assert the threshold is well below that range."""
    ex = _exit_hold_to_expiry()
    # rhs EphReal value
    rhs = ex.children[1]
    assert rhs.defn.name == "EphReal"
    assert rhs.value <= -5.0
    assert rhs.value < -1.349  # below the robust-normalized MTC minimum


def test_all_base_templates_keep_stop_seed_zero():
    """All 5 base templates (incl. RPB) seed stop_mult at 0.0 (hold-to-expiry)."""
    for factory in BASE_TEMPLATE_FACTORIES:
        t = factory()
        assert t.stop_seed == 0.0, f"{t.name} stop_seed != 0.0"


# ---------------------------------------------------------------------------
# (3) condition-agnostic: ATM_IV + IVRVGap5d in every condition's terminal set
# ---------------------------------------------------------------------------

def test_ivrvgap5d_present_in_all_conditions():
    """IVRVGap5d is a multi-day-context terminal available DIRECTLY in every
    condition (A/B/C/D) — it is intentionally NOT substituted (grammar.py:1658).
    So the gap half of the VRP gate is condition-agnostic by construction."""
    for builder, label in (
        (build_scalar_only_terminal_set, "scalar-only (A)"),
        (build_probes_only_terminal_set, "probes-only (B)"),
        (build_emb_only_terminal_set, "emb-only (C)"),
    ):
        names = {t.name for t in builder()}
        assert "IVRVGap5d" in names, f"IVRVGap5d missing from {label}"


def test_atm_iv_present_in_run_conditions_and_substituted_elsewhere():
    """ATM_IV is a raw bypass scalar: present DIRECTLY in scalar-only (A) and
    real-l1 (D) — the conditions the production run uses — and mapped to a
    per-condition ANALOG by the existing seed-substitution machinery in B/C
    (B: ->PredRV15, C: ->SessionReturn). The seed is therefore condition-FAIR via
    the same ablation mechanism every seed terminal uses, even though ATM_IV is not
    a literal terminal in B/C. (Refines the task's 'in ALL conditions' premise.)"""
    from layer2.grammar import (
        _PROBES_PURE_SUBSTITUTIONS, _EMB_PURE_SUBSTITUTIONS,
        substitute_for_probes_pure, substitute_for_emb_pure,
    )
    # Directly present in A.
    assert "ATM_IV" in {t.name for t in build_scalar_only_terminal_set()}
    # Substituted (not dropped) in B and C.
    assert _PROBES_PURE_SUBSTITUTIONS.get("ATM_IV") == "PredRV15"
    assert _EMB_PURE_SUBSTITUTIONS.get("ATM_IV") == "SessionReturn"
    # The substitution actually rewrites a VRP seed to valid per-condition trees.
    seed = _vrp_compound_entry()
    b_seed = substitute_for_probes_pure(seed)
    c_seed = substitute_for_emb_pure(seed)
    assert b_seed.ret_type == GType.BOOL and "PredRV15" in to_str(b_seed)
    assert c_seed.ret_type == GType.BOOL
    # IVRVGap5d survives substitution in both (available to all conditions).
    assert "IVRVGap5d" in to_str(b_seed) and "IVRVGap5d" in to_str(c_seed)


# ---------------------------------------------------------------------------
# (4) per-fold θ derivation produces finite thresholds on real fold-1 data
# ---------------------------------------------------------------------------

@_skip_no_parquet
def test_per_fold_vrp_threshold_derivation_real_data():
    """derive_vrp_compound_seed yields a compound seed with FINITE, sensibly-ordered
    θ thresholds (top-quartile of the normalized terminals) on fold-1 train."""
    import pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    from layer2.terminal_stats import compute_norm_stats_from_arrays
    from layer2.experiment import (
        derive_vrp_compound_seed, _norm_terminal_quantile, VRP_SEED_QUANTILE,
    )
    df = pd.read_parquet(_PARQUET)
    df["date"] = df["date"].astype(str)
    train = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                          embargo_days=0)["train"]
    raw = prepare_terminal_data(train, normalize_terminals=False)
    fold_stats = compute_norm_stats_from_arrays(raw, robust=True)
    td = prepare_terminal_data(train, norm_stats_override=fold_stats)

    seed = derive_vrp_compound_seed(td)
    assert seed is not None
    assert _is_vrp_compound(seed), to_str(seed)
    # The derived θ are the 75th percentile of the NORMALIZED terminals -> positive
    # (top quartile sits above the median=0 of a centered distribution) and finite.
    t1 = _norm_terminal_quantile(td, "ATM_IV", VRP_SEED_QUANTILE)
    t2 = _norm_terminal_quantile(td, "IVRVGap5d", VRP_SEED_QUANTILE)
    assert t1 is not None and t2 is not None
    assert np.isfinite(t1) and np.isfinite(t2)
    assert t1 > 0.0 and t2 > 0.0, f"top-quartile θ not positive: θ1={t1}, θ2={t2}"
    # The seed's embedded EphReal thresholds equal the derived quantiles.
    embedded = sorted(ch.children[1].value for ch in seed.children)
    assert embedded == pytest.approx(sorted([t1, t2]), abs=1e-6)


@_skip_no_parquet
def test_derive_per_fold_entry_seed_includes_vrp_candidate_real_data():
    """derive_per_fold_entry_seed runs end-to-end on real fold-1 data and returns a
    valid BOOL entry seed (the VRP compound candidate is in its scored pool)."""
    import pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    from layer2.terminal_stats import compute_norm_stats_from_arrays
    from layer2.experiment import derive_per_fold_entry_seed, candidate_entry_seeds
    df = pd.read_parquet(_PARQUET)
    df["date"] = df["date"].astype(str)
    train = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                          embargo_days=0)["train"]
    raw = prepare_terminal_data(train, normalize_terminals=False)
    fold_stats = compute_norm_stats_from_arrays(raw, robust=True)
    td = prepare_terminal_data(train, norm_stats_override=fold_stats)

    # With train terminal data, the candidate pool gains the data-derived VRP seed.
    pool_static = candidate_entry_seeds()
    pool_with_vrp = candidate_entry_seeds(train_terminal_data=td)
    assert len(pool_with_vrp) == len(pool_static) + 1
    assert any(_is_vrp_compound(c) for c in pool_with_vrp)

    t = base_template_by_name("bull_put_credit")
    chosen = derive_per_fold_entry_seed(t, train, td, min_trades=20)
    assert chosen.ret_type == GType.BOOL
    # round-trips
    assert to_str(from_sexpr(to_str(chosen))) == to_str(chosen)


def test_candidate_pool_without_data_is_unchanged_fixed_size():
    """No train data -> the pool is the FIXED 10-member hypothesis space (no VRP
    candidate appended) — backward-compatible with the existing per-fold-seeds test."""
    from layer2.experiment import candidate_entry_seeds
    pool = candidate_entry_seeds()
    assert len(pool) == 10
    for t in pool:
        assert t.ret_type == GType.BOOL


# ---------------------------------------------------------------------------
# (5) structural validity / round-trip
# ---------------------------------------------------------------------------

def test_vrp_seeds_parse_and_round_trip():
    for factory in _CREDIT_BASES:
        t = factory()
        for tree in (t.entry_seed, t.exit_seed):
            s = to_str(tree)
            reparsed = from_sexpr(s)
            assert to_str(reparsed) == s, f"{t.name}: {s} != {to_str(reparsed)}"
            assert tree.ret_type == GType.BOOL


def test_vrp_compound_threshold_params_are_normalized_sigma_defaults():
    """The static fallback θ are the +1.16-probe's ~1σ IV / ~0.5σ gap (normalized)."""
    seed = _vrp_compound_entry()
    vals = {ch.children[0].defn.name: ch.children[1].value for ch in seed.children}
    assert vals["ATM_IV"] == pytest.approx(1.0)
    assert vals["IVRVGap5d"] == pytest.approx(0.5)
    # custom thresholds flow through
    seed2 = _vrp_compound_entry(iv_threshold_norm=0.8, gap_threshold_norm=0.6)
    vals2 = {ch.children[0].defn.name: ch.children[1].value for ch in seed2.children}
    assert vals2["ATM_IV"] == pytest.approx(0.8)
    assert vals2["IVRVGap5d"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# (6) RPB (debit) deliberately NOT VRP-seeded — documented divergence
# ---------------------------------------------------------------------------

def test_rpb_debit_template_not_vrp_seeded():
    """The ratio_put_backspread is a DEBIT structure (premium-BUYING); the VRP gate
    is a premium-SELLING signal that inverts its thesis (measured -0.52 Sharpe).
    Its entry is deliberately NOT the compound VRP form (it keeps a debit-
    appropriate IV+down-move entry); the per-fold pool still lets it re-derive."""
    t = ratio_put_backspread_base()
    assert not t.is_credit
    assert not _is_vrp_compound(t.entry_seed), (
        "RPB should NOT carry the premium-selling VRP gate (thesis inversion)")
    # It still references ATM_IV (high-IV gate) — but combined with SessionReturn,
    # not IVRVGap5d.
    s = to_str(t.entry_seed)
    assert "ATM_IV" in s and "SessionReturn" in s
