"""Per-fold seed re-derivation (P1-A: look-ahead fix for Level-B base-template
entry seeds grid-searched on the full dataset). The candidate pool is a FIXED
hypothesis space (no leak); per-fold selection on train data removes the leak.
"""
import pandas as pd

from layer2.experiment import candidate_entry_seeds, derive_per_fold_entry_seed
from layer2.grammar import GType, to_str
from layer2.templates import base_template_by_name


def test_candidate_pool_is_fixed_valid_bool_trees():
    pool = candidate_entry_seeds()
    assert len(pool) >= 8, "expected a pool of simple entry conditions"
    for t in pool:
        assert t.ret_type == GType.BOOL, f"candidate not BOOL-typed: {to_str(t)}"
    strs = [to_str(t) for t in pool]
    assert len(set(strs)) == len(strs), "candidate pool has duplicates"
    # must include the IV x SessionReturn AND-combos the real 2026-05-19 seeds use
    assert any(("ATM_IV" in s and "SessionReturn" in s and "AND" in s) for s in strs)


def test_candidate_pool_carries_no_lookahead():
    """The pool is fixed across calls — a hypothesis space, not data-derived."""
    a = [to_str(t) for t in candidate_entry_seeds()]
    b = [to_str(t) for t in candidate_entry_seeds()]
    assert a == b


def test_derive_falls_back_to_template_seed_when_none_qualify():
    """With data that makes every candidate backtest fail/under-trade, the selector
    safely returns the template's existing entry_seed (no crash, no empty seed)."""
    tpl = base_template_by_name("iron_condor")
    out = derive_per_fold_entry_seed(tpl, pd.DataFrame(), {}, min_trades=20)
    assert to_str(out) == to_str(tpl.entry_seed)
