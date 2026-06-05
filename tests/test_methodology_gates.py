"""Methodology-gate fixes (2026-06-01 holistic review, Batch 3).

MB2: cross-fold persistence is SAME-KEY (the SAME canonical strategy positive in 2+
folds), not per-fold BESTS (different trees each fold over autocorrelated folds).
MH2: F4 deploy rate uses the VAL trade count over val days (one consistent window),
not the train-run count.
"""
from layer2.experiment import _compute_cross_fold_persistence


def _ind(entry, val_sharpe, n_trades_val=150, exit_="LT(MinutesToClose, EphReal(0.0))",
         size="EphReal(0.5)"):
    return {"entry_tree": entry, "exit_tree": exit_, "size_tree": size,
            "delta_tree": None, "val_sharpe": val_sharpe, "n_trades_val": n_trades_val,
            "total_trades": n_trades_val}


def _folds(*per_fold):
    """per_fold[i] = list of inds in fold i+1 (all template iron_condor)."""
    return {i + 1: {"template_results": [{"template_name": "iron_condor",
                                          "pareto_front": inds}]}
            for i, inds in enumerate(per_fold)}


# ----------------------------- MB2: same-key persistence --------------------- #

def test_same_key_positive_in_two_folds_counts_two():
    """The SAME strategy positive in 2 folds -> folds_positive == 2."""
    same = "GT(ATM_IV, EphReal(0.5))"
    fr = _folds([_ind(same, 0.8)], [_ind(same, 0.6)])
    results = _compute_cross_fold_persistence(fr)
    assert results, "expected persisted records"
    # every record for this key carries folds_positive == 2
    assert all(r["folds_positive"] == 2 for r in results), \
        "same canonical strategy positive in 2 folds must count as 2-fold-persistent"
    assert all(r["structurally_similar"] for r in results)


def test_different_keys_each_fold_do_not_aggregate():
    """DIFFERENT strategies winning each fold must NOT be credited as persistent —
    each distinct key only has folds_positive == 1 (the prior per-fold-bests bug)."""
    a = "GT(ATM_IV, EphReal(0.5))"
    b = "LT(VIXSpot, EphReal(0.5))"   # a structurally different strategy
    fr = _folds([_ind(a, 0.8)], [_ind(b, 0.7)])
    results = _compute_cross_fold_persistence(fr)
    assert results
    assert all(r["folds_positive"] == 1 for r in results), \
        "different per-fold winners must NOT aggregate to 2-fold persistence"


def test_commutative_twins_share_a_key():
    """canonical_key collapses commutative reordering: AND(a,b) == AND(b,a) is the
    SAME strategy, so it persists across folds."""
    f1 = "AND(GT(ATM_IV, EphReal(0.5)), LT(VIXSpot, EphReal(0.5)))"
    f2 = "AND(LT(VIXSpot, EphReal(0.5)), GT(ATM_IV, EphReal(0.5)))"  # swapped args
    fr = _folds([_ind(f1, 0.8)], [_ind(f2, 0.6)])
    results = _compute_cross_fold_persistence(fr)
    assert all(r["folds_positive"] == 2 for r in results), \
        "commutative-equivalent trees are the same strategy and must co-persist"


def test_ephemeral_jitter_collapses_to_same_key():
    """campaign review HIGH-2: EphReal JITTER (full-precision threshold noise from
    independent fold evolutions) must collapse to ONE key (_eph_round=2), else G2 is
    structurally dead. Same structure + thresholds differing only in noise digits = 2-
    fold persistent."""
    fr = _folds([_ind("GT(ATM_IV, EphReal(0.5740740740))", 0.8)],
                [_ind("GT(ATM_IV, EphReal(0.5712345678))", 0.6)])
    results = _compute_cross_fold_persistence(fr)
    assert all(r["folds_positive"] == 2 for r in results), \
        "jitter-only threshold differences must collapse to the same key"


def test_genuinely_different_thresholds_stay_distinct():
    """...but thresholds differing in the 1st-2nd decimal are DIFFERENT behavior and
    must NOT collapse (0.50 vs 0.70 = different strategies)."""
    fr = _folds([_ind("GT(ATM_IV, EphReal(0.50))", 0.8)],
                [_ind("GT(ATM_IV, EphReal(0.70))", 0.6)])
    results = _compute_cross_fold_persistence(fr)
    assert all(r["folds_positive"] == 1 for r in results), \
        "genuinely different thresholds must stay distinct keys"


def test_negative_fold_not_counted():
    """A fold where the same key is NEGATIVE does not add to folds_positive."""
    same = "GT(ATM_IV, EphReal(0.5))"
    fr = _folds([_ind(same, 0.8)], [_ind(same, -0.3)])  # fold 2 negative
    results = _compute_cross_fold_persistence(fr)
    assert all(r["folds_positive"] == 1 for r in results), \
        "only POSITIVE folds count toward persistence"


# ----------------------------- MH2: F4 consistent window --------------------- #

def test_f4_rate_uses_val_trade_count():
    """_estimate_trades_per_day must divide the VAL trade count by val days, not the
    train-run total_trades (which is over the much longer anchored train window)."""
    from scripts.pilot_to_qc_handoff import _estimate_trades_per_day
    ind = {"n_trades_val": 100, "total_trades": 800}  # train count 8x the val count
    rate = _estimate_trades_per_day(ind, n_val_days=125)
    assert abs(rate - 100 / 125) < 1e-9, "rate must be val_count/val_days (~0.8), not train"


def test_f4_rate_falls_back_to_total_when_val_absent():
    """Old records without n_trades_val fall back to total_trades (approximate)."""
    from scripts.pilot_to_qc_handoff import _estimate_trades_per_day
    rate = _estimate_trades_per_day({"total_trades": 100}, n_val_days=125)
    assert abs(rate - 100 / 125) < 1e-9


def test_f4_rate_none_when_no_counts():
    from scripts.pilot_to_qc_handoff import _estimate_trades_per_day
    assert _estimate_trades_per_day({}, n_val_days=125) is None


# ----------------------- MEDIUM-11: F4 test null-out ------------------------ #

def test_null_out_invalid_test_fields_marks_and_nulls():
    """A VAL-ONLY fold (test_end=None) must have test_* NULLED and test_invalid=True
    so PBO/handoff never read the ~1-day holdout-test noise as OOS evidence."""
    from layer2.experiment import _null_out_invalid_test_fields
    front = [
        {"entry_tree": "x", "test_sharpe": 2.46, "test_sortino": 1.9,
         "n_trades_test": 1, "val_sharpe": 0.8},
        {"entry_tree": "y", "test_sharpe": -3.0, "test_sortino": -2.0,
         "n_trades_test": 1, "val_sharpe": 0.5},
    ]
    out = _null_out_invalid_test_fields(front)
    assert out is front  # mutates in place
    for m in front:
        assert m["test_sharpe"] is None
        assert m["test_sortino"] is None
        assert m["n_trades_test"] is None
        assert m["test_invalid"] is True
        # val_* untouched (val IS the OOS evidence for a VAL-ONLY fold)
        assert "val_sharpe" in m and m["val_sharpe"] is not None


def test_f4_fold_def_has_real_holdout():
    """F4 now carries a GENUINE >=20-trading-day untouched test holdout (2026-06-02
    methodology decision: keep the anchored folds but carve a real OOS holdout from
    F4 instead of the prior 1-day/val-only test)."""
    from layer2.experiment import WALK_FORWARD_FOLDS
    f4 = [f for f in WALK_FORWARD_FOLDS if f["fold_id"] == 4]
    assert f4 and f4[0]["test_end"] == "2026-04-09", "F4 must carry a real test holdout"
    assert f4[0]["val_end"] == "2026-03-04", "F4 val_end pulled back to leave >=20 test days"


# ----------------------- MEDIUM-8: DSR inconclusive flag -------------------- #

def test_dsr_front_gate_inconclusive_sets_false_not_none():
    """On a short TRAIN window the front gate must annotate dsr_passed=False (NOT
    None) + dsr_inconclusive=True, so the canonical `dsr_passed is True` reader fails
    closed and can distinguish 'could not assess' from 'assessed and failed'."""
    from layer2.experiment import _apply_dsr_front_gate
    champs = [{"id": "x", "train_sharpe": 5.0, "val_sharpe": 5.0,
               "train_return_skew": 0.0, "train_return_kurtosis": 0.0}]
    kept, kept_front = _apply_dsr_front_gate(
        list(champs), ["ind_x"], train_T=5, evo_records={"a": (1.0, 300)},
        dsr_gate_enabled=True, dsr_threshold=0.5, template_name="unit", min_days=20)
    assert kept_front == ["ind_x"], "inconclusive must NOT filter"
    assert champs[0]["dsr_passed"] is False
    assert champs[0]["dsr_inconclusive"] is True
    assert champs[0]["dsr"] is None


# ----------------------- MEDIUM-10: methodology limitation notes ------------ #

def test_methodology_notes_documented_in_module():
    """The cross-template G9 / anchored-window / G2-annotate limitations must be
    DOCUMENTED in run_walk_forward's summary (a no-op cross-template gate must not be
    silently relied upon). We assert the literals are present in the source so the
    REQUIRED-MANUAL-STEP note cannot be silently deleted."""
    import inspect
    from layer2.experiment import run_walk_forward
    src = inspect.getsource(run_walk_forward)
    assert "methodology_notes" in src
    assert "G9_cross_template" in src
    assert "REQUIRED MANUAL" in src or "required manual" in src.lower()
    assert "anchored_expanding_non_independence" in src
    assert "walk_forward_progress.json" in src  # MEDIUM-10 incremental marker
