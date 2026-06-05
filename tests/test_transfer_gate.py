"""G11 transferability gate: a strategy whose edge does not survive calibrated
terminal noise (mc_median_sharpe <= 0) is rejected; an un-annotated strategy
(mc_median_sharpe absent -> discovery mode) is a no-op. Wires the previously-dead
mc_noise_robustness_gate into the survival pipeline.
"""
from layer2.experiment import _apply_proxy_survival_gates, SURVIVAL_GATES


def _strat(mc, name, val_sharpe=0.5, train_sharpe=0.5):
    # fields needed to pass G1(static)/G2/G3/G10 so we isolate G11.
    # train_sharpe defaults POSITIVE (WFE=val/train=1.0): the 2026-06-01 G10 fix
    # rejects train_sharpe<=0 with positive val (overfit-to-val), so a legitimate
    # strategy under test must have in-sample edge to reach G11.
    s = {
        "val_sharpe": val_sharpe, "folds_positive": 2, "structurally_similar": True,
        "n_trades_val": 150, "train_sharpe": train_sharpe, "template_name": name,
    }
    if mc is not None:
        s["mc_median_sharpe"] = mc
    return s


def test_g11_constant_exists():
    assert "G11_min_mc_median_sharpe" in SURVIVAL_GATES
    assert SURVIVAL_GATES["G11_min_mc_median_sharpe"] == 0.0


def test_g11_rejects_non_transferable_keeps_robust_and_noops_unannotated():
    # three distinct correlation groups so G9 keeps all that pass earlier gates
    strats = [
        _strat(0.5, "bear_call_credit"),   # robust under noise -> kept, tagged G11
        _strat(-0.5, "bull_put_credit"),   # edge dies under noise -> kept, g11 False
        _strat(None, "iron_condor"),       # not annotated (discovery) -> no-op, kept
    ]
    survivors = _apply_proxy_survival_gates(strats, n_trials=0, n_val_days=125)
    # ANNOTATE-DON'T-DESTROY (2026-06-01): non-transferable strategies are no longer
    # DROPPED — they are KEPT in the front but TAGGED g11_passed=False /
    # survival_passed=False. Intent preserved (an edge that dies under calibrated
    # terminal noise is not deployable), now expressed via tags.
    by = {s["template_name"]: s for s in survivors}
    assert "bear_call_credit" in by, "noise-robust strategy must be present"
    assert "bull_put_credit" in by, "non-transferable strategy is KEPT (annotate-only)"
    assert "iron_condor" in by, "un-annotated strategy must be present (G11 no-op)"

    bcc = by["bear_call_credit"]
    bpc = by["bull_put_credit"]
    ic = by["iron_condor"]
    assert bcc["g11_passed"] is True, "noise-robust strategy passes G11"
    assert "G11" in bcc["gates_passed"], "G11 tag present when the gate was applied"
    assert bpc["g11_passed"] is False, "non-transferable strategy fails the G11 tag"
    assert bpc["survival_passed"] is False, "non-transferable strategy is not viable"
    assert "G11" not in bpc["gates_passed"], "no G11 pass-tag when the gate failed"
    assert ic["g11_passed"] is True, "un-annotated strategy is a G11 no-op (N/A passes)"
    assert "G11" not in ic["gates_passed"], "no G11 tag when mc_median_sharpe absent"


def test_g11_boundary_zero_is_rejected():
    # exactly at the floor (mc == 0.0, <= -> not strictly positive). ANNOTATE-DON'T-
    # DESTROY (2026-06-01): the boundary member is KEPT but tagged g11_passed=False /
    # survival_passed=False (intent preserved: zero edge under noise is not viable).
    survivors = _apply_proxy_survival_gates([_strat(0.0, "bear_call_credit")],
                                            n_trials=0, n_val_days=125)
    assert len(survivors) == 1, "boundary member is KEPT (annotate-only)"
    assert survivors[0]["g11_passed"] is False, "mc == floor fails G11 (strictly-positive)"
    assert survivors[0]["survival_passed"] is False, "boundary member is not viable"
    assert "G11" not in survivors[0]["gates_passed"], "no G11 pass-tag at the floor"


def test_g11_annotation_survives_persistence_rebuild():
    """BLOCKER regression (2026-06-01 audit): _compute_cross_fold_persistence rebuilds
    strategy dicts from a field whitelist that originally DROPPED mc_median_sharpe, so
    G11 silently no-op'd in the real pipeline (the other unit tests inject it directly,
    bypassing this rebuild). This proves the annotation now propagates."""
    from layer2.experiment import _compute_cross_fold_persistence
    ind = {"entry_tree": "GT(VIXChange, EphReal(0.0))",
           "exit_tree": "LT(MinutesToClose, EphReal(0.0))", "size_tree": "EphReal(0.5)",
           "delta_tree": None, "val_sharpe": 0.5, "train_sharpe": 0.0,
           "n_trades_val": 150, "total_trades": 150, "mc_median_sharpe": -0.5}
    fr = {1: {"template_results": [{"template_name": "bear_call_credit",
                                    "pareto_front": [ind]}]}}
    persisted = _compute_cross_fold_persistence(fr)
    assert persisted, "expected one persisted record"
    assert persisted[0].get("mc_median_sharpe") == -0.5, \
        "mc_median_sharpe must survive the persistence rebuild (else G11 is dead)"
    assert "delta_tree" in persisted[0], "delta_tree must also propagate for codegen"


def test_g11_fires_after_g10_passes():
    """With a positive train_sharpe (G10 active and passing, val/train WFE=1.0),
    G11 still flags mc<=0 and keeps mc>0 — proves G11 isn't accidentally tied to
    the train_sharpe<=0.1 G10-skip path. ANNOTATE-DON'T-DESTROY (2026-06-01): the
    mc<=0 strategy is KEPT but tagged g11_passed=False / survival_passed=False
    (intent preserved via tags rather than a drop)."""
    s_bad = _strat(-0.5, "bear_call_credit"); s_bad["train_sharpe"] = 0.5  # WFE=0.5/0.5=1.0
    s_good = _strat(0.5, "bull_put_credit"); s_good["train_sharpe"] = 0.5
    survivors = _apply_proxy_survival_gates([s_bad, s_good], n_trials=0, n_val_days=125)
    by = {s["template_name"]: s for s in survivors}
    assert "bull_put_credit" in by and "bear_call_credit" in by, \
        "both KEPT (annotate-don't-destroy)"
    bad = by["bear_call_credit"]
    good = by["bull_put_credit"]
    # G10 applied+passed for BOTH (positive train, WFE=1.0) — proves G11 is the
    # deciding tag, independent of the G10-skip gray zone.
    assert "G10" in bad["gates_passed"], "G10 applies+passes even for the mc<=0 member"
    assert bad["g11_passed"] is False, "mc<=0 fails the G11 tag despite passing G10"
    assert bad["survival_passed"] is False, "mc<=0 strategy is not viable"
    assert "G11" not in bad["gates_passed"], "no G11 pass-tag when the gate failed"
    assert good["g11_passed"] is True
    assert "G10" in good["gates_passed"] and "G11" in good["gates_passed"]
