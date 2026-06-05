"""Proxy survival gates — ANNOTATE-DON'T-DESTROY (2026-06-01, USER DECISION).

The GP run must ALWAYS yield the FULL Pareto front (champions) for inspection —
NEVER an empty fleet from a hard survival filter. `_apply_proxy_survival_gates`
therefore DROPS NOTHING: it returns the full input fleet, every member TAGGED with
which gates it passes:

  * positivity_passed  — val_sharpe > 0            (the only old G1 *filter* signal)
  * g3_passed          — n_trades_val >= G3 (40)
  * g10_passed         — WFE >= 0.3 (when train>0.1), False on negative-train/positive-
                         val overfit, True (N/A) in the gray zone / no-data
  * g11_passed         — mc_median_sharpe > floor when annotated, else True (N/A)
  * g9_passed          — the per-correlation-group champion (1 per group, among the
                         survival_passed set, ranked dd-adjusted)
  * g5_passed          — worst-regime Sharpe >= floor (True when no regime data)
  * survival_passed    — COMPOSITE = positivity AND g3 AND g10 AND g11 (the STRICT
                         set a CONFIRMATORY run would filter on)

Advisory-only (NOT part of survival_passed, by design, else they empty the fleet):
  * g2_persisted       — same canonical strategy positive in >=2 folds (+ structurally
                         similar)
  * val_dsr_passed     — cross-fold VAL DSR significance (P(SR>SR*) >= 0.05)

These tests pin annotate-don't-destroy: previously-DROPPED individuals (negative
Sharpe, low trades, bad WFE, non-transferable) are now KEPT but tagged
survival_passed=False (and the specific gate bool False); a fully-passing individual
is tagged survival_passed=True; and the returned front size ALWAYS equals the input
front size (nothing dropped).

Also pins the 2026-06-01 audit fixes that survive the reframing:
  * HIGH-5: the G9 champion sort (1-per-correlation-group) ranks on the DRAWDOWN-
    ADJUSTED val Sharpe (same penalty as the evolution objective).
  * MEDIUM-7: G3_min_trades_val == 40 (aligned with the trade-count objective).
"""
from layer2.experiment import _apply_proxy_survival_gates
from layer2.fitness import DRAWDOWN_FREE_LEVEL, DRAWDOWN_PENALTY_WEIGHT


def _base(name, val_sharpe=0.5, train_sharpe=0.5, **extra):
    """A strategy dict that is survival_passed=True unless overridden."""
    s = {
        "val_sharpe": val_sharpe, "train_sharpe": train_sharpe,
        "folds_positive": 2, "structurally_similar": True,
        "n_trades_val": 150, "template_name": name,
    }
    s.update(extra)
    return s


# ====================== ANNOTATE-DON'T-DESTROY INVARIANTS ===================== #

def test_full_front_returned_nothing_dropped():
    """The returned front size ALWAYS equals the input size — annotate-don't-destroy
    keeps EVERY individual (mix of passing and failing members)."""
    fleet = [
        _base("iron_condor", val_sharpe=0.5),            # fully passing
        _base("bull_put_credit", val_sharpe=-0.4),       # negative -> survival False
        _base("bear_call_credit", n_trades_val=10),      # low trades -> survival False
        _base("iron_butterfly", train_sharpe=-0.3),      # overfit-to-val -> survival False
    ]
    survivors = _apply_proxy_survival_gates(fleet, n_trials=0, n_val_days=125)
    assert len(survivors) == len(fleet), "annotate-don't-destroy must drop NOTHING"
    # every member carries the full tag schema
    for s in survivors:
        for tag in ("positivity_passed", "g3_passed", "g10_passed", "g11_passed",
                    "g9_passed", "g5_passed", "survival_passed", "g2_persisted",
                    "gates_passed"):
            assert tag in s, f"missing tag {tag!r}"


def test_fully_passing_individual_is_survival_passed():
    """A clean individual (positive val, positive train WFE=1.0, >=40 trades, no mc
    annotation) is tagged survival_passed=True with all component bools True."""
    s = _base("iron_condor", val_sharpe=0.5, train_sharpe=0.5, n_trades_val=150)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    r = survivors[0]
    assert r["survival_passed"] is True
    assert r["positivity_passed"] is True
    assert r["g3_passed"] is True
    assert r["g10_passed"] is True
    assert r["g11_passed"] is True            # N/A (no mc) -> passes
    assert "G1" in r["gates_passed"] and "G3" in r["gates_passed"]
    assert "G10" in r["gates_passed"]          # WFE applied + passed


# ----------------------------- G10 (was: HIGH-7) --------------------------- #

def test_g10_keeps_but_flags_negative_train_positive_val():
    """train_sharpe STRICTLY < 0 with positive val is overfit-to-val. ANNOTATE-DON'T-
    DESTROY: the individual is KEPT but tagged g10_passed=False and survival_passed
    =False (it would be filtered by a CONFIRMATORY run, not dropped here)."""
    s = _base("bear_call_credit", val_sharpe=0.5, train_sharpe=-0.3)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1, "overfit-to-val individual must be KEPT (annotate-only)"
    assert survivors[0]["g10_passed"] is False
    assert survivors[0]["survival_passed"] is False
    assert "G10" not in survivors[0]["gates_passed"], "no G10 pass-tag when it failed"


def test_g10_zero_train_sharpe_is_na_and_passes():
    """train_sharpe == 0.0 is the absent-field / errored / breakeven sentinel, NOT
    overfit. g10_passed=True (N/A), no G10 tag (WFE not applicable in the gray zone)."""
    s = _base("bear_call_credit", val_sharpe=0.5, train_sharpe=0.0)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g10_passed"] is True, "train==0.0 is the sentinel, not overfit"
    assert survivors[0]["survival_passed"] is True
    assert "G10" not in survivors[0]["gates_passed"], "no G10 tag in the 0..0.1 gray zone"


def test_g10_keeps_positive_train_positive_val():
    """Control: legitimate in-sample edge (WFE = val/train = 1.0 >= 0.3) -> g10_passed
    True, tagged G10."""
    s = _base("bear_call_credit", val_sharpe=0.5, train_sharpe=0.5)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g10_passed"] is True
    assert "G10" in survivors[0]["gates_passed"]


def test_g10_bad_wfe_kept_but_flagged():
    """WFE < 0.3 (lose >70% of in-sample edge OOS) -> KEPT, g10_passed=False,
    survival_passed=False, no G10 pass-tag."""
    # train 1.0, val 0.2 -> WFE = 0.2 < 0.3
    s = _base("bear_call_credit", val_sharpe=0.2, train_sharpe=1.0)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1, "bad-WFE individual must be KEPT (annotate-only)"
    assert survivors[0]["g10_passed"] is False
    assert survivors[0]["survival_passed"] is False
    assert "G10" not in survivors[0]["gates_passed"]


def test_g10_gray_zone_thin_positive_train_is_na():
    """0.0 < train_sharpe <= 0.1 is exempt from the WFE ratio (numerically unstable)
    and from the negative-train clause -> g10_passed=True (N/A), no G10 tag."""
    s = _base("bear_call_credit", val_sharpe=0.5, train_sharpe=0.05)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g10_passed"] is True
    assert survivors[0]["survival_passed"] is True
    assert "G10" not in survivors[0]["gates_passed"], "no G10 tag in the gray zone"


# ----------------------------- G1 positivity (was: HIGH-6) ----------------- #

def test_g1_static_keeps_but_flags_nonpositive_val_sharpe():
    """Static-threshold branch (n_trials=0): zero/negative val Sharpe is KEPT but
    positivity_passed=False and survival_passed=False (annotate-don't-destroy)."""
    for vs in (0.0, -0.5):
        survivors = _apply_proxy_survival_gates(
            [_base("iron_condor", val_sharpe=vs)], n_trials=0, n_val_days=125)
        assert len(survivors) == 1, f"val_sharpe={vs} must be KEPT (annotate-only)"
        assert survivors[0]["positivity_passed"] is False
        assert survivors[0]["survival_passed"] is False
        assert "G1" not in survivors[0]["gates_passed"], "no G1 pass-tag for non-positive"


def test_g1_per_strategy_dsr_keeps_negative_but_flags_it():
    """Per-strategy DSR branch (n_trials>0): a low-variance pool drives SR*->0 so a
    negative Sharpe's DSR can exceed 0.05 — but the negative member is now KEPT
    (annotate-don't-destroy) with positivity_passed=False / survival_passed=False,
    and its val_dsr is annotated. Front size == input size."""
    pool = [_base(f"t{i}", val_sharpe=0.45 + 0.001 * i) for i in range(6)]
    neg = _base("bull_put_credit", val_sharpe=-0.4)
    survivors = _apply_proxy_survival_gates(pool + [neg], n_trials=64, n_val_days=125)
    assert len(survivors) == len(pool) + 1, "nothing dropped"
    rneg = next(s for s in survivors if s["template_name"] == "bull_put_credit")
    assert rneg["positivity_passed"] is False
    assert rneg["survival_passed"] is False
    assert "G1" not in rneg["gates_passed"]
    assert rneg.get("val_dsr") is not None  # DSR annotated even for the negative member


# ----------------------------- G9 (was: HIGH-5) ---------------------------- #

def test_g9_prefers_low_drawdown_sibling_in_same_group():
    """Two neutral_credit strategies: A has higher RAW val Sharpe but huge drawdown;
    B slightly lower Sharpe but tiny drawdown. The dd-adjusted G9 sort must tag B as
    the per-group champion. Both are KEPT; exactly one carries the G9 tag."""
    # A: val 0.50, adj_dd = 3.0/1.0 = 3.0 -> penalty 0.20*(3-1)=0.40 -> adj 0.10
    a = _base("iron_condor", val_sharpe=0.50,
              val_max_drawdown=3.0, val_avg_position_size=1.0)
    # B: val 0.45, adj_dd = 0.5/1.0 = 0.5 -> penalty 0 -> adj 0.45
    b = _base("iron_butterfly", val_sharpe=0.45,
              val_max_drawdown=0.5, val_avg_position_size=1.0)
    survivors = _apply_proxy_survival_gates([a, b], n_trials=0, n_val_days=125)
    assert len(survivors) == 2, "both siblings KEPT (annotate-only)"
    g9 = [s for s in survivors if "G9" in s["gates_passed"]]
    assert len(g9) == 1, "G9 tags exactly 1 per correlation group"
    assert g9[0]["template_name"] == "iron_butterfly", (
        "dd-adjusted sort must prefer the low-drawdown sibling, not the raw-Sharpe one")
    assert g9[0]["g9_passed"] is True
    # the raw sort WOULD have picked A (so the fix changed the outcome)
    assert a["val_sharpe"] > b["val_sharpe"]
    other = next(s for s in survivors if s["template_name"] == "iron_condor")
    assert other["g9_passed"] is False, "non-champion sibling tagged g9_passed=False"


def test_g9_dd_sort_backcompat_without_drawdown_fields():
    """Records lacking val_max_drawdown/val_avg_position_size (older runs) degrade to
    raw val_sharpe ordering (penalty 0), not a crash. Both KEPT; higher raw wins G9."""
    a = _base("iron_condor", val_sharpe=0.50)      # no dd fields
    b = _base("iron_butterfly", val_sharpe=0.40)
    survivors = _apply_proxy_survival_gates([a, b], n_trials=0, n_val_days=125)
    assert len(survivors) == 2
    g9 = [s for s in survivors if "G9" in s["gates_passed"]]
    assert len(g9) == 1 and g9[0]["template_name"] == "iron_condor", \
        "with no dd fields, G9 falls back to raw val_sharpe (higher wins)"


def test_g9_champion_only_from_survival_passed_set():
    """A correlation group whose members all FAIL survival (negative Sharpe) gets NO
    G9 champion — G9 marks the deployable champion, not a non-viable one. The members
    are still KEPT (annotate-only), just g9_passed=False."""
    a = _base("iron_condor", val_sharpe=-0.5)
    b = _base("iron_butterfly", val_sharpe=-0.2)
    survivors = _apply_proxy_survival_gates([a, b], n_trials=0, n_val_days=125)
    assert len(survivors) == 2, "non-viable members still KEPT"
    assert all(s["g9_passed"] is False for s in survivors)
    assert all("G9" not in s["gates_passed"] for s in survivors)


def test_g9_dd_penalty_constants_match_evolution():
    """The sort's penalty must use the SAME constants the evolution objective applies
    (fitness.py), else champion selection and search disagree."""
    assert DRAWDOWN_FREE_LEVEL == 1.0
    assert DRAWDOWN_PENALTY_WEIGHT == 0.20


# ----------------------- G2 annotate-only (unchanged) ---------------------- #

def test_g2_does_not_empty_fleet_when_no_cross_fold_persistence():
    """A single-fold candidate (folds_positive==1) SURVIVES — G2 is ANNOTATE-ONLY."""
    s = _base("iron_condor", folds_positive=1, structurally_similar=False)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1, "G2 must not drop a non-persistent strategy (annotate-only)"
    assert survivors[0]["g2_persisted"] is False, "non-persistent → g2_persisted False"
    assert "G2" not in survivors[0]["gates_passed"], "G2 is not a passed HARD gate"
    assert "G2_advisory" not in survivors[0]["gates_passed"], "no advisory tag when not persisted"
    # G2 is advisory: it must NOT pull down survival_passed (other gates pass).
    assert survivors[0]["survival_passed"] is True


def test_g2_persisted_flag_set_when_persistent():
    """A strategy positive in 2+ folds AND structurally similar gets g2_persisted=True
    and the advisory tag — but persistence is NOT required to survive."""
    s = _base("iron_condor", folds_positive=2, structurally_similar=True)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g2_persisted"] is True
    assert "G2_advisory" in survivors[0]["gates_passed"]


def test_g2_structurally_dissimilar_still_survives():
    """folds_positive>=2 but NOT structurally similar → still survives (annotate-only),
    just g2_persisted=False (the prior hard `structurally_similar` filter is gone)."""
    s = _base("iron_condor", folds_positive=3, structurally_similar=False)
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g2_persisted"] is False


# ----------------------- G1 DSR annotate-only ------------------------------ #

def test_g1_dsr_annotate_only_positive_sharpe_survives_low_dsr():
    """A POSITIVE-Sharpe candidate survives even if its DSR is below 0.05 — the
    cross-fold DSR is annotate-only; only positivity is the (now-annotated) signal."""
    pool = [_base(f"t{i}", val_sharpe=0.30 + 0.0001 * i, folds_positive=1,
                  structurally_similar=False) for i in range(8)]
    survivors = _apply_proxy_survival_gates(pool, n_trials=64, n_val_days=125)
    assert len(survivors) == len(pool), "positive-Sharpe candidates must not be DSR-filtered"
    for s in survivors:
        assert "val_dsr" in s and s["val_dsr"] is not None
        assert "val_dsr_passed" in s  # significance ANNOTATED, not enforced
        assert s["positivity_passed"] is True
        assert s["survival_passed"] is True  # DSR significance not part of composite


def test_g1_positivity_flag_records_negative():
    """The negative-Sharpe member is KEPT (annotate-don't-destroy) but flagged
    positivity_passed=False / survival_passed=False, with val_dsr annotated."""
    pool = [_base(f"t{i}", val_sharpe=0.45) for i in range(6)]
    neg = _base("bull_put_credit", val_sharpe=-0.4)
    survivors = _apply_proxy_survival_gates(pool + [neg], n_trials=64, n_val_days=125)
    assert len(survivors) == len(pool) + 1, "nothing dropped"
    rneg = next(s for s in survivors if s["template_name"] == "bull_put_credit")
    assert rneg["positivity_passed"] is False
    assert rneg["survival_passed"] is False
    assert rneg.get("val_dsr") is not None  # annotated even though it fails positivity


def test_g1_carries_skew_kurtosis_into_dsr():
    """The persistence dicts must carry val_return_skew/kurtosis so the DSR reads the
    REAL moments. Two identical-Sharpe pools differing ONLY in tail moments must
    produce DIFFERENT val_dsr — proving skew/kurt are wired, not 0/3."""
    base_pool = lambda sk, ku: [
        _base(f"t{i}", val_sharpe=0.8, val_return_skew=sk, val_return_kurtosis=ku)
        for i in range(6)]
    symmetric = base_pool(0.0, 0.0)
    heavy_neg = base_pool(-2.0, 8.0)  # negative skew + heavy tails -> lower DSR
    _apply_proxy_survival_gates(symmetric, n_trials=64, n_val_days=125)
    _apply_proxy_survival_gates(heavy_neg, n_trials=64, n_val_days=125)
    assert heavy_neg[0]["val_dsr"] < symmetric[0]["val_dsr"], (
        "heavy-tailed / negative-skew returns must yield a LOWER DSR — proves the "
        "real moments are read (skew/kurt wired through, not defaulted to 0/3)")


# ----------------------- G11 transferability ------------------------------- #

def test_g11_keeps_non_transferable_but_flags_it():
    """A non-transferable strategy (mc_median_sharpe <= floor) is KEPT (annotate-don't-
    destroy) but g11_passed=False / survival_passed=False; a robust one (mc>floor) is
    g11_passed=True with the G11 tag; an un-annotated one (no mc) is N/A -> g11_passed
    =True with no G11 tag (discovery no-op)."""
    strats = [
        _base("bear_call_credit", mc_median_sharpe=0.5),   # robust -> g11 True, tagged
        _base("bull_put_credit", mc_median_sharpe=-0.5),   # dies under noise -> g11 False
        _base("iron_condor"),                              # no mc -> N/A -> g11 True, untagged
    ]
    survivors = _apply_proxy_survival_gates(strats, n_trials=0, n_val_days=125)
    assert len(survivors) == 3, "annotate-don't-destroy keeps all 3"
    by = {s["template_name"]: s for s in survivors}
    assert by["bear_call_credit"]["g11_passed"] is True
    assert "G11" in by["bear_call_credit"]["gates_passed"]
    assert by["bear_call_credit"]["survival_passed"] is True
    assert by["bull_put_credit"]["g11_passed"] is False
    assert by["bull_put_credit"]["survival_passed"] is False
    assert "G11" not in by["bull_put_credit"]["gates_passed"]
    assert by["iron_condor"]["g11_passed"] is True      # N/A passes
    assert "G11" not in by["iron_condor"]["gates_passed"]
    assert by["iron_condor"]["survival_passed"] is True


def test_g11_boundary_zero_flagged_not_dropped():
    """Exactly at the floor (mc == 0.0, strictly-positive required) -> g11_passed=False
    / survival_passed=False, but KEPT."""
    survivors = _apply_proxy_survival_gates(
        [_base("bear_call_credit", mc_median_sharpe=0.0)], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g11_passed"] is False
    assert survivors[0]["survival_passed"] is False


# ----------------------- G3 aligned with objective (MEDIUM-7) -------------- #

def test_g3_floor_aligned_with_trade_count_objective():
    """G3_min_trades_val is the val-trade analog of min_trades (2026-06-02 (b):
    lowered 40→20 alongside min_trades 30→20 so SELECTIVE signals can be tagged
    g3_passed/survival_passed). Annotate-only — it flags, it no longer drops."""
    from layer2.experiment import SURVIVAL_GATES
    assert SURVIVAL_GATES["G3_min_trades_val"] == 20, \
        "G3 must be 20 (the val-trade analog of the relaxed min_trades=20)"


def test_g3_search_optimal_low_freq_champion_passes_g3():
    """A ~0.3/day champion (≈40 val trades) — fully rewarded by the optimizer — has
    g3_passed=True; one just below the floor is KEPT but g3_passed=False (annotate-
    don't-destroy: the floor flags, it no longer drops)."""
    s = _base("iron_condor", n_trades_val=25)   # ~0.20/day over 125 days, >= G3=20
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1
    assert survivors[0]["g3_passed"] is True
    assert survivors[0]["survival_passed"] is True
    # one just below the floor (G3=20): KEPT, but g3_passed=False / survival_passed=False
    s2 = _base("iron_butterfly", n_trades_val=15)
    survivors2 = _apply_proxy_survival_gates([s2], n_trials=0, n_val_days=125)
    assert len(survivors2) == 1, "below-floor individual is KEPT (annotate-only)"
    assert survivors2[0]["g3_passed"] is False
    assert survivors2[0]["survival_passed"] is False
    assert "G3" not in survivors2[0]["gates_passed"]


# ----------------------- G5 regime annotate-only --------------------------- #

def test_g5_no_regime_data_passes_by_default():
    """No regime_sharpes -> g5_passed=True (pass by default), G5 tag present."""
    s = _base("iron_condor")
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert survivors[0]["g5_passed"] is True
    assert "G5" in survivors[0]["gates_passed"]


def test_g5_bad_regime_kept_but_flagged():
    """A strategy with a regime Sharpe below the floor is KEPT (annotate-don't-destroy)
    but g5_passed=False, no G5 tag. (G5 is not part of survival_passed.)"""
    s = _base("iron_condor", regime_sharpes={"LOW": 0.4, "HIGH": -0.9})  # -0.9 < -0.3
    survivors = _apply_proxy_survival_gates([s], n_trials=0, n_val_days=125)
    assert len(survivors) == 1, "bad-regime individual must be KEPT (annotate-only)"
    assert survivors[0]["g5_passed"] is False
    assert "G5" not in survivors[0]["gates_passed"]


# ===== SMOKE-RUN PARITY: negative-Sharpe fleet is KEPT (non-empty front) ===== #

def test_negative_sharpe_fleet_yields_nonempty_front():
    """Regression for the EMPTY-fleet bug: a fleet of ALL-negative-Sharpe individuals
    (the case the smoke run's positivity filter used to drop entirely) must now yield
    a NON-EMPTY front — every member KEPT and tagged survival_passed=False. This is the
    annotate-don't-destroy guarantee the operator relies on for inspection."""
    fleet = [_base(f"neg{i}", val_sharpe=-0.1 * (i + 1)) for i in range(5)]
    survivors = _apply_proxy_survival_gates(fleet, n_trials=0, n_val_days=125)
    assert len(survivors) == len(fleet), "front must be NON-EMPTY (full fleet kept)"
    assert all(s["positivity_passed"] is False for s in survivors)
    assert all(s["survival_passed"] is False for s in survivors)
