"""F4 deploy-filter trade/day band (2026-06-01 MEDIUM-11 realignment).

The band was realigned (3, 20) -> (0.3, 8.0) to match the L2 optimizer's FULL-REWARD
plateau (`trade_count_score`: 0.3-3.0/day =1.0, decaying to 0 at 8.0), so the deploy
filter and the evolution objective agree. The old 3/day floor rejected every realistic
champion (QC fires ~0.75/day post eval-schedule fix); these tests pin the new behavior
and prove the band edges map to the optimizer's full-reward support (not the wider
(0.1, 0.3) partial-reward ramp, which the floor deliberately excludes).
"""
from scripts.pilot_to_qc_handoff import (
    _apply_stage1_filters, FILTER_TRADES_PER_DAY)
from layer2.fitness import trade_count_score


def _passing_ind(total_trades):
    """An individual that clears F1/F2/F3/F5 so only F4 (trades/day) is exercised."""
    return {"total_nodes": 10, "total_trades": total_trades,
            "fitness": [0.5, 0.5, 0.5]}  # non-sentinel


def _f4(total_trades, n_val_days=125):
    # val_hv>0 (F1), test_hv>=0.5*val_hv (F3) so the chain reaches F4.
    return _apply_stage1_filters(_passing_ind(total_trades),
                                 val_hv=1.0, test_hv=1.0, n_val_days=n_val_days)


def test_band_value_is_realigned():
    assert FILTER_TRADES_PER_DAY == (0.3, 8.0)


def test_f4_accepts_realistic_champion():
    # ~0.75/day — the QC-observed realistic firing rate (94/125 = 0.752).
    assert _f4(94, 125) is None, "a ~0.75/day champion must pass F4"


def test_f4_rejects_too_rare():
    # 0.1/day — noise-Sharpe risk (cf. BPC_wf1: 9 trades -> +2.46 spurious).
    reason = _f4(12, 125)  # 0.096/day
    assert reason is not None and "trades_per_day" in reason


def test_f4_rejects_too_frequent():
    reason = _f4(1250, 125)  # 10/day — past the 8.0 churn ceiling
    assert reason is not None and "trades_per_day" in reason


def test_f4_boundaries_inclusive():
    lo, hi = FILTER_TRADES_PER_DAY
    n = 100
    assert _f4(int(round(lo * n)), n) is None, "lower bound 0.3/day inclusive"
    assert _f4(int(round(hi * n)), n) is None, "upper bound 8.0/day inclusive"


def test_band_edges_are_optimizer_full_reward_support():
    """The whole point of the realignment: F4 = the optimizer's FULL-REWARD plateau
    (0.3-3.0/day, =1.0) extended to its zero-point (8.0/day). The floor (0.3) scores
    ~1.0; the ceiling (8.0) is the zero-point. The band deliberately EXCLUDES the
    (0.1, 0.3) ramp zone (positive but partial reward) as a thin-sample noise risk."""
    lo, hi = FILTER_TRADES_PER_DAY
    n = 1000
    # lo=0.3 sits on the ramp/plateau boundary -> ~1.0 (FP: 0.2/0.2 == 0.9999..).
    assert trade_count_score(int(lo * n), n) >= 0.999        # plateau floor (full reward)
    assert trade_count_score(int(hi * n), n) == 0.0          # decay zero-point
    assert trade_count_score(int(3.0 * n), n) == 1.0         # plateau ceiling inside band
    assert trade_count_score(int(7.0 * n), n) > 0.0          # decay zone still positive
    # just OUTSIDE the band edges -> non-positive reward (the support boundary)
    assert trade_count_score(int(0.15 * n), n) <= 0.5        # below floor: ramp/dead zone
    assert trade_count_score(int(9.0 * n), n) == 0.0         # above ceiling: zero


def test_old_3_20_band_would_have_rejected_the_champion():
    """Regression: the realistic ~0.75/day champion that F4 now ACCEPTS was REJECTED
    by the old (3,20) band — the training-serving contradiction this amendment fixes."""
    tpd = 94 / 125  # 0.752/day
    old_lo, old_hi = (3, 20)
    new_lo, new_hi = FILTER_TRADES_PER_DAY
    assert not (old_lo <= tpd <= old_hi), "old band rejected the realistic champion"
    assert new_lo <= tpd <= new_hi, "new band accepts it"
