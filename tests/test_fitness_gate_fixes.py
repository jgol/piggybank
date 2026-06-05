"""Fitness-layer gate fixes from the 2026-06-01 SECOND adversarial audit.

MH5 (finding 5): the per-regime hard gate must be SKIPPED on val/test scoring —
  it fires against TRAIN-fit terciles + a TRAIN-fit per-regime random-entry
  baseline, which are invalid on a regime-shifted OOS window and could sentinel
  the whole val front (collapsing the fleet via val_sharpe reconciliation).
MEDIUM-6 (finding 6, SUPERSEDED 2026-06-02 by BLOCKER B2): originally the DSR
  evolution-trial recorder was gated to record only once the gp_engine min_trades
  ramp completed (min_trades == post-ramp target). B2 REMOVED that gate: a backtest
  that RAN is a multiple-testing trial whether or not it later cleared min_trades,
  and the ramp gate undercounted the DSR trial count N by ~30x (recording did not
  start until ~gen 30), making SR* far too lenient. The recorder now fires from
  gen 0; V is kept valid by evolution_n_v_from_records's n_days >= min_days filter.
  The _target_min_trades snapshot attribute is retained (still set at construction)
  but no longer gates RECORDING.
"""
import numpy as np
import pandas as pd

from layer2.evaluator import BacktestResult
from layer2.fitness import (
    DEFAULT_OBJECTIVES, FAILED_FITNESS_SENTINEL, VectorizedFitnessEvaluator,
    _canonical_signature, _record_trial,
)
from layer2.grammar import from_sexpr
from layer2.io import (
    PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import iron_condor


def _make_data(n_dates=4, bars_per_date=40) -> pd.DataFrame:
    """Synthetic minute data; ATM_IV varied across dates so terciles are non-trivial."""
    rng = np.random.default_rng(0)
    rows = []
    for di, date in enumerate([f"2024-{m:02d}-15" for m in range(1, n_dates + 1)]):
        spot = 4500.0
        iv = 0.12 + 0.04 * di  # rising IV across dates -> distinct regimes
        for w in range(bars_per_date):
            spot += rng.normal(0, 1.0)
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: spot, "ATM_IV": iv,
                "MinutesToClose": 60.0 - w,
                "VIXSpot": 18.0, "VIXTermSlope": -0.5,
                "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": 0.18,
                "PredRV15": 0.30, "PredRV30": 0.30,
                "PredRegime": 0, "PredSpread": 0.0,
            }
            for col in TYPED_VECTOR_COLUMNS:
                row[col] = np.zeros(384, dtype=np.float32)
            for col in REGIME_PROB_COLUMNS:
                row[col] = 0.25
            rows.append(row)
    return pd.DataFrame(rows)


# =============================== MEDIUM-6 ================================= #

def test_target_min_trades_snapshotted_at_construction():
    """__post_init__ snapshots the post-ramp min_trades target. gp_engine ramps
    min_trades DOWN to 1 then back up, reading the constructed value as the target;
    the snapshot must equal that constructed value."""
    from layer2.evaluator_vectorized import prepare_terminal_data
    data = _make_data(n_dates=2, bars_per_date=20)
    td = prepare_terminal_data(data)
    fe = VectorizedFitnessEvaluator(
        template=iron_condor(), data=data, terminal_data=td,
        min_trades=30, regime_gate_enabled=False,
    )
    assert fe._target_min_trades == 30, "target must snapshot the constructed min_trades"


def test_record_trial_not_gated_by_min_trades_ramp():
    """BLOCKER B2: the recorder must add a trial EVEN while min_trades < target
    (mid-ramp). A backtest that produced a finite Sharpe IS a multiple-testing
    trial regardless of the ramp; gating recording on the ramp undercounts N ~30x.

    We monkeypatch the backtest to return a deterministic finite-Sharpe result so
    the test asserts the RECORDER GUARD (not the synthetic backtest outcome): a
    distinct individual seen at min_trades=5 (< target 30) must be recorded."""
    from layer2.evaluator_vectorized import prepare_terminal_data
    import layer2.evaluator_vectorized as _EV
    data = _make_data(n_dates=4, bars_per_date=40)
    td = prepare_terminal_data(data)
    fe = VectorizedFitnessEvaluator(
        template=iron_condor(), data=data, terminal_data=td,
        min_trades=30, regime_gate_enabled=False,
    )
    sink = fe.enable_trial_recording()

    # Deterministic finite-Sharpe backtest result (enough trades to clear gates,
    # entry_fire_rate low so the tautology/near-never gates do not sentinel).
    def _make_result(sharpe):
        return BacktestResult(
            returns=np.array([0.01, -0.005] * 50), trades=[],
            equity_curve=np.zeros(10), max_drawdown=0.05, sharpe=sharpe,
            sortino=sharpe + 0.1, total_trades=50, win_rate=0.6, n_days=4,
            exit_utilization=0.5, entry_fire_rate=0.10,
            entry_fire_rate_flat=0.10, conditional_sharpe_gap=0.2,
            avg_position_size=0.5, max_drawdown_uncapped=0.05,
        )

    entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
    exit_ = from_sexpr("LT(MinutesToClose, EphReal(0.0))")
    size = from_sexpr("EphReal(0.5)")
    _orig_bt = _EV.vectorized_backtest
    try:
        _EV.vectorized_backtest = lambda *a, **kw: _make_result(1.5)
        # gp_engine sets min_trades low during the ramp:
        fe.min_trades = 5  # < target (30)
        fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
        assert len(sink) == 1, (
            "B2: a finite-Sharpe trial during the min_trades ramp MUST be recorded "
            "(N counts every backtest that ran, not just post-ramp ones)"
        )
        # A second DISTINCT individual, still mid-ramp, is also recorded.
        entry2 = from_sexpr("GT(VIXSpot, EphReal(0.0))")
        fe._evaluate_on_data(entry2, exit_, size, fe.data, fe.terminal_data)
        assert len(sink) == 2, "B2: distinct mid-ramp individuals each count as a trial"
        # First-seen-wins still holds: re-evaluating the same individual at the
        # post-ramp target does NOT overwrite the mid-ramp record.
        fe.min_trades = 30
        _EV.vectorized_backtest = lambda *a, **kw: _make_result(9.9)
        fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
        assert len(sink) == 2
        sig1 = _canonical_signature(entry, exit_, size)
        assert sink[sig1][0] == 1.5, "first-seen mid-ramp Sharpe is retained"
    finally:
        _EV.vectorized_backtest = _orig_bt


def test_record_trial_no_ramp_records_normally():
    """When there is no ramp (target == 1, the DEFAULT_MIN_TRADES case), the guard
    `min_trades >= target` is always satisfied, so recording works as before."""
    from layer2.evaluator_vectorized import prepare_terminal_data
    data = _make_data(n_dates=4, bars_per_date=40)
    td = prepare_terminal_data(data)
    fe = VectorizedFitnessEvaluator(
        template=iron_condor(), data=data, terminal_data=td,
        regime_gate_enabled=False,   # min_trades defaults to 1
    )
    assert fe._target_min_trades == fe.min_trades
    sink = fe.enable_trial_recording()
    entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
    exit_ = from_sexpr("LT(MinutesToClose, EphReal(0.0))")
    size = from_sexpr("EphReal(0.5)")
    fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
    # No assertion on count (depends on whether the synthetic backtest trades), but
    # the guard must not have BLOCKED recording: min_trades(1) >= target(1) is True.
    assert fe.min_trades >= fe._target_min_trades


def test_record_trial_helper_first_seen_and_sentinel_skip():
    """The _record_trial helper itself: first-seen wins, sentinel/non-finite skipped
    (unchanged contract — pinned so the guard refactor didn't alter it)."""
    sink = {}
    _record_trial(sink, "sig", 1.5, 120)
    _record_trial(sink, "sig", 9.9, 120)  # second write ignored (first-seen)
    assert sink["sig"] == (1.5, 120)
    _record_trial(sink, "phantom", -1e6, 120)   # churning sentinel -> skipped
    _record_trial(sink, "naninf", float("nan"), 120)
    assert "phantom" not in sink and "naninf" not in sink


# =============================== MH5 ===================================== #

def test_score_on_data_passes_skip_regime_gate():
    """score_on_data must invoke _evaluate_on_data with _skip_regime_gate=True so the
    train-fit regime gate never fires on a val/test split."""
    from layer2.evaluator_vectorized import prepare_terminal_data
    data = _make_data(n_dates=2, bars_per_date=20)
    td = prepare_terminal_data(data)
    fe = VectorizedFitnessEvaluator(
        template=iron_condor(), data=data, terminal_data=td,
    )
    captured = {}
    _orig = fe._evaluate_on_data

    def _spy(*a, **kw):
        captured.update(kw)
        return _orig(*a, **kw)

    fe._evaluate_on_data = _spy  # type: ignore[assignment]
    entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
    exit_ = from_sexpr("LT(MinutesToClose, EphReal(0.0))")
    size = from_sexpr("EphReal(0.5)")
    fe.score_on_data(entry, exit_, size, data, terminal_data=td)
    assert captured.get("_skip_regime_gate") is True, \
        "score_on_data must skip the regime gate on val/test (MH5)"
    assert captured.get("_skip_random_gate") is True, \
        "score_on_data must also skip the random-entry gate (existing behavior)"


def test_skip_regime_gate_bypasses_sentinel_on_val():
    """A regime-gate-FAILING result is sentinel'd by _evaluate_on_data when the gate
    is active, but NOT when _skip_regime_gate=True. This is the val/test path: a
    regime-shifted OOS window must not sentinel an otherwise-valid front member."""
    from layer2.evaluator_vectorized import prepare_terminal_data
    data = _make_data(n_dates=4, bars_per_date=40)
    td = prepare_terminal_data(data)
    fe = VectorizedFitnessEvaluator(
        template=iron_condor(), data=data, terminal_data=td,
        regime_gate_enabled=True,
    )
    # Force a deterministic regime-gate failure by monkeypatching the gate predicate
    # and the backtest to a clean positive result, isolating the gate's effect.
    import layer2.fitness as _F
    _orig_gate = _F._regime_gate_fails
    _orig_bt = _F.vectorized_backtest if hasattr(_F, "vectorized_backtest") else None

    good = BacktestResult(
        returns=np.array([0.01, -0.005] * 50), trades=[],
        equity_curve=np.zeros(10), max_drawdown=0.05, sharpe=1.5, sortino=1.6,
        total_trades=50, win_rate=0.6, n_days=4, exit_utilization=0.5,
        entry_fire_rate=0.10, conditional_sharpe_gap=0.2, avg_position_size=0.5,
        max_drawdown_uncapped=0.05,
    )

    def _fake_bt(*a, **kw):
        return good

    def _always_fail_gate(*a, **kw):
        return True

    entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
    exit_ = from_sexpr("LT(MinutesToClose, EphReal(0.0))")
    size = from_sexpr("EphReal(0.5)")
    try:
        _F._regime_gate_fails = _always_fail_gate
        # patch the symbol the method imports locally
        import layer2.evaluator_vectorized as _EV
        _orig_ev_bt = _EV.vectorized_backtest
        _EV.vectorized_backtest = _fake_bt
        # With the gate ACTIVE (training path) -> sentinel.
        f_train = fe._evaluate_on_data(entry, exit_, size, fe.data, td)
        assert f_train[0] >= FAILED_FITNESS_SENTINEL, \
            "active regime gate must sentinel a failing result"
        # With the gate SKIPPED (val/test path) -> real fitness, NOT sentinel.
        f_val = fe._evaluate_on_data(entry, exit_, size, fe.data, td,
                                     _skip_random_gate=True, _skip_regime_gate=True)
        assert f_val[0] < FAILED_FITNESS_SENTINEL, \
            "skipped regime gate must NOT sentinel on the val/test path (MH5)"
    finally:
        _F._regime_gate_fails = _orig_gate
        _EV.vectorized_backtest = _orig_ev_bt
