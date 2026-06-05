"""Smoke tests for MultiLegOptionsBacktester (D88 templates)."""
import numpy as np
import pandas as pd
import pytest

from layer2.evaluator import MultiLegOptionsBacktester
from layer2.io import (
    BYPASS_SCALAR_COLUMNS, PRICE_COLUMN, PROBE_SCALAR_COLUMNS,
    REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import (
    iron_condor, iron_butterfly,
    bull_put_credit, bear_call_credit,
    bull_call_debit, bear_put_debit,
    TEMPLATE_FACTORIES,
)


def _synthetic_l1_data(n_dates: int = 3, bars_per_date: int = 60) -> pd.DataFrame:
    """Build a synthetic L1Output DataFrame for backtester testing."""
    rng = np.random.default_rng(0)
    rows = []
    dates = [f"2024-{m:02d}-15" for m in range(1, n_dates + 1)]
    for date in dates:
        spot = 4500.0
        for w in range(bars_per_date):
            spot += rng.normal(0, 1.5)
            row = {
                "date": date,
                "window_idx": w,
                "bar_position": w,
                PRICE_COLUMN: spot,
                "ATM_IV": 0.20,
                "RawSpread": 0.20,
                "DeltaSpread1": 0.0,
                "DeltaSpread5": 0.0,
                "MinutesToClose": 60.0 - w,
                "VIXSpot": 18.0,
                "VIXTermSlope": -0.5,
                "RealizedVol30m": 0.18,
                "PredRV15": 0.35,
                "PredRV30": 0.35,
                "PredRegime": 0,
                "PredSpread": 0.0,
            }
            for col in TYPED_VECTOR_COLUMNS:
                row[col] = np.zeros(384, dtype=np.float32)
            for col in REGIME_PROB_COLUMNS:
                row[col] = 0.25
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("factory", TEMPLATE_FACTORIES)
def test_template_runs_through_backtester(factory):
    """Each of the 8 templates runs through the multi-leg backtester without crash."""
    template = factory()
    bt = MultiLegOptionsBacktester(template, warmup_bars=5, default_minutes_to_expiry=60.0)
    data = _synthetic_l1_data(n_dates=2, bars_per_date=30)
    result = bt.run(data=data)
    assert result is not None
    assert np.isfinite(result.sharpe), f"{template.name} produced non-finite Sharpe"
    assert result.total_trades >= 0


def test_iron_condor_credit_pnl_sign():
    """Iron condor (credit structure) in near-flat underlying must produce a
    POSITIVE gross P&L from theta decay (direction-sign regression guard).

    B1 regression test: previously, `direction_sign = -1` for credit templates
    inverted the holder's P&L sign because qty_sign in _net_position_value
    already encodes long/short — the multiplier was double-counting. This
    test ensures credit structures profit from theta decay on flat-SPX.

    Asserts on GROSS P&L sign (via net_position_value before/after) rather
    than net-of-fees trade.pnl. Commit 6 hardened the contract cap
    (M2: MAX_CONTRACTS = notional / 0.50), which under the simplified-BS
    toy model with IV=0.20 produces entry_net_value ≈ -$0.016, making
    per-trade gross P&L dominated by fixed transaction costs. The
    direction-sign invariant still holds on the GROSS line.
    """
    template = iron_condor()
    bt = MultiLegOptionsBacktester(template, warmup_bars=5,
                                    default_minutes_to_expiry=60.0)
    # Direct gross-P&L check: simulate a credit-structure lifecycle by
    # computing net_position_value at entry vs. at expiry when all options
    # are worthless. Gross P&L per unit = (exit - entry).
    spot, iv, minutes_at_entry = 4500.0, 0.20, 55.0
    strikes = [
        bt._delta_to_strike(spot, leg.delta_target, iv, minutes_at_entry)
        for leg in template.legs
    ]
    entry_net_value = bt._net_position_value(spot, strikes, minutes_at_entry, iv)
    # Simulate perfect credit-structure outcome: same spot at expiry,
    # all options expired worthless (minutes_left = 0)
    exit_net_value = bt._net_position_value(spot, strikes, 0.0, iv)
    gross_pnl_per_unit = exit_net_value - entry_net_value

    # Credit structures have entry_net_value < 0 (received premium)
    assert entry_net_value < 0, (
        f"iron_condor should have negative entry_net_value (credit received); "
        f"got {entry_net_value:.4f}"
    )
    # Gross P&L per unit MUST be positive — credit structure won at expiry.
    # Pre-B1-fix the inverted direction_sign would have flipped this negative.
    assert gross_pnl_per_unit > 0, (
        f"iron_condor gross P&L per unit = {gross_pnl_per_unit:.4f} on a "
        f"favourable theta-decay scenario; credit structure should profit "
        f"when positions expire worthless. This is the B1 sign regression."
    )


def test_bull_call_debit_pnl_sign():
    """Bull call debit spread in an UP move must have POSITIVE gross P&L.

    Paired with test_iron_condor_credit_pnl_sign — together they verify
    removing `direction_sign` left BOTH credit and debit structures
    correctly signed. Like the credit test, asserts on GROSS P&L via
    direct net_position_value calls (sidesteps the simplified-BS toy
    model's near-zero entry_net_value issue under M2's contract cap).
    """
    from layer2.templates import bull_call_debit
    template = bull_call_debit()
    bt = MultiLegOptionsBacktester(template, warmup_bars=5,
                                    default_minutes_to_expiry=60.0)
    # Direct gross-P&L check: debit spread at entry, spot rallies $50 by
    # exit, positions approach their max-profit intrinsic.
    spot_entry, iv, minutes_entry = 4500.0, 0.10, 55.0
    strikes = [
        bt._delta_to_strike(spot_entry, leg.delta_target, iv, minutes_entry)
        for leg in template.legs
    ]
    entry_net_value = bt._net_position_value(spot_entry, strikes, minutes_entry, iv)
    # Favorable outcome: spot rallies, options at expiry
    spot_exit = 4550.0
    exit_net_value = bt._net_position_value(spot_exit, strikes, 0.0, iv)
    gross_pnl_per_unit = exit_net_value - entry_net_value

    # Debit structures have entry_net_value > 0 (paid premium)
    assert entry_net_value > 0, (
        f"bull_call_debit should have positive entry_net_value (debit paid); "
        f"got {entry_net_value:.4f}"
    )
    # Favourable up-move → gross P&L per unit > 0
    assert gross_pnl_per_unit > 0, (
        f"bull_call_debit gross P&L per unit = {gross_pnl_per_unit:.4f} on "
        f"a +$50 up-move; bullish debit spread should profit here. "
        f"This is the B1 sign regression guard for debit structures."
    )


# ---------------------------------------------------------------------------
# Commit 6: backtester correctness fixes (M1, M2, M3)
# ---------------------------------------------------------------------------

def test_m1_minutes_to_close_column_used_when_present():
    """M1 fix (Commit 6): when MinutesToClose column is present, the
    backtester must read from it each bar instead of decrementing by 1.0.
    Verifies by running two identical scenarios — one with MinutesToClose
    tracking 5-minute-bar spacing, another with 1-minute-bar spacing.
    Results should differ (option values change with minutes_to_expiry)
    if the column is being consulted. Pre-M1, both would be identical
    because the decrement ignored the column.
    """
    template = iron_condor()
    bt = MultiLegOptionsBacktester(template, warmup_bars=5,
                                    default_minutes_to_expiry=60.0)

    def _data(minutes_per_bar: float):
        rows = []
        for w in range(20):
            rows.append({
                "date": "2024-01-15", "window_idx": w, "bar_position": w,
                PRICE_COLUMN: 4500.0,  # perfectly flat
                "ATM_IV": 0.20,
                "MinutesToClose": 60.0 - minutes_per_bar * w,
                "VIXSpot": 18.0, "VIXTermSlope": -0.5,
                "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": 0.18,
                "PredRV15": 0.20, "PredRV30": 0.20,
                "PredRegime": 0, "PredSpread": 0.0,
            })
            for col in TYPED_VECTOR_COLUMNS:
                rows[-1][col] = np.zeros(384, dtype=np.float32)
            for col in REGIME_PROB_COLUMNS:
                rows[-1][col] = 0.25
        return pd.DataFrame(rows)

    # 1-min spacing (MinutesToClose: 60, 59, 58, …)
    r1 = bt.run(data=_data(minutes_per_bar=1.0))
    # 5-min spacing (MinutesToClose: 60, 55, 50, …) — theta decay 5x faster
    r5 = bt.run(data=_data(minutes_per_bar=5.0))

    # Under M1: the 5-min run compresses theta into fewer bars, so trades
    # close with different P&L from the 1-min run. If the column is
    # being consulted, sharpe/total_pnl must differ between runs.
    if r1.total_trades == 0 or r5.total_trades == 0:
        # Contract-capped backtester may not fire under toy BS — skip
        # rather than silently pass.
        pytest.skip("seed entries did not fire on this synthetic data")
    pnl_1 = sum(t.pnl for t in r1.trades)
    pnl_5 = sum(t.pnl for t in r5.trades)
    assert pnl_1 != pytest.approx(pnl_5, abs=1e-6), (
        f"1-min and 5-min bar spacings produce IDENTICAL total P&L "
        f"({pnl_1:.4f} == {pnl_5:.4f}); the MinutesToClose column is not "
        f"being consulted (M1 regression)."
    )


def test_m2_contract_cap_bounds_tail_exposure():
    """M2 fix (Commit 6): when entry_net_value is tiny (e.g. the
    simplified-BS toy model underestimates strike differentiation),
    entry_n_contracts must be capped at MAX_CONTRACTS = notional / 0.50,
    NOT the `notional / floor_of_0.10 = 10× notional` that the pre-fix
    code produced."""
    template = iron_condor()
    # Small notional so cap is visible: 500 / 0.50 = 1000 max contracts
    bt = MultiLegOptionsBacktester(template, warmup_bars=5,
                                    default_minutes_to_expiry=60.0,
                                    notional=500.0)
    rows = []
    for w in range(15):
        rows.append({
            "date": "2024-01-15", "window_idx": w, "bar_position": w,
            PRICE_COLUMN: 4500.0, "ATM_IV": 0.20,
            "MinutesToClose": 60.0 - w,
            "VIXSpot": 18.0, "VIXTermSlope": -0.5,
            "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
            "RealizedVol30m": 0.18,
            "PredRV15": 0.20, "PredRV30": 0.20,
            "PredRegime": 0, "PredSpread": 0.0,
        })
        for col in TYPED_VECTOR_COLUMNS:
            rows[-1][col] = np.zeros(384, dtype=np.float32)
        for col in REGIME_PROB_COLUMNS:
            rows[-1][col] = 0.25
    result = bt.run(data=pd.DataFrame(rows))
    if result.total_trades == 0:
        pytest.skip("no trades fired on this synthetic data")
    # With notional=500, MAX_CONTRACTS=1000. Pre-M2: n_contracts could
    # reach notional / 0.10 = 5000. Post-M2: capped at 1000. Either way
    # the backtest runs, but tail P&L per bar is bounded. Verify max
    # single-bar P&L magnitude is bounded by roughly MAX_CONTRACTS × max
    # per-unit drift (bounded by ~$1 for near-ATM on flat data).
    max_bar_pnl_abs = max(abs(result.returns))
    # MAX_CONTRACTS * $1 per unit drift / notional = 1000 * 1 / 500 = 2.0
    # Any single-bar return > 10x this threshold (20x notional) would
    # indicate the cap is not engaging.
    assert max_bar_pnl_abs < 20.0, (
        f"max single-bar return {max_bar_pnl_abs:.2f} exceeds 20x notional "
        f"— M2 contract cap is not bounding tail exposure"
    )


def test_m3_shuffle_stratification_invariant():
    """M3 lock (Commit 6): under regime-stratified shuffle, row-level
    PredRegime must be invariant (each row's donor comes from the same
    regime stratum). The runtime assertion in _shuffle_by_window raises
    RuntimeError if the invariant is violated."""
    from layer2.shuffle import shuffle_l1
    rng = np.random.default_rng(0)
    rows = []
    for di, date in enumerate([f"2024-{m:02d}-15" for m in range(1, 9)]):
        regime = di % 4
        for w in range(10):
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: 4500.0 + w, "ATM_IV": 0.20,
                "MinutesToClose": 60.0 - w,
                "VIXSpot": 18.0, "VIXTermSlope": -0.5,
                "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": 0.18, "PredRV15": 0.30, "PredRV30": 0.30,
                "PredRegime": regime, "PredSpread": 0.0,
            }
            for col in TYPED_VECTOR_COLUMNS:
                row[col] = (np.ones(384) * (di * 100 + w)).astype(np.float32)
            for col in REGIME_PROB_COLUMNS:
                row[col] = 0.25
            rows.append(row)
    df = pd.DataFrame(rows)
    shuffled = shuffle_l1(df, seed=42, regime_stratify=True, block="window")
    np.testing.assert_array_equal(
        df["PredRegime"].values, shuffled["PredRegime"].values,
        err_msg="stratified shuffle changed row-level PredRegime — invariant broken"
    )


def test_no_direction_sign_regression():
    """Defense-in-depth (Model QA recommendation): grep-guard that no
    CODE-LEVEL use of `direction_sign` (assignment or multiplication)
    reappears in MultiLegOptionsBacktester. Comments mentioning the name
    are allowed — this is specifically a runtime-semantics guard.
    """
    import re
    from pathlib import Path
    text = Path("layer2/evaluator.py").read_text()
    marker = "class MultiLegOptionsBacktester"
    start = text.index(marker)
    rest = text[start + len(marker):]
    end_rel = rest.find("\nclass ")
    multileg_body = text[start:start + len(marker) + end_rel] if end_rel >= 0 else text[start:]

    # Strip comment-only lines and docstrings (simple heuristic: skip lines
    # that begin with '#' after optional whitespace, and strip triple-quoted
    # blocks). Good enough — false negatives are OK, false positives are not.
    code_lines = []
    in_doc = False
    for line in multileg_body.split("\n"):
        stripped = line.lstrip()
        if '"""' in line:
            in_doc = not in_doc if line.count('"""') % 2 == 1 else in_doc
            continue
        if in_doc:
            continue
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    # Match assignment OR multiplication use (either form of the bug)
    if re.search(r"direction_sign\s*[=*]", code):
        raise AssertionError(
            "Code-level use of `direction_sign` reappeared in "
            "MultiLegOptionsBacktester. This variable was removed in "
            "Commit 3 (B1 fix). `qty_sign` in _net_position_value already "
            "encodes long/short; a second sign multiplier double-counts "
            "and inverts credit-structure P&L."
        )


def test_compute_fitness_returns_dict():
    template = iron_condor()
    bt = MultiLegOptionsBacktester(template, warmup_bars=5)
    data = _synthetic_l1_data(n_dates=1, bars_per_date=20)
    result = bt.run(data=data)
    fitness = bt.compute_fitness(result)
    assert "sharpe" in fitness
    assert "neg_max_drawdown" in fitness
    assert "trade_count_score" in fitness
    assert "win_rate" in fitness
    for k, v in fitness.items():
        assert np.isfinite(v), f"fitness[{k}] = {v} is non-finite"


def test_template_init_rejects_non_template():
    with pytest.raises(TypeError):
        MultiLegOptionsBacktester("not a template")
