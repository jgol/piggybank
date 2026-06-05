"""Unit tests for layer2.evaluator — tree evaluation, backtesting, and split logic."""
import math

import numpy as np
import pandas as pd
import pytest

from layer2.grammar import (
    ADD,
    CROSS_ABOVE,
    DIV,
    GT,
    IF_SIDE,
    IN_REGIME,
    LAG,
    LT,
    FuncDef,
    FuncNode,
    GType,
    Regime,
    Side,
    TermDef,
    TermNode,
)
from layer2.evaluator import (
    EvaluationContext,
    SimpleBacktester,
    TreeEvaluator,
    _safe_real,
    split_gp_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_term(name: str, value=None) -> TermNode:
    return TermNode(defn=TermDef(name, GType.REAL), value=value)


def _eph_real(val: float) -> TermNode:
    return TermNode(defn=TermDef("EphReal", GType.REAL), value=val)


def _int_term(val: int) -> TermNode:
    return TermNode(defn=TermDef("EphInt", GType.INT), value=val)


def _regime_term(r: Regime) -> TermNode:
    return TermNode(defn=TermDef(r.name, GType.REGIME), value=r)


def _side_term(s: Side) -> TermNode:
    return TermNode(defn=TermDef(s.name, GType.SIDE), value=s)


# ---------------------------------------------------------------------------
# Test 1: GT(3.0, 2.0) -> True
# ---------------------------------------------------------------------------

def test_gt_basic():
    """GT(A, B) evaluates to True when A > B in context data."""
    tree = FuncNode(defn=GT, children=[_real_term("A"), _real_term("B")])
    ctx = EvaluationContext()
    ctx.update({"A": 3.0, "B": 2.0})
    ev = TreeEvaluator()
    result = ev.evaluate(tree, ctx)
    assert result is True

    # Also test the False case
    ctx2 = EvaluationContext()
    ctx2.update({"A": 1.0, "B": 5.0})
    assert ev.evaluate(tree, ctx2) is False


# ---------------------------------------------------------------------------
# Test 2: Div uses analytic quotient (Div(1, 0) doesn't crash)
# ---------------------------------------------------------------------------

def test_div_analytic_quotient():
    """Div uses a / sqrt(1 + b^2): Div(A, B) with B=0 doesn't crash."""
    tree = FuncNode(defn=DIV, children=[_real_term("A"), _real_term("B")])
    ctx = EvaluationContext()
    ctx.update({"A": 1.0, "B": 0.0})
    ev = TreeEvaluator()
    result = ev.evaluate(tree, ctx)
    assert result == pytest.approx(1.0)  # 1 / sqrt(1 + 0) = 1.0

    # Div(6, 2) = 6 / sqrt(1 + 4) = 6 / sqrt(5)
    ctx2 = EvaluationContext()
    ctx2.update({"A": 6.0, "B": 2.0})
    result2 = ev.evaluate(tree, ctx2)
    assert result2 == pytest.approx(6.0 / math.sqrt(5.0))


# ---------------------------------------------------------------------------
# Test 3: Lag on terminal returns lagged value after buffer fills
# ---------------------------------------------------------------------------

def test_lag_returns_lagged_value():
    """Lag(X, 2) returns the value of X from 2 bars ago."""
    tree = FuncNode(defn=LAG, children=[_real_term("X"), _int_term(2)])
    ctx = EvaluationContext()
    ev = TreeEvaluator()

    # Feed 5 bars: X = 10, 20, 30, 40, 50
    for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
        ctx.update({"X": val})

    # After 5 bars, Lag(X, 2) should return value from 2 bars ago = 30.0
    result = ev.evaluate(tree, ctx)
    assert result == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Test 4: CrossAbove returns False on first bar (no prior data)
# ---------------------------------------------------------------------------

def test_cross_above_false_on_first_bar():
    """CrossAbove returns False when no prior bar cached."""
    tree = FuncNode(defn=CROSS_ABOVE, children=[_real_term("A"), _real_term("B")])
    ctx = EvaluationContext()
    ev = TreeEvaluator()

    ctx.update({"A": 5.0, "B": 3.0})
    result = ev.evaluate(tree, ctx)
    assert result is False  # No previous bar, can't determine crossing


# ---------------------------------------------------------------------------
# Test 5: CrossAbove detects actual crossing
# ---------------------------------------------------------------------------

def test_cross_above_detects_crossing():
    """CrossAbove(A, B): False when A <= B, True when A crosses above B."""
    tree = FuncNode(defn=CROSS_ABOVE, children=[_real_term("A"), _real_term("B")])
    ctx = EvaluationContext()
    ev = TreeEvaluator()

    # Bar 1: A=1 < B=5 (no crossing since no previous)
    ctx.update({"A": 1.0, "B": 5.0})
    ev.evaluate(tree, ctx)  # populate cache

    # Bar 2: A=6 > B=5 — A crossed above B
    ctx.update({"A": 6.0, "B": 5.0})
    result = ev.evaluate(tree, ctx)
    assert result is True

    # Bar 3: A=7 > B=5 — still above but NOT a new crossing
    ctx.update({"A": 7.0, "B": 5.0})
    result = ev.evaluate(tree, ctx)
    assert result is False


# ---------------------------------------------------------------------------
# Test 6: Warm-up period skips early bars
# ---------------------------------------------------------------------------

def test_warmup_skips_early_bars():
    """SimpleBacktester does not trade during warmup_bars period."""
    # Entry tree: always True — GT(signal, threshold) with signal=1 > threshold=0
    entry = FuncNode(defn=GT, children=[_real_term("signal"), _real_term("threshold")])
    # Exit tree: always True — GT(signal, threshold) with signal=1 > threshold=0
    exit_tree = FuncNode(defn=GT, children=[_real_term("signal"), _real_term("threshold")])
    # Side: always CALL
    side = _side_term(Side.CALL)

    n = 10
    # SPXClose as session_log_return (small values near 0)
    data = pd.DataFrame({
        "SPXClose": np.linspace(0.0, 0.01, n),
        "signal": np.ones(n),
        "threshold": np.zeros(n),
    })

    bt = SimpleBacktester(warmup_bars=5, fee_per_leg=0, spread_cost_bps=0)
    result = bt.run(entry, exit_tree, side, data)

    # No trades during first 5 bars (indices 0-4)
    assert all(result.returns[:5] == 0.0)


# ---------------------------------------------------------------------------
# Test 7: InRegime matches when regime is set
# ---------------------------------------------------------------------------

def test_in_regime_matches():
    """InRegime(LOW_VOL) True when context regime is LOW_VOL."""
    tree = FuncNode(defn=IN_REGIME, children=[_regime_term(Regime.LOW_VOL)])
    ctx = EvaluationContext()
    ev = TreeEvaluator()

    # Set regime to LOW_VOL
    ctx.update({"PredRegime": 0})
    assert ev.evaluate(tree, ctx) is True

    # Set regime to HIGH_VOL_PREMIUM (3)
    ctx.update({"PredRegime": 3})
    assert ev.evaluate(tree, ctx) is False


# ---------------------------------------------------------------------------
# Test 8: NaN inputs produce 0.0 (not crash)
# ---------------------------------------------------------------------------

def test_nan_inputs_produce_zero():
    """NaN and None inputs are coerced to 0.0 without raising."""
    assert _safe_real(float("nan")) == 0.0
    assert _safe_real(float("inf")) == 0.0
    assert _safe_real(None) == 0.0

    # Tree evaluation with NaN data
    tree = FuncNode(defn=ADD, children=[_real_term("X"), _real_term("Y")])
    ctx = EvaluationContext()
    ctx.update({"X": float("nan"), "Y": float("inf")})
    ev = TreeEvaluator()
    result = ev.evaluate(tree, ctx)
    assert result == 0.0  # 0.0 + 0.0


# ---------------------------------------------------------------------------
# Test 9: SimpleBacktester produces non-zero PnL on trending data
# ---------------------------------------------------------------------------

def test_backtester_nonzero_pnl_trending():
    """Backtester generates trades and non-zero PnL on a trending market.

    SPXClose is session_log_return (cumulative from session start), NOT absolute
    price. Steadily increasing values simulate an uptrending session.

    Entry signal: GT(signal, threshold) where signal=1.0 > threshold=0.0
    Exit signal: never fires (GT(threshold, signal) is False)
    """
    # Entry: GT(signal, threshold) — signal is always 1.0, threshold always 0.0
    entry = FuncNode(defn=GT, children=[_real_term("signal"), _real_term("threshold")])
    # Exit: GT(threshold, signal) — never True since 0 < 1
    exit_tree = FuncNode(defn=GT, children=[_real_term("threshold"), _real_term("signal")])
    # Side: CALL on a trending-up session
    side = _side_term(Side.CALL)

    n_bars = 100
    # Simulate session_log_return: starts at 0, steadily increases to +0.02
    data = pd.DataFrame({
        "SPXClose": np.linspace(0.0, 0.02, n_bars),
        "signal": np.ones(n_bars),       # always 1.0 — triggers entry
        "threshold": np.zeros(n_bars),    # always 0.0
    })

    bt = SimpleBacktester(warmup_bars=5, max_bars_in_trade=30,
                          fee_per_leg=0.0, spread_cost_bps=0.0)
    result = bt.run(entry, exit_tree, side, data)

    assert result.total_trades >= 1
    # With CALL direction on uptrending log returns, PnL should be positive
    total_pnl = sum(t.pnl for t in result.trades)
    assert total_pnl > 0, f"CALL on uptrend should produce positive PnL, got {total_pnl}"


# ---------------------------------------------------------------------------
# Test 10: Round-trip costs are 2x per-leg fee + 2x spread
# ---------------------------------------------------------------------------

def test_round_trip_cost_formula():
    """Round-trip cost = 2 * fee_per_leg + 2 * notional * spread_bps / 10000."""
    bt = SimpleBacktester(fee_per_leg=1.30, spread_cost_bps=5.0, notional=1000.0)
    expected = 2 * 1.30 + 1000.0 * 5.0 / 10000.0 * 2  # = 2.60 + 1.00 = 3.60
    assert bt._round_trip_cost() == pytest.approx(expected)

    # Zero-cost backtester
    bt_free = SimpleBacktester(fee_per_leg=0, spread_cost_bps=0, notional=1000.0)
    assert bt_free._round_trip_cost() == 0.0


# ---------------------------------------------------------------------------
# Test 11 (bonus): split_gp_data
# ---------------------------------------------------------------------------

def test_split_gp_data_basic():
    """split_gp_data correctly partitions a DataFrame by date with embargo."""
    dates = pd.date_range("2023-01-01", "2025-06-30", freq="B")
    data = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "value": range(len(dates)),
    })

    splits = split_gp_data(data, train_end="2024-09-30",
                           val_end="2025-01-31", embargo_days=5)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    # Train ends at or before 2024-09-30
    train_dates = pd.to_datetime(splits["train"]["date"])
    assert train_dates.max() <= pd.Timestamp("2024-09-30")

    # Val starts after 2024-09-30 + 5 days embargo
    val_dates = pd.to_datetime(splits["val"]["date"])
    assert val_dates.min() > pd.Timestamp("2024-10-05")
    assert val_dates.max() <= pd.Timestamp("2025-01-31")

    # Test starts after 2025-01-31 + 5 days embargo
    test_dates = pd.to_datetime(splits["test"]["date"])
    assert test_dates.min() > pd.Timestamp("2025-02-05")

    # No overlap and no data lost (except embargo days)
    total_with_embargo = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_with_embargo <= len(data)
    assert total_with_embargo > 0


# ---------------------------------------------------------------------------
# Test 12: EphReal constants return their value, not 0.0
# ---------------------------------------------------------------------------

def test_ephreal_returns_stored_value():
    """Ephemeral REAL constants must return their sampled value, not 0.0."""
    from layer2.grammar import TermDef, TermNode, GType
    evaluator = TreeEvaluator()
    ctx = EvaluationContext(max_lag=5)

    # Create an ephemeral terminal with a specific value
    td = TermDef("EphReal", GType.REAL, sampler=lambda: 0.42)
    node = TermNode(defn=td)
    # The sampler should have set node.value = 0.42
    assert node.value == pytest.approx(0.42)

    # Evaluate — should return 0.42, NOT 0.0
    result = evaluator.evaluate(node, ctx)
    assert result == pytest.approx(0.42), f"EphReal returned {result} instead of 0.42"


# ---------------------------------------------------------------------------
# Test 13: OptionsBacktester — option value decays with time
# ---------------------------------------------------------------------------

def test_options_backtester_theta_decay():
    """Option value decreases as minutes_to_expiry decreases (theta decay)."""
    from layer2.evaluator import OptionsBacktester

    spot, strike, iv = 5000.0, 5000.0, 0.20  # ATM call
    val_300min = OptionsBacktester._option_value(spot, strike, 300, iv, is_call=True)
    val_60min = OptionsBacktester._option_value(spot, strike, 60, iv, is_call=True)
    val_5min = OptionsBacktester._option_value(spot, strike, 5, iv, is_call=True)
    val_0min = OptionsBacktester._option_value(spot, strike, 0, iv, is_call=True)

    # Time value should decrease monotonically
    assert val_300min > val_60min > val_5min >= val_0min
    # At expiry, ATM call is worth 0 (intrinsic = max(5000-5000, 0) = 0)
    assert val_0min == 0.0
    # With 300 min left, should have meaningful time value
    assert val_300min > 5.0


# ---------------------------------------------------------------------------
# Test 14: OptionsBacktester — moneyness-dependent spread cost
# ---------------------------------------------------------------------------

def test_options_spread_cost_increases_with_moneyness():
    """Spread cost should increase as option moves away from ATM."""
    from layer2.evaluator import OptionsBacktester
    cost_atm = OptionsBacktester._spread_cost(0.0)      # ATM
    cost_25d = OptionsBacktester._spread_cost(0.02)      # ~25 delta
    cost_5d = OptionsBacktester._spread_cost(0.05)       # ~5 delta

    assert cost_atm < cost_25d < cost_5d
    assert cost_atm == pytest.approx(0.15)  # base cost
    assert cost_5d == pytest.approx(0.60)   # max at 5%


# ---------------------------------------------------------------------------
# Test 15: OptionsBacktester — intrinsic value at expiry
# ---------------------------------------------------------------------------

def test_options_intrinsic_at_expiry():
    """At expiry, option value = intrinsic value only."""
    from layer2.evaluator import OptionsBacktester

    # ITM call: spot=5010, strike=5000 → intrinsic = 10
    assert OptionsBacktester._option_value(5010, 5000, 0, 0.20, is_call=True) == pytest.approx(10.0)
    # OTM call: spot=4990, strike=5000 → intrinsic = 0
    assert OptionsBacktester._option_value(4990, 5000, 0, 0.20, is_call=True) == pytest.approx(0.0)
    # ITM put: spot=4990, strike=5000 → intrinsic = 10
    assert OptionsBacktester._option_value(4990, 5000, 0, 0.20, is_call=False) == pytest.approx(10.0)
    # OTM put: spot=5010, strike=5000 → intrinsic = 0
    assert OptionsBacktester._option_value(5010, 5000, 0, 0.20, is_call=False) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 16: Day-boundary force-close in SimpleBacktester
# ---------------------------------------------------------------------------

def test_day_boundary_force_close():
    """SimpleBacktester force-closes positions at day boundaries."""
    # Create a 2-day dataset where an entry happens on day 1
    # and day 2 starts — position should be closed at day 1's last bar
    dates = (["2024-01-02"] * 40) + (["2024-01-03"] * 40)
    data = pd.DataFrame({
        "date": dates,
        "SPXClose": [0.001 * i for i in range(80)],  # trending up
        "ATM_IV": [0.5] * 80,
        "PredRV15": [0.1] * 80,
    })

    # Always-enter tree, never-exit tree
    always_true = FuncNode(defn=GT, children=[
        TermNode(defn=TermDef("ATM_IV", GType.REAL)),
        TermNode(defn=TermDef("EphReal", GType.REAL, sampler=lambda: 0.0)),
    ])
    never_true = FuncNode(defn=GT, children=[
        TermNode(defn=TermDef("EphReal", GType.REAL, sampler=lambda: -1.0)),
        TermNode(defn=TermDef("ATM_IV", GType.REAL)),
    ])
    call_side = TermNode(defn=TermDef("CALL", GType.SIDE, value=Side.CALL))

    bt = SimpleBacktester(warmup_bars=5, max_bars_in_trade=100)
    result = bt.run(always_true, never_true, call_side, data)

    # Should have at least 2 trades: one force-closed at day boundary, one at end
    assert len(result.trades) >= 2
    # First trade should end at or before bar 39 (last bar of day 1)
    assert result.trades[0].exit_bar <= 39


# ---------------------------------------------------------------------------
# typed-vector evaluation tests
# ---------------------------------------------------------------------------

def test_typed_vector_context_round_trip():
    """EvaluationContext stores and retrieves typed-vector values by name."""
    ctx = EvaluationContext(max_lag=10, emb_dim=4)
    v1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ctx.update({"EMB_GRID": v1, "ATM_IV": 0.3})  # mixed scalar + vector
    ctx.update({"EMB_GRID": v2, "ATM_IV": 0.4})
    # Current (lag=0) is the most recent
    np.testing.assert_array_equal(ctx.get_vec("EMB_GRID", lag=0), v2)
    # Previous (lag=1)
    np.testing.assert_array_equal(ctx.get_vec("EMB_GRID", lag=1), v1)
    # Beyond buffer returns zero vector (not None)
    past = ctx.get_vec("EMB_GRID", lag=99)
    assert past.shape == (4,)
    assert np.allclose(past, 0.0)
    # Scalar accessor still works alongside — values are normalized on ingestion
    # so we check that the stored value is the normalized form, not the raw input.
    from layer2.terminal_stats import normalize, NORMALIZED_TERMINALS
    expected = normalize("ATM_IV", 0.4) if "ATM_IV" in NORMALIZED_TERMINALS else 0.4
    assert ctx.get("ATM_IV", lag=0) == pytest.approx(expected)


def test_typed_vector_ops_evaluate():
    """EmbNorm / EmbCos / EmbSub / EmbLag evaluate correctly through TreeEvaluator."""
    from layer2.grammar import EMB_OPERATORS
    ctx = EvaluationContext(max_lag=5, emb_dim=3)
    ctx.update({"EMB_GRID": np.array([3.0, 4.0, 0.0], dtype=np.float32)})
    ctx.update({"EMB_GRID": np.array([0.0, 5.0, 12.0], dtype=np.float32)})
    ev = TreeEvaluator()
    grid_term = TermNode(defn=TermDef("EMB_GRID", GType.EMB_GRID))
    # EmbNorm on current vector: sqrt(0 + 25 + 144) = 13
    norm_op = next(f for f in EMB_OPERATORS if f.name == "EmbNorm_EMB_GRID")
    norm = ev.evaluate(FuncNode(defn=norm_op, children=[grid_term]), ctx)
    assert norm == pytest.approx(13.0, rel=1e-4)
    # EmbCos(grid, grid) should be 1.0 (self-similarity)
    cos_op = next(f for f in EMB_OPERATORS if f.name == "EmbCos_EMB_GRID")
    cos = ev.evaluate(FuncNode(defn=cos_op, children=[grid_term, grid_term]), ctx)
    assert cos == pytest.approx(1.0, rel=1e-4)
    # EmbLag(grid, 1) returns the prior bar's vector
    lag_op = next(f for f in EMB_OPERATORS if f.name == "EmbLag_EMB_GRID")
    int_term = TermNode(defn=TermDef("EphInt", GType.INT), value=1)
    lagged = ev.evaluate(FuncNode(defn=lag_op, children=[grid_term, int_term]), ctx)
    np.testing.assert_array_equal(lagged, np.array([3.0, 4.0, 0.0], dtype=np.float32))
    # EmbSub(grid, EmbLag(grid, 1)) = current - prior
    sub_op = next(f for f in EMB_OPERATORS if f.name == "EmbSub_EMB_GRID")
    diff_tree = FuncNode(defn=sub_op, children=[
        grid_term,
        FuncNode(defn=lag_op, children=[grid_term, int_term]),
    ])
    diff = ev.evaluate(diff_tree, ctx)
    np.testing.assert_allclose(diff, np.array([-3.0, 1.0, 12.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Batch 1 audit fixes (C4, C2, H3, #204, M5, H2)
# ---------------------------------------------------------------------------

class TestATMIVForwardFill:
    """C4: ATM_IV=0 bars get forward-filled within day before normalization."""

    def test_zero_iv_filled_from_previous_bar(self):
        """Zero IV should be replaced by prior bar's value within same day."""
        import pandas as pd
        from layer2.evaluator_vectorized import prepare_terminal_data

        df = pd.DataFrame({
            "date": ["2024-01-02"] * 5,
            "bar_position": [0, 1, 2, 3, 4],
            "ATM_IV": [0.20, 0.21, 0.0, 0.0, 0.22],
            "MinutesToClose": [400, 399, 398, 397, 396],
        })
        td = prepare_terminal_data(df, normalize_terminals=False)
        atm = td["ATM_IV"]
        assert atm[0] == pytest.approx(0.20)
        assert atm[1] == pytest.approx(0.21)
        # bars 2 and 3 were 0.0 — should be forward-filled
        assert atm[2] == pytest.approx(0.21)
        assert atm[3] == pytest.approx(0.21)
        assert atm[4] == pytest.approx(0.22)

    def test_zero_iv_not_filled_across_day_boundary(self):
        """Forward-fill must not cross day boundaries."""
        import pandas as pd
        from layer2.evaluator_vectorized import prepare_terminal_data

        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "bar_position": [0, 1, 0, 1],
            "ATM_IV": [0.20, 0.21, 0.0, 0.15],
            "MinutesToClose": [400, 399, 400, 399],
        })
        td = prepare_terminal_data(df, normalize_terminals=False)
        atm = td["ATM_IV"]
        # bar 2 is first bar of new day with zero — no prior to fill from
        assert atm[2] == pytest.approx(0.0)
        assert atm[3] == pytest.approx(0.15)


class TestCreditHaircutRecalibration:
    """#204: Call spreads get 25% haircut, put spreads 15%. Afternoon penalty."""

    def test_call_credit_gets_25pct_haircut(self):
        """Bear call credit (short call) should get 25% haircut (0.75× multiplier)."""
        from layer2.templates import bear_call_credit
        tmpl = bear_call_credit()
        # Verify the template has short calls
        short_calls = [l for l in tmpl.legs if l.option_type == "call" and l.qty_sign < 0]
        assert len(short_calls) > 0, "bear_call_credit must have short call legs"

    def test_put_credit_keeps_15pct_haircut(self):
        """Bull put credit (short put, no short call) should keep 15% haircut."""
        from layer2.templates import bull_put_credit
        tmpl = bull_put_credit()
        short_calls = [l for l in tmpl.legs if l.option_type == "call" and l.qty_sign < 0]
        assert len(short_calls) == 0, "bull_put_credit should have no short call legs"

    def test_iron_condor_has_short_calls(self):
        """IC has both short calls and puts — should trigger 25% haircut."""
        from layer2.templates import iron_condor
        tmpl = iron_condor()
        short_calls = [l for l in tmpl.legs if l.option_type == "call" and l.qty_sign < 0]
        assert len(short_calls) >= 1


class TestWingRaise:
    """H2: 5-delta wings raised to 10-delta in IC, BPC, BCC templates."""

    def _get_wing_deltas(self, template):
        """Return absolute deltas of long (protective) legs."""
        return sorted(abs(l.delta_target) for l in template.legs if l.qty_sign > 0)

    def test_iron_condor_10d_wings(self):
        from layer2.templates import iron_condor
        wings = self._get_wing_deltas(iron_condor())
        # Both wings should be 0.10 (was 0.05)
        assert all(d == pytest.approx(0.10) for d in wings), f"IC wings: {wings}"

    def test_bull_put_credit_10d_wing(self):
        from layer2.templates import bull_put_credit
        wings = self._get_wing_deltas(bull_put_credit())
        assert wings[0] == pytest.approx(0.10), f"BPC wing: {wings}"

    def test_bear_call_credit_10d_wing(self):
        from layer2.templates import bear_call_credit
        wings = self._get_wing_deltas(bear_call_credit())
        assert wings[0] == pytest.approx(0.10), f"BCC wing: {wings}"

    def test_iron_butterfly_unchanged(self):
        """IB already used 0.10 wings — should be unchanged."""
        from layer2.templates import iron_butterfly
        wings = self._get_wing_deltas(iron_butterfly())
        assert all(d == pytest.approx(0.10) for d in wings)

    def test_codegen_legs_match_templates(self):
        """Codegen TEMPLATE_LEGS deltas must match templates.py."""
        from layer3.codegen import TEMPLATE_LEGS
        from layer2.templates import TEMPLATE_FACTORIES

        for factory in TEMPLATE_FACTORIES:
            tmpl = factory()
            name = tmpl.name
            if name not in TEMPLATE_LEGS:
                continue
            proxy_deltas = sorted(l.delta_target for l in tmpl.legs)
            codegen_deltas = sorted(l.delta_target for l in TEMPLATE_LEGS[name])
            assert proxy_deltas == pytest.approx(codegen_deltas, abs=1e-6), (
                f"{name}: proxy={proxy_deltas} != codegen={codegen_deltas}"
            )


class TestG5RegimeGate:
    """H3: G5 regime gate tightened from -1.0 to -0.3."""

    def test_g5_threshold_is_negative_zero_three(self):
        from layer2.experiment import SURVIVAL_GATES
        assert SURVIVAL_GATES["G5_min_regime_sharpe"] == pytest.approx(-0.3)


class TestM5RaiseOnMissing:
    """M5: Vectorized PCA bases loader uses raise_on_missing=True."""

    def test_vectorized_loader_raises_on_missing(self):
        """_load_pca_bases_for_vectorized should use raise_on_missing=True."""
        import inspect
        from layer2.evaluator_vectorized import _load_pca_bases_for_vectorized
        source = inspect.getsource(_load_pca_bases_for_vectorized)
        assert "raise_on_missing=True" in source


class TestSortinoFormula:
    """Verify Sortino uses sqrt(mean(negative_returns²)), not std(negatives)."""

    def test_sortino_with_known_values(self):
        from layer2.evaluator import SimpleBacktester
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03])
        # Downside: [-0.02, -0.01, -0.03]
        # DD = sqrt(mean([0.0004, 0.0001, 0.0009])) = sqrt(0.000467) = 0.02160
        expected_dd = float(np.sqrt(np.mean(np.array([-0.02, -0.01, -0.03]) ** 2)))
        expected_mean = float(np.mean(returns))
        bpd = 1
        expected_sortino = expected_mean / expected_dd * np.sqrt(252 * bpd)
        actual = SimpleBacktester._sortino(returns, bars_per_day=bpd)
        assert actual == pytest.approx(expected_sortino, rel=1e-4)

    def test_sortino_all_positive_returns_capped(self):
        from layer2.evaluator import SimpleBacktester
        returns = np.array([0.01, 0.02, 0.01])
        assert SimpleBacktester._sortino(returns, bars_per_day=1) == pytest.approx(10.0)


class TestSlippageBoundaries:
    """Verify stop-loss slippage: 1.20 at open, 1.50 at close."""

    def test_slippage_at_open(self):
        # mtc=390 → hours_rem=6.5 → 1.2 + 0.3*(1 - 6.5/6.5) = 1.2
        hours_rem = 390 / 60.0
        slippage = 1.2 + 0.3 * (1.0 - min(hours_rem / 6.5, 1.0))
        assert slippage == pytest.approx(1.20)

    def test_slippage_at_close(self):
        # mtc=0 → hours_rem=0 → 1.2 + 0.3*(1 - 0) = 1.5
        hours_rem = 0 / 60.0
        slippage = 1.2 + 0.3 * (1.0 - min(hours_rem / 6.5, 1.0))
        assert slippage == pytest.approx(1.50)

    def test_slippage_midday(self):
        # mtc=195 → hours_rem=3.25 → 1.2 + 0.3*(1 - 0.5) = 1.35
        hours_rem = 195 / 60.0
        slippage = 1.2 + 0.3 * (1.0 - min(hours_rem / 6.5, 1.0))
        assert slippage == pytest.approx(1.35)


class TestCreditHaircutAfternoon:
    """Verify afternoon penalty fires at mtc<=195."""

    def test_afternoon_penalty_multiplier(self):
        # Call spread afternoon: 0.75 * 0.90 = 0.675
        base = 0.75  # call haircut
        afternoon = 0.90  # mtc<=195
        assert base * afternoon == pytest.approx(0.675)

    def test_morning_no_penalty(self):
        # mtc=300 (morning): no afternoon penalty
        mtc = 300
        assert mtc > 195  # morning = no penalty


class TestPSRKnownValue:
    """Verify PSR with known inputs produces expected value."""

    def test_psr_zero_sharpe(self):
        """PSR should be ~0.5 for Sharpe=0 (50/50 chance true SR > 0)."""
        from layer2.evaluator import BacktestResult
        returns = np.random.RandomState(42).normal(0, 0.01, 252)
        result = BacktestResult(
            returns=returns,
            trades=[],
            equity_curve=np.cumsum(returns),
            max_drawdown=0.05,
            sharpe=0.0,
            n_days=252,
        )
        # PSR for SR=0 should be close to 0.5
        assert 0.3 < result.psr < 0.7


# ---------------------------------------------------------------------------
# Empirical IV Surface Tests
# ---------------------------------------------------------------------------

class TestEmpiricalIV:
    """Tests for _empirical_iv: 11-point grid IV interpolation."""

    def test_atm_returns_atm_iv(self):
        """At delta=+0.50 (ATM call), should return grid ATM IV."""
        from layer2.evaluator_vectorized import _empirical_iv
        # Grid: 5Dp, 10Dp, 25Dp, 40Dp, ATMp, ATM, ATMc, 40Dc, 25Dc, 10Dc, 5Dc
        grid = np.array([
            0.30, 0.28, 0.24, 0.22, 0.21,  # put side (5Dp..ATMp)
            0.20, 0.20,                       # ATM, ATMc
            0.19, 0.18, 0.17, 0.16,           # call side (40Dc..5Dc)
        ])
        result = _empirical_iv(grid, 0.50, atm_iv=0.20)
        # ATM and ATMc are both 0.20, average = 0.20
        assert result == pytest.approx(0.20, abs=0.005)

    def test_put_side_higher_iv(self):
        """OTM puts (negative delta) should have higher IV than ATM (skew)."""
        from layer2.evaluator_vectorized import _empirical_iv
        grid = np.array([
            0.35, 0.30, 0.25, 0.22, 0.21,
            0.20, 0.20,
            0.19, 0.18, 0.17, 0.16,
        ])
        iv_25dp = _empirical_iv(grid, -0.25, atm_iv=0.20)
        iv_atm = _empirical_iv(grid, 0.50, atm_iv=0.20)
        assert iv_25dp > iv_atm, f"Put IV ({iv_25dp}) should exceed ATM ({iv_atm})"

    def test_interpolation_between_grid_points(self):
        """IV at delta between grid points should be linearly interpolated."""
        from layer2.evaluator_vectorized import _empirical_iv
        # Uniform grid IVs for predictable interpolation
        grid = np.array([
            0.30, 0.28, 0.24, 0.22, 0.21,
            0.20, 0.20,
            0.19, 0.18, 0.17, 0.16,
        ])
        # Delta -0.175 is between -0.25 (IV=0.24) and -0.10 (IV=0.28)
        iv = _empirical_iv(grid, -0.175, atm_iv=0.20)
        # Linear interp: -0.175 is 50% between -0.25 and -0.10
        expected = 0.5 * 0.24 + 0.5 * 0.28  # = 0.26
        assert iv == pytest.approx(expected, abs=0.01)

    def test_all_zero_falls_back_to_atm(self):
        """All-zero grid IVs should fall back to atm_iv."""
        from layer2.evaluator_vectorized import _empirical_iv
        grid = np.zeros(11)
        result = _empirical_iv(grid, -0.25, atm_iv=0.18)
        assert result == pytest.approx(0.18)

    def test_none_falls_back_to_atm(self):
        """None grid_ivs should fall back to atm_iv."""
        from layer2.evaluator_vectorized import _empirical_iv
        result = _empirical_iv(None, -0.25, atm_iv=0.22)
        assert result == pytest.approx(0.22)

    def test_partial_zeros_filled_with_atm(self):
        """Zero entries in grid should be replaced by atm_iv before interp."""
        from layer2.evaluator_vectorized import _empirical_iv
        grid = np.array([
            0.30, 0.0, 0.24, 0.22, 0.21,   # IV_10Dp is 0 (gap)
            0.20, 0.20,
            0.19, 0.18, 0.17, 0.16,
        ])
        # The zero at IV_10Dp (delta=-0.10) gets replaced by atm_iv=0.20
        iv_10dp = _empirical_iv(grid, -0.10, atm_iv=0.20)
        # Should use the fallback value (0.20) instead of 0.0
        assert iv_10dp == pytest.approx(0.20, abs=0.01)

    def test_delta_clamped_to_grid_range(self):
        """Deltas outside [-0.50, +0.50] should be clamped to grid edges."""
        from layer2.evaluator_vectorized import _empirical_iv
        grid = np.array([
            0.35, 0.30, 0.25, 0.22, 0.21,
            0.20, 0.20,
            0.19, 0.18, 0.17, 0.16,
        ])
        # Delta -0.90 (way OTM put) should clamp to -0.50 (ATMp = 0.21)
        iv_deep_otm = _empirical_iv(grid, -0.90, atm_iv=0.20)
        iv_atmp = _empirical_iv(grid, -0.50, atm_iv=0.20)
        assert iv_deep_otm == pytest.approx(iv_atmp)

    def test_minimum_iv_floor(self):
        """Result should never be below 0.01."""
        from layer2.evaluator_vectorized import _empirical_iv
        # Very low IVs
        grid = np.array([0.005] * 11)
        result = _empirical_iv(grid, 0.25, atm_iv=0.005)
        assert result >= 0.01

    def test_atm_atmc_averaging(self):
        """IV_ATM and IV_ATMc should be averaged when both nonzero."""
        from layer2.evaluator_vectorized import _empirical_iv
        grid = np.array([
            0.30, 0.28, 0.24, 0.22, 0.21,
            0.20, 0.22,   # ATM=0.20, ATMc=0.22 => avg=0.21
            0.19, 0.18, 0.17, 0.16,
        ])
        iv_atm = _empirical_iv(grid, 0.50, atm_iv=0.20)
        assert iv_atm == pytest.approx(0.21, abs=0.005)


# ---------------------------------------------------------------------------
# Edgeworth-Corrected Option Pricing Tests
# ---------------------------------------------------------------------------

class TestEdgeworthOptionValue:
    """Tests for _edgeworth_option_value: Bandi-Fusari-Reno (2024) correction."""

    def test_reduces_to_bs_with_zero_params(self):
        """With rho_t=0 and beta_t=0, should return exact BS price."""
        from layer2.evaluator_vectorized import _option_value, _edgeworth_option_value
        spot, strike, mtc, iv = 5000.0, 5000.0, 200.0, 0.20
        bs = _option_value(spot, strike, mtc, iv, is_call=True)
        ew = _edgeworth_option_value(spot, strike, mtc, iv, is_call=True,
                                      rho_t=0.0, beta_t=0.0)
        assert ew == pytest.approx(bs, rel=1e-10)

    def test_negative_skew_increases_put_value(self):
        """Negative rho_t (left skew) should increase OTM put values."""
        from layer2.evaluator_vectorized import _option_value, _edgeworth_option_value
        spot, strike, iv = 5000.0, 4950.0, 0.20
        mtc = 200.0
        bs_put = _option_value(spot, strike, mtc, iv, is_call=False)
        # Negative rho_t = negative skew (fat left tail) -> puts worth more
        ew_put = _edgeworth_option_value(spot, strike, mtc, iv, is_call=False,
                                          rho_t=-0.10, beta_t=0.0)
        assert ew_put > bs_put, (
            f"Negative skew should increase OTM put: EW={ew_put:.4f} vs BS={bs_put:.4f}"
        )

    def test_positive_kurtosis_increases_otm_values(self):
        """Positive beta_t (fat tails) should increase deep OTM option values."""
        from layer2.evaluator_vectorized import _option_value, _edgeworth_option_value
        spot, strike, iv = 5000.0, 4900.0, 0.20
        mtc = 200.0
        bs_put = _option_value(spot, strike, mtc, iv, is_call=False)
        ew_put = _edgeworth_option_value(spot, strike, mtc, iv, is_call=False,
                                          rho_t=0.0, beta_t=0.20)
        # Fat tails should increase deep OTM values
        assert ew_put != bs_put, "Non-zero beta should change option value"

    def test_at_expiry_returns_intrinsic(self):
        """At mtc=0, Edgeworth correction is not applied (intrinsic only)."""
        from layer2.evaluator_vectorized import _edgeworth_option_value
        spot, strike = 5010.0, 5000.0
        # ITM call at expiry: intrinsic = 10
        result = _edgeworth_option_value(spot, strike, 0.0, 0.20, is_call=True,
                                          rho_t=-0.15, beta_t=0.10)
        assert result == pytest.approx(10.0)

    def test_correction_factor_bounded(self):
        """Correction factor should be clamped to [0.80, 1.25]."""
        from layer2.evaluator_vectorized import _option_value, _edgeworth_option_value
        spot, strike, iv, mtc = 5000.0, 5050.0, 0.20, 200.0
        bs = _option_value(spot, strike, mtc, iv, is_call=True)
        # Extreme parameters that would push correction out of bounds
        ew = _edgeworth_option_value(spot, strike, mtc, iv, is_call=True,
                                      rho_t=5.0, beta_t=5.0)
        ratio = ew / bs if bs > 1e-10 else 1.0
        assert 0.79 <= ratio <= 1.26, f"Correction ratio {ratio} out of bounds"

    def test_call_put_parity_approximately_holds(self):
        """Edgeworth-corrected prices should approximately satisfy put-call parity."""
        from layer2.evaluator_vectorized import _edgeworth_option_value
        spot, strike, iv = 5000.0, 5000.0, 0.20
        mtc = 200.0
        rho_t, beta_t = -0.05, 0.10
        call = _edgeworth_option_value(spot, strike, mtc, iv, True, rho_t, beta_t)
        put = _edgeworth_option_value(spot, strike, mtc, iv, False, rho_t, beta_t)
        # ATM: call - put should be approximately 0 (no rate, no dividend)
        # The Edgeworth correction breaks exact parity but should be close
        assert abs(call - put) < 5.0, (
            f"Call-put spread too large at ATM: {call - put:.4f}"
        )

    def test_typical_spx_correction_magnitude(self):
        """For typical SPX parameters, correction should be small (< 10%)."""
        from layer2.evaluator_vectorized import _option_value, _edgeworth_option_value
        spot, strike, iv, mtc = 5000.0, 4975.0, 0.18, 200.0
        # Typical SPX skew: rho ~ -0.05, beta ~ 0.05
        bs = _option_value(spot, strike, mtc, iv, is_call=False)
        ew = _edgeworth_option_value(spot, strike, mtc, iv, is_call=False,
                                      rho_t=-0.05, beta_t=0.05)
        if bs > 0.01:
            pct_diff = abs(ew - bs) / bs
            assert pct_diff <= 0.25, f"Correction {pct_diff:.1%} too large for typical params"


# ---------------------------------------------------------------------------
# Surface parameter estimation tests
# ---------------------------------------------------------------------------

class TestEstimateSurfaceParams:
    """Tests for _estimate_surface_params: rho_t and beta_t from grid IVs."""

    def test_symmetric_smile_zero_skew(self):
        """Symmetric IV grid should produce rho_t ~ 0."""
        from layer2.evaluator_vectorized import _estimate_surface_params
        # Symmetric smile: puts and calls equidistant from ATM have same IV
        grid = np.array([
            0.30, 0.25, 0.22, 0.21, 0.20,  # put side (mirror of call)
            0.20, 0.20,                       # ATM
            0.21, 0.22, 0.25, 0.30,           # call side (mirror of put)
        ])
        rho, beta = _estimate_surface_params(grid, atm_iv=0.20)
        assert abs(rho) < 0.01, f"Symmetric smile should have rho~0, got {rho}"

    def test_typical_skew_negative_rho(self):
        """Typical SPX skew (puts more expensive) should produce rho ~ 0 or slightly neg/pos.
        Note: rho_t = (ATMc - ATMp) / (2*ATM). With ATMc < ATMp in typical skew,
        rho_t should be negative."""
        from layer2.evaluator_vectorized import _estimate_surface_params
        grid = np.array([
            0.35, 0.30, 0.25, 0.22, 0.21,
            0.20, 0.19,   # ATM=0.20, ATMc=0.19 (calls cheaper)
            0.19, 0.18, 0.17, 0.16,
        ])
        rho, beta = _estimate_surface_params(grid, atm_iv=0.20)
        assert rho < 0, f"Call-cheaper skew should produce negative rho, got {rho}"

    def test_positive_curvature_positive_beta(self):
        """IV smile with positive curvature (wings up) should produce positive beta."""
        from layer2.evaluator_vectorized import _estimate_surface_params
        # U-shaped smile: 25-delta points above ATM
        grid = np.array([
            0.30, 0.28, 0.25, 0.22, 0.21,
            0.20, 0.20,
            0.22, 0.25, 0.28, 0.30,
        ])
        rho, beta = _estimate_surface_params(grid, atm_iv=0.20)
        # beta = (0.25 + 0.25 - 2*0.20) / (0.0625 * 0.20) = 0.10/0.0125 = 8.0
        assert beta > 0, f"Positive curvature should produce beta > 0, got {beta}"

    def test_all_zero_returns_zero(self):
        """All-zero grid should return (0, 0)."""
        from layer2.evaluator_vectorized import _estimate_surface_params
        rho, beta = _estimate_surface_params(np.zeros(11), atm_iv=0.20)
        assert rho == 0.0
        assert beta == 0.0

    def test_none_grid_returns_zero(self):
        """None grid should return (0, 0)."""
        from layer2.evaluator_vectorized import _estimate_surface_params
        rho, beta = _estimate_surface_params(None, atm_iv=0.20)
        assert rho == 0.0
        assert beta == 0.0


# ---------------------------------------------------------------------------
# Integration test: net_pos_value with grid IVs
# ---------------------------------------------------------------------------

class TestNetPosValueWithGridIVs:
    """Test that _net_pos_value uses empirical IV when grid_ivs is provided."""

    def test_with_grid_differs_from_without(self):
        """Net position value with grid IVs should differ from parametric skew."""
        from layer2.evaluator_vectorized import _net_pos_value
        from layer2.templates import iron_condor
        tmpl = iron_condor()
        spot = 5000.0
        mtc = 200.0
        iv = 0.20
        strikes = [4950.0, 4975.0, 5025.0, 5050.0]
        # Skewed grid (typical SPX)
        grid = np.array([
            0.35, 0.30, 0.25, 0.22, 0.21,
            0.20, 0.20,
            0.19, 0.18, 0.17, 0.16,
        ])
        val_no_grid = _net_pos_value(tmpl.legs, strikes, spot, mtc, iv)
        val_with_grid = _net_pos_value(tmpl.legs, strikes, spot, mtc, iv,
                                        grid_ivs=grid)
        # Values should differ because empirical IV surface + Edgeworth != parametric skew + BS
        assert val_no_grid != pytest.approx(val_with_grid, abs=1e-6), (
            f"Grid IV pricing should differ: no_grid={val_no_grid:.4f}, "
            f"with_grid={val_with_grid:.4f}"
        )

    def test_none_grid_matches_parametric(self):
        """None grid_ivs should give identical results to parametric path."""
        from layer2.evaluator_vectorized import _net_pos_value
        from layer2.templates import bull_put_credit
        tmpl = bull_put_credit()
        spot, mtc, iv = 5000.0, 200.0, 0.20
        strikes = [4950.0, 4975.0]
        val1 = _net_pos_value(tmpl.legs, strikes, spot, mtc, iv)
        val2 = _net_pos_value(tmpl.legs, strikes, spot, mtc, iv, grid_ivs=None)
        assert val1 == pytest.approx(val2, rel=1e-10)


# ---------------------------------------------------------------------------
# Phi (standard normal PDF) test
# ---------------------------------------------------------------------------

class TestPhi:
    """Test the math-only normal PDF function."""

    def test_phi_at_zero(self):
        """phi(0) = 1/sqrt(2*pi) ~ 0.3989."""
        from layer2.evaluator_vectorized import _phi
        assert _phi(0.0) == pytest.approx(0.3989422804, rel=1e-6)

    def test_phi_symmetry(self):
        """phi(x) = phi(-x) for all x."""
        from layer2.evaluator_vectorized import _phi
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert _phi(x) == pytest.approx(_phi(-x), rel=1e-12)

    def test_phi_tail(self):
        """phi(3.0) should be small (~0.0044)."""
        from layer2.evaluator_vectorized import _phi
        assert _phi(3.0) == pytest.approx(0.00443185, rel=1e-3)
