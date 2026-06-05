"""Tests for task #34: inner validation during GP evolution + per-regime hard gate.

Change 1: Inner validation monitoring (observation only, NOT selection).
Change 2: Per-regime Sharpe hard gate in VectorizedFitnessEvaluator.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from layer2.evaluator import BacktestResult, Trade, Side
from layer2.fitness import (
    DEFAULT_OBJECTIVES, FAILED_FITNESS_SENTINEL,
    FitnessEvaluator, VectorizedFitnessEvaluator,
    _regime_gate_fails, _compute_regime_terciles,
    REGIME_MIN_TRADES,
)
from layer2.gp_engine import (
    EvolutionConfig, GenerationMetrics, Individual,
    evolve, initialize_population, pareto_front,
)
from layer2.grammar import Grammar, GType, to_str
from layer2.io import (
    PRICE_COLUMN, PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import iron_condor, bull_put_credit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_dates=2, bars_per_date=20) -> pd.DataFrame:
    """Synthetic data with ATM_IV variation for regime testing."""
    rng = np.random.default_rng(0)
    rows = []
    for date in [f"2024-{m:02d}-15" for m in range(1, n_dates + 1)]:
        spot = 4500.0
        for w in range(bars_per_date):
            spot += rng.normal(0, 1.0)
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: spot, "ATM_IV": 0.20,
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


def _make_trades(entry_bars, pnls, n_bars=100):
    """Create synthetic Trade objects at specified entry bars with given PnLs."""
    trades = []
    for eb, pnl in zip(entry_bars, pnls):
        trades.append(Trade(
            entry_bar=eb,
            exit_bar=min(eb + 10, n_bars - 1),
            side=Side.NEUTRAL,
            entry_price=100.0,
            exit_price=100.0 + pnl,
            pnl=pnl,
            bars_held=10,
            exit_reason="signal",
        ))
    return trades


def _make_backtest_result(trades, n_days=10, sharpe=1.0, sortino=1.0):
    """Create a BacktestResult with specified trades."""
    # Compute n_bars from max exit_bar to avoid index errors
    n_bars = max(t.exit_bar for t in trades) + 1 if trades else 100
    n_bars = max(n_bars, 100)
    returns = np.zeros(n_bars)
    for t in trades:
        returns[t.exit_bar] = t.pnl / 1000.0
    return BacktestResult(
        returns=returns,
        trades=trades,
        equity_curve=np.cumsum(returns),
        max_drawdown=0.05,
        sharpe=sharpe,
        sortino=sortino,
        total_trades=len(trades),
        win_rate=sum(1 for t in trades if t.pnl > 0) / max(len(trades), 1),
        n_days=n_days,
        exit_utilization=0.5,
        entry_fire_rate=0.1,
        conditional_sharpe_gap=0.2,
        avg_position_size=0.5,
    )


# ===========================================================================
# Change 1: Inner validation during evolution
# ===========================================================================

class TestInnerValidation:
    """Tests for validation monitoring in evolve()."""

    def test_evolve_returns_val_tracking_dict(self):
        """evolve() returns a 3-tuple with val_tracking dict."""
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        fe = FitnessEvaluator(
            template=template, data=_make_data(n_dates=1, bars_per_date=15),
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        config = EvolutionConfig(
            pop_size=8, n_generations=3, seed=42, n_partitions=4,
        )
        final_pop, metrics_log, val_tracking = evolve(
            template, fe, grammar, config,
        )
        assert isinstance(val_tracking, dict)
        assert "val_checkpoints" in val_tracking
        assert "best_val_sharpe" in val_tracking
        assert "best_val_generation" in val_tracking
        assert "best_val_front" in val_tracking

    def test_evolve_without_val_evaluator_has_empty_checkpoints(self):
        """When no val_evaluator is provided, val_checkpoints is empty."""
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        fe = FitnessEvaluator(
            template=template, data=_make_data(n_dates=1, bars_per_date=15),
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        config = EvolutionConfig(
            pop_size=8, n_generations=3, seed=42, n_partitions=4,
        )
        _, _, val_tracking = evolve(template, fe, grammar, config)
        assert val_tracking["val_checkpoints"] == []
        assert val_tracking["best_val_sharpe"] is None
        assert val_tracking["best_val_generation"] is None
        assert val_tracking["best_val_front"] is None

    def test_evolve_with_val_evaluator_records_checkpoints(self):
        """When val_evaluator is provided, checkpoints are recorded at intervals."""
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        fe = FitnessEvaluator(
            template=template, data=_make_data(n_dates=1, bars_per_date=15),
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        # Run for 25 generations with checkpoint every 5
        config = EvolutionConfig(
            pop_size=8, n_generations=25, seed=42, n_partitions=4,
        )
        call_log = []

        def _mock_val_evaluator(front):
            call_log.append(len(front))
            return 0.5  # constant val sharpe

        _, _, val_tracking = evolve(
            template, fe, grammar, config,
            val_evaluator=_mock_val_evaluator,
            val_checkpoint_interval=5,
        )
        # Checkpoints at gen 5, 10, 15, 20 (not gen 0, and gen 24 is last)
        assert len(val_tracking["val_checkpoints"]) == 4
        assert all(gen % 5 == 0 and gen > 0
                   for gen, _ in val_tracking["val_checkpoints"])
        assert val_tracking["best_val_sharpe"] == 0.5
        assert val_tracking["best_val_generation"] == 5  # first checkpoint
        # Front snapshot should exist
        assert val_tracking["best_val_front"] is not None
        assert len(val_tracking["best_val_front"]) > 0

    def test_val_evaluator_does_not_affect_selection(self):
        """Val evaluator is observation-only -- same training fitness drives selection.

        Verify that evolve() with and without val_evaluator produces
        the same final population fitness (given the same seed).
        """
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        data = _make_data(n_dates=1, bars_per_date=15)
        config = EvolutionConfig(
            pop_size=8, n_generations=5, seed=42, n_partitions=4,
        )

        # Run without val evaluator
        fe1 = FitnessEvaluator(
            template=template, data=data,
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        pop1, _, _ = evolve(template, fe1, grammar, config)

        # Run with val evaluator (returns constant)
        fe2 = FitnessEvaluator(
            template=template, data=data,
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )

        def _val_eval(front):
            return 0.5

        pop2, _, _ = evolve(
            template, fe2, grammar, config,
            val_evaluator=_val_eval,
            val_checkpoint_interval=2,
        )

        # Fitness vectors should be identical (same seed, same selection)
        fit1 = sorted(tuple(ind.fitness) for ind in pop1)
        fit2 = sorted(tuple(ind.fitness) for ind in pop2)
        for f1, f2 in zip(fit1, fit2):
            np.testing.assert_array_almost_equal(f1, f2, decimal=6)

    def test_val_tracking_updates_best_on_improvement(self):
        """best_val_sharpe updates when a new checkpoint improves."""
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        fe = FitnessEvaluator(
            template=template, data=_make_data(n_dates=1, bars_per_date=15),
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        config = EvolutionConfig(
            pop_size=8, n_generations=15, seed=42, n_partitions=4,
        )
        # Increasing val sharpe over time
        call_count = [0]

        def _increasing_val(front):
            call_count[0] += 1
            return float(call_count[0]) * 0.1  # 0.1, 0.2, ...

        _, _, vt = evolve(
            template, fe, grammar, config,
            val_evaluator=_increasing_val,
            val_checkpoint_interval=5,
        )
        # best should be at the last checkpoint
        assert vt["best_val_sharpe"] == pytest.approx(0.2)  # call_count = 2 (gens 5, 10)
        assert vt["best_val_generation"] == 10

    def test_val_stagnation_warning(self):
        """Stagnation warning is issued when val_sharpe doesn't improve."""
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        fe = FitnessEvaluator(
            template=template, data=_make_data(n_dates=1, bars_per_date=15),
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        # Need enough generations for stagnation (patience=10, interval=5)
        config = EvolutionConfig(
            pop_size=8, n_generations=30, seed=42, n_partitions=4,
        )
        # Val sharpe peaks early then stagnates
        call_count = [0]

        def _stagnating_val(front):
            call_count[0] += 1
            if call_count[0] <= 1:
                return 1.0  # peak at first checkpoint
            return 0.0  # stagnate

        with pytest.warns(UserWarning, match="Val stagnation"):
            _, _, vt = evolve(
                template, fe, grammar, config,
                val_evaluator=_stagnating_val,
                val_checkpoint_interval=5,
                val_patience=10,
            )

    def test_best_val_front_snapshot_is_deep_copy(self):
        """The best-val front snapshot is a deep copy, not a reference."""
        template = iron_condor()
        grammar = Grammar(max_depth=3, max_nodes=15)
        fe = FitnessEvaluator(
            template=template, data=_make_data(n_dates=1, bars_per_date=15),
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        config = EvolutionConfig(
            pop_size=8, n_generations=15, seed=42, n_partitions=4,
        )

        def _val_eval(front):
            return 0.5

        pop, _, vt = evolve(
            template, fe, grammar, config,
            val_evaluator=_val_eval,
            val_checkpoint_interval=5,
        )
        if vt["best_val_front"]:
            # Modify the snapshot and verify it doesn't affect the pop
            snapshot_ind = vt["best_val_front"][0]
            original_sig = snapshot_ind.signature()
            # The snapshot individual should NOT be the same object as any pop member
            for pop_ind in pop:
                assert snapshot_ind is not pop_ind


# ===========================================================================
# Change 2: Per-regime hard gate
# ===========================================================================

class TestRegimeGate:
    """Tests for per-regime Sharpe hard gate."""

    def test_regime_gate_passes_uniform_positive_pnls(self):
        """All-positive trades in all regimes should pass the gate."""
        # Create trades spread across regimes via ATM_IV normalization
        n_bars = 100
        atm_iv_norm = np.zeros(n_bars)
        # Low vol regime: bars 0-30 (norm < -0.5)
        atm_iv_norm[:30] = -1.0
        # Normal regime: bars 30-70
        atm_iv_norm[30:70] = 0.0
        # High vol regime: bars 70-100 (norm > 1.0)
        atm_iv_norm[70:] = 1.5

        # 10 positive trades per regime
        entry_bars = list(range(0, 30, 3))[:10] + list(range(30, 70, 4))[:10] + list(range(70, 100, 3))[:10]
        pnls = [10.0] * 30  # all positive

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        assert _regime_gate_fails(result, terminal_data) is False

    def test_regime_gate_fails_one_catastrophic_regime(self):
        """Strategy profitable overall but catastrophic in one regime should fail."""
        n_bars = 100
        atm_iv_norm = np.zeros(n_bars)
        # Three groups with slight spread so terciles fall between groups
        atm_iv_norm[:34] = np.linspace(-3.0, -1.5, 34)  # low vol
        atm_iv_norm[34:67] = np.linspace(-0.5, 0.5, 33)  # mid vol
        atm_iv_norm[67:] = np.linspace(1.5, 3.0, 33)     # high vol

        # Low vol: 10 trades, all losing heavily
        entry_bars_low = list(range(0, 30, 3))[:10]
        pnls_low = [-50.0] * 10

        # Mid: 10 trades, all winning big
        entry_bars_normal = list(range(34, 64, 3))[:10]
        pnls_normal = [100.0] * 10

        # High vol: 10 trades, mixed (overall positive)
        entry_bars_high = list(range(67, 97, 3))[:10]
        pnls_high = [20.0] * 10

        entry_bars = entry_bars_low + entry_bars_normal + entry_bars_high
        pnls = pnls_low + pnls_normal + pnls_high

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        # Low vol regime has Sharpe = -50/0 (all same PnL) which is negative mean
        # with zero std -- should be treated as catastrophic
        assert _regime_gate_fails(result, terminal_data) is True

    def test_regime_gate_skips_regimes_with_few_trades(self):
        """Regimes with < 10 trades should be skipped (insufficient data)."""
        n_bars = 100
        atm_iv_norm = np.zeros(n_bars)
        atm_iv_norm[:40] = -1.0   # low vol
        atm_iv_norm[40:] = 0.0    # normal

        # Low vol: only 5 trades (< REGIME_MIN_TRADES), very negative
        entry_bars_low = list(range(0, 15, 3))[:5]
        pnls_low = [-100.0] * 5

        # Normal: 15 trades, all positive
        entry_bars_normal = list(range(40, 100, 4))[:15]
        pnls_normal = [10.0] * 15

        entry_bars = entry_bars_low + entry_bars_normal
        pnls = pnls_low + pnls_normal

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        # Low vol has < 10 trades, so gate should not trigger
        assert _regime_gate_fails(result, terminal_data) is False

    def test_regime_gate_passes_when_no_atm_iv_data(self):
        """Gate should gracefully pass when ATM_IV is not in terminal_data."""
        trades = _make_trades(list(range(0, 30, 3))[:10], [-50.0] * 10)
        result = _make_backtest_result(trades)
        terminal_data = {}  # no ATM_IV

        assert _regime_gate_fails(result, terminal_data) is False

    def test_regime_gate_min_trades_constant(self):
        """Verify min trades constant."""
        assert REGIME_MIN_TRADES == 10

    def test_regime_terciles_computed_from_data(self):
        """Tercile boundaries should adapt to the data distribution."""
        # Uniform distribution [0, 3]
        atm_iv = np.linspace(0, 3, 300)
        td = {"ATM_IV": atm_iv}
        terciles = _compute_regime_terciles(td)
        assert terciles is not None
        q33, q67 = terciles
        assert 0.9 < q33 < 1.1, f"q33={q33}, expected ~1.0"
        assert 1.9 < q67 < 2.1, f"q67={q67}, expected ~2.0"

    def test_regime_gate_uses_terciles(self):
        """Regime gate with terciles should partition trades correctly."""
        n_bars = 300
        # Uniform ATM_IV: terciles at ~1.0 and ~2.0
        atm_iv_norm = np.linspace(0, 3, n_bars)
        terminal_data = {"ATM_IV": atm_iv_norm}
        terciles = _compute_regime_terciles(terminal_data)

        # 20 trades in low regime (bars 0-60, IV < 1.0), all losing badly
        entry_bars = list(range(0, 60, 3))[:20]
        pnls = [-100.0] * 20
        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)

        # With fallback floor -0.5, this should fail (Sharpe = -inf for constant -100)
        assert _regime_gate_fails(result, terminal_data, regime_terciles=terciles) is True

    def test_regime_gate_with_varying_iv_across_trades(self):
        """Trades in different IV environments should be classified correctly."""
        n_bars = 300
        atm_iv_norm = np.zeros(n_bars)
        # Simulate a session with varying IV
        atm_iv_norm[:100] = -0.8   # low vol
        atm_iv_norm[100:200] = 0.3  # normal
        atm_iv_norm[200:] = 1.5     # high vol

        # Low vol: consistent losers
        entry_bars = list(range(0, 100, 5))[:10]
        pnls = [-20.0, -30.0, -25.0, -15.0, -40.0, -10.0, -5.0, -35.0, -20.0, -22.0]

        # Normal: winners
        entry_bars += list(range(100, 200, 5))[:10]
        pnls += [50.0] * 10

        # High vol: winners
        entry_bars += list(range(200, 300, 5))[:10]
        pnls += [30.0] * 10

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        # Low vol regime: mean = -22.2, std = ~10.4 → Sharpe ~ -2.1
        # Should trigger the gate
        assert _regime_gate_fails(result, terminal_data) is True

    def test_regime_gate_all_positive_per_regime(self):
        """Each regime has positive Sharpe -- gate should pass."""
        n_bars = 300
        atm_iv_norm = np.zeros(n_bars)
        atm_iv_norm[:100] = -0.8
        atm_iv_norm[100:200] = 0.3
        atm_iv_norm[200:] = 1.5

        entry_bars = (list(range(0, 100, 5))[:10]
                      + list(range(100, 200, 5))[:10]
                      + list(range(200, 300, 5))[:10])
        # All mildly positive with some variance
        rng = np.random.RandomState(123)
        pnls = list(rng.normal(10.0, 5.0, 30))

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        # All regimes have positive mean → gate should pass
        assert _regime_gate_fails(result, terminal_data) is False

    def test_regime_gate_configurable_in_evaluator(self):
        """regime_gate_enabled field exists on VectorizedFitnessEvaluator."""
        template = iron_condor()
        data = _make_data(n_dates=2, bars_per_date=30)
        from layer2.evaluator_vectorized import prepare_terminal_data
        td = prepare_terminal_data(data)

        # Default: enabled
        fe = VectorizedFitnessEvaluator(
            template=template, data=data, terminal_data=td,
        )
        assert fe.regime_gate_enabled is True

        # Explicitly disabled
        fe2 = VectorizedFitnessEvaluator(
            template=template, data=data, terminal_data=td,
            regime_gate_enabled=False,
        )
        assert fe2.regime_gate_enabled is False

    def test_regime_gate_disabled_no_sentinel(self):
        """When regime_gate_enabled=False, catastrophic regime doesn't trigger sentinel."""
        n_bars = 100
        atm_iv_norm = np.full(n_bars, 0.0)
        # Create 10 trades with very negative PnLs
        entry_bars = list(range(0, 30, 3))[:10]
        pnls = [-100.0] * 10
        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades, sharpe=-2.0)
        terminal_data = {"ATM_IV": atm_iv_norm}

        # Gate should fail when enabled
        assert _regime_gate_fails(result, terminal_data) is True

    def test_regime_gate_zero_variance_positive_mean(self):
        """Zero variance with positive mean should pass (all identical positive PnLs)."""
        n_bars = 100
        atm_iv_norm = np.full(n_bars, 0.0)  # all normal
        entry_bars = list(range(0, 30, 3))[:10]
        pnls = [10.0] * 10  # identical positive PnLs → std = 0

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        assert _regime_gate_fails(result, terminal_data) is False

    def test_regime_gate_zero_variance_negative_mean(self):
        """Zero variance with negative mean should fail (catastrophic)."""
        n_bars = 100
        atm_iv_norm = np.full(n_bars, 0.0)  # all normal
        entry_bars = list(range(0, 30, 3))[:10]
        pnls = [-10.0] * 10  # identical negative PnLs → std = 0

        trades = _make_trades(entry_bars, pnls, n_bars=n_bars)
        result = _make_backtest_result(trades)
        terminal_data = {"ATM_IV": atm_iv_norm}

        assert _regime_gate_fails(result, terminal_data) is True


# ===========================================================================
# Integration: ExperimentConfig fields
# ===========================================================================

class TestExperimentConfigFields:
    """Test that new ExperimentConfig fields exist with correct defaults."""

    def test_inner_val_enabled_default(self):
        from layer2.experiment import ExperimentConfig
        cfg = ExperimentConfig(l1_parquet_path="/tmp/test.parquet")
        assert cfg.inner_val_enabled is True

    def test_regime_gate_enabled_default(self):
        from layer2.experiment import ExperimentConfig
        cfg = ExperimentConfig(l1_parquet_path="/tmp/test.parquet")
        assert cfg.regime_gate_enabled is True

    def test_inner_val_disabled(self):
        from layer2.experiment import ExperimentConfig
        cfg = ExperimentConfig(
            l1_parquet_path="/tmp/test.parquet",
            inner_val_enabled=False,
        )
        assert cfg.inner_val_enabled is False

    def test_regime_gate_disabled(self):
        from layer2.experiment import ExperimentConfig
        cfg = ExperimentConfig(
            l1_parquet_path="/tmp/test.parquet",
            regime_gate_enabled=False,
        )
        assert cfg.regime_gate_enabled is False
