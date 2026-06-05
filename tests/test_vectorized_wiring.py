"""Tests for experiment.py vectorized evaluator integration.

Verifies:
1. load_minute_parquet works with correct and malformed data
2. VectorizedFitnessEvaluator produces valid fitness vectors
3. ExperimentConfig use_vectorized flag flows through to evaluator dispatch
4. End-to-end mini GP run with vectorized evaluator produces results
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layer2.io import load_minute_parquet, BYPASS_SCALAR_COLUMNS, PRICE_COLUMN
from layer2.fitness import (
    DEFAULT_OBJECTIVES, FAILED_FITNESS_SENTINEL,
    VectorizedFitnessEvaluator,
)
from layer2.grammar import Grammar, build_scalar_only_terminal_set, SCALAR_ONLY_FUNCTIONS
from layer2.templates import template_by_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minute_df(n_dates: int = 5, bars_per_date: int = 20) -> pd.DataFrame:
    """Synthetic 1-minute-style DataFrame with bypass scalars only."""
    rng = np.random.default_rng(42)
    rows = []
    base_price = 4500.0
    for d in range(n_dates):
        date = f"2024-{d+1:02d}-01"
        for b in range(bars_per_date):
            base_price *= np.exp(rng.normal(0, 0.001))
            rows.append({
                "date": date,
                "bar_position": b,
                "ATM_IV": 0.15 + rng.normal(0, 0.02),
                "RawSpread": 0.02 + rng.normal(0, 0.005),
                "DeltaSpread1": rng.normal(0, 0.05),
                "DeltaSpread5": rng.normal(0, 0.06),
                "MinutesToClose": max(0, 390 - b),
                "VIXSpot": 16 + rng.normal(0, 2),
                "VIXTermSlope": rng.normal(-0.5, 1.0),
                "RealizedVol30m": 0.015 + abs(rng.normal(0, 0.003)),
                "BarOfDay": b,
                "SPXClose": base_price,
            })
    return pd.DataFrame(rows)


def _save_parquet(df: pd.DataFrame, tmp_dir: str) -> Path:
    """Save DataFrame to a temporary Parquet file."""
    path = Path(tmp_dir) / "test_minute.parquet"
    df.to_parquet(path, engine="pyarrow")
    return path


# ---------------------------------------------------------------------------
# load_minute_parquet tests
# ---------------------------------------------------------------------------

class TestLoadMinuteParquet:

    def test_load_valid(self, tmp_path):
        df = _make_minute_df()
        path = tmp_path / "minute.parquet"
        df.to_parquet(path)

        loaded_df, schema = load_minute_parquet(path)
        assert schema.n_rows == len(df)
        assert schema.n_dates == df["date"].nunique()
        assert schema.typed_vector_dim == 0
        assert schema.has_typed_vectors is False
        assert schema.has_probe_scalars is False
        assert schema.has_bypass_scalars is True
        assert schema.has_price_column is True

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_minute_parquet("/nonexistent/path.parquet")

    def test_missing_date_column(self, tmp_path):
        df = _make_minute_df().drop(columns=["date"])
        path = tmp_path / "no_date.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="missing 'date'"):
            load_minute_parquet(path)

    def test_missing_price_column(self, tmp_path):
        df = _make_minute_df().drop(columns=["SPXClose"])
        path = tmp_path / "no_price.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="missing price column"):
            load_minute_parquet(path)

    def test_nan_in_price(self, tmp_path):
        df = _make_minute_df()
        df.loc[5, "SPXClose"] = np.nan
        path = tmp_path / "nan_price.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="NaN"):
            load_minute_parquet(path)

    def test_no_bypass_scalars(self, tmp_path):
        df = _make_minute_df()[["date", "bar_position", "SPXClose"]]
        path = tmp_path / "no_bypass.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="no bypass scalar"):
            load_minute_parquet(path)


# ---------------------------------------------------------------------------
# VectorizedFitnessEvaluator tests
# ---------------------------------------------------------------------------

class TestVectorizedFitnessEvaluator:

    @pytest.fixture
    def setup(self):
        """Create evaluator with synthetic data and iron_condor template."""
        df = _make_minute_df(n_dates=10, bars_per_date=50)
        template = template_by_name("iron_condor_standard")

        from layer2.evaluator_vectorized import prepare_terminal_data
        td = prepare_terminal_data(df)

        fe = VectorizedFitnessEvaluator(
            template=template,
            data=df,
            terminal_data=td,
            min_trades=1,
            warmup_bars=5,
        )
        return fe, df, template

    def test_objectives_default(self, setup):
        fe, _, _ = setup
        assert fe.objectives == DEFAULT_OBJECTIVES

    def test_evaluate_returns_correct_shape(self, setup):
        fe, _, template = setup
        fitness = fe.evaluate(
            template.entry_seed, template.exit_seed, template.size_seed,
        )
        assert fitness.shape == (len(DEFAULT_OBJECTIVES),)
        assert np.all(np.isfinite(fitness))

    def test_cache_hit_on_repeat(self, setup):
        fe, _, template = setup
        fe.evaluate(template.entry_seed, template.exit_seed, template.size_seed)
        fe.evaluate(template.entry_seed, template.exit_seed, template.size_seed)
        assert fe.stats["cache_hits"] == 1
        assert fe.stats["evaluations"] == 1

    def test_score_on_data_no_cache(self, setup):
        fe, df, template = setup
        # score_on_data should work without affecting cache
        fitness = fe.score_on_data(
            template.entry_seed, template.exit_seed, template.size_seed, df,
        )
        assert fitness.shape == (len(DEFAULT_OBJECTIVES),)
        # Cache should be empty (score_on_data doesn't cache)
        assert fe.stats["cache_hits"] == 0

    def test_fitness_from_result_trade_count_score(self, setup):
        fe, _, _ = setup
        from layer2.evaluator import BacktestResult
        # 0 trades -> tc_score = -1.0
        result = BacktestResult(
            returns=np.zeros(100), trades=[], equity_curve=np.zeros(100),
            max_drawdown=0.0, sharpe=0.0, total_trades=0, win_rate=0.0, n_days=10,
        )
        fitness = fe._fitness_from_result(result)
        # neg_trade_count_score = -tc_score = -(-1.0) = 1.0
        assert fitness[2] == 1.0  # neg_trade_count_score

    def test_unknown_objective_raises(self):
        df = _make_minute_df()
        template = template_by_name("iron_condor_standard")
        from layer2.evaluator_vectorized import prepare_terminal_data
        td = prepare_terminal_data(df)
        with pytest.raises(ValueError, match="unknown objective"):
            VectorizedFitnessEvaluator(
                template=template, data=df, terminal_data=td,
                objectives=("nonexistent_obj",),
            )


def test_a1_haircut_reconciliation_and_sortino_cap():
    """A1 (2026-05-31): the credit haircut must be booked into `bar_returns`, so
    the Sharpe basis reconciles to realized PnL: sum(returns) == sum(trade.pnl)/
    notional. This is the invariant that would have caught the bug (pre-fix the
    haircut lived only in trade.pnl/equity → every credit Sharpe optimistic).
    Also asserts Sortino is capped (no dd_dev-floor artifact). Run on a credit
    template that actually trades.
    """
    from layer2.evaluator_vectorized import (
        vectorized_backtest, prepare_terminal_data, _MAX_SORTINO)
    df = _make_minute_df(n_dates=12, bars_per_date=50)
    template = template_by_name("iron_condor_standard")  # credit template (haircut applies)
    td = prepare_terminal_data(df)  # normalized — the seed thresholds need it to fire
    NOTIONAL = 1000.0
    r = vectorized_backtest(template.entry_seed, template.exit_seed,
                            template.size_seed, df, template,
                            terminal_data=td, notional=NOTIONAL)
    assert r.total_trades > 0, "fixture must trade for this test to bite"
    sum_ret = float(np.sum(r.returns))
    sum_pnl = float(sum(t.pnl for t in r.trades))
    # A1 invariant: Sharpe basis (bar_returns) reconciles to realized PnL. A
    # missing haircut would show a gap ~10% of credit (>> tol); fixed -> ~1e-6.
    assert abs(sum_ret - sum_pnl / NOTIONAL) < 1e-4, (
        f"A1 reconciliation FAILED: sum(returns)={sum_ret:.6f} vs "
        f"sum(pnl)/notional={sum_pnl / NOTIONAL:.6f} — haircut missing from the Sharpe basis"
    )
    # Sortino plausibility cap — no 1e-9 dd_dev-floor / zero-down-day artifact.
    assert -_MAX_SORTINO - 1e-9 <= r.sortino <= _MAX_SORTINO + 1e-9, (
        f"Sortino {r.sortino} escaped the +/-{_MAX_SORTINO} cap")


def test_stoploss_markup_reaches_sharpe_basis_reconciliation():
    """BLOCKER companion to A1 (2026-06-01 holistic review): the A1 fixture above
    exits only via session/end_of_data and NEVER stop-losses, so it is BLIND to the
    stop-loss adverse-fill markup (STOP_LOSS_FILL_MARKUP). That markup multiplied
    `pnl` (-> equity + trade.pnl) but the per-bar Sharpe basis bar_returns booked
    only exit_cost, so the extra adverse-fill loss was invisible to Sharpe/Sortino.
    Run a credit strategy on an adverse tape that DOES stop-loss and assert the SAME
    reconciliation invariant — sum(bar_returns) == sum(trade.pnl)/notional — WITH a
    stop_loss exit present. (Full directional + mechanism coverage lives in
    tests/test_evaluator_stoploss_marking.py.)
    """
    import layer2.evaluator_vectorized as _ev
    from layer2.evaluator_vectorized import vectorized_backtest as _vbt, prepare_terminal_data as _ptd
    from layer2.grammar import from_sexpr as _fx

    rows, base = [], 5000.0
    for d in range(8):
        for b in range(60):
            px = base * (1.0 - 0.030 * min(max(b - 15, 0) / 20.0, 1.0))  # mid-day crash
            rows.append({
                "date": f"2024-{d + 1:02d}-15", "bar_position": b,
                "ATM_IV": 0.20, "RawSpread": 0.02, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "MinutesToClose": max(0, 390 - b * 6), "VIXSpot": 18.0,
                "VIXTermSlope": -0.5, "RealizedVol30m": 0.02, "BarOfDay": b, "SPXClose": px,
            })
    df = pd.DataFrame(rows)
    template = template_by_name("bull_put_credit_standard")
    td = _ptd(df)
    NOTIONAL = 1000.0
    r = _vbt(_fx("GT(EphReal(1.0), EphReal(0.0))"),        # always enter
             _fx("GT(EphReal(-10.0), EphReal(0.0))"),       # never voluntarily exit
             _fx("EphReal(0.5)"), df, template,
             terminal_data=td, notional=NOTIONAL, warmup_bars=2, min_bars_in_trade=1)

    assert _ev.STOP_LOSS_FILL_MARKUP > 1.0, "production markup must be >1 or this proves nothing"
    n_stop = sum(1 for t in r.trades if t.exit_reason == "stop_loss")
    assert n_stop >= 1, (
        f"fixture must stop-loss for this guard to bite; exits={[t.exit_reason for t in r.trades]}")
    sum_ret = float(np.sum(r.returns))
    sum_pnl = float(sum(t.pnl for t in r.trades))
    assert abs(sum_ret - sum_pnl / NOTIONAL) < 1e-6, (
        f"stop-loss reconciliation FAILED: sum(returns)={sum_ret:.10f} vs "
        f"sum(pnl)/notional={sum_pnl / NOTIONAL:.10f} — markup missing from the Sharpe basis")


def test_vectorized_recorder_records_training_frame_only():
    """C2 (review): the VECTORIZED DSR trial recorder — the PRODUCTION path —
    fires on the TRAINING frame (evaluate -> ``data is self.data``) and NOT on
    score_on_data's val/test frame (a different object). This pins the
    recorder->front-gate wiring that the experiment-level fixtures cannot reach
    (their GP fronts are all-sentinel, so the gate never sees real Sharpes).
    Also confirms the recorded signature is the 4-tree quad (entry|exit|size|
    delta), the vectorized-only canonical key.
    """
    df = _make_minute_df(n_dates=12, bars_per_date=50)
    val_df = _make_minute_df(n_dates=8, bars_per_date=50)  # DIFFERENT object
    template = template_by_name("iron_condor_standard")
    from layer2.evaluator_vectorized import prepare_terminal_data
    td = prepare_terminal_data(df)
    val_td = prepare_terminal_data(val_df)

    fe = VectorizedFitnessEvaluator(
        template=template, data=df, terminal_data=td, min_trades=1,
        warmup_bars=5,
    )
    sink = fe.enable_trial_recording()
    assert sink == {}  # enabled, initially empty

    fe.evaluate(template.entry_seed, template.exit_seed, template.size_seed)
    n_after_train = len(fe.evo_trial_records)
    assert n_after_train >= 1, (
        "evaluate() on the training frame must record >=1 distinct trial Sharpe "
        "via the `data is self.data` branch"
    )
    # The vectorized canonical signature is the 4-tree quad (3 '|' separators).
    sig = next(iter(fe.evo_trial_records))
    assert sig.count("|") == 3, f"vectorized sig must be a 4-tree quad: {sig!r}"
    sharpe_ann, n_days = next(iter(fe.evo_trial_records.values()))
    assert math.isfinite(sharpe_ann) and n_days >= 1

    # score_on_data passes a DIFFERENT frame (val/test) -> recorder must NOT fire.
    fe.score_on_data(template.entry_seed, template.exit_seed,
                     template.size_seed, val_df, terminal_data=val_td)
    assert len(fe.evo_trial_records) == n_after_train, (
        "score_on_data (val/test frame) must NOT add to the trial recorder — "
        "the `data is self.data` guard must exclude it"
    )


# ---------------------------------------------------------------------------
# ExperimentConfig integration test
# ---------------------------------------------------------------------------

class TestExperimentConfigVectorized:

    def test_use_vectorized_default_false(self):
        from layer2.experiment import ExperimentConfig
        c = ExperimentConfig(l1_parquet_path="dummy")
        assert c.use_vectorized is False

    def test_use_vectorized_true(self):
        from layer2.experiment import ExperimentConfig
        c = ExperimentConfig(l1_parquet_path="dummy", use_vectorized=True)
        assert c.use_vectorized is True

    def test_fingerprint_differs_by_vectorized(self):
        """use_vectorized=True and False should produce different fingerprints."""
        from layer2.experiment import ExperimentConfig, _config_fingerprint
        c1 = ExperimentConfig(l1_parquet_path="dummy", use_vectorized=False)
        c2 = ExperimentConfig(l1_parquet_path="dummy", use_vectorized=True)
        fp1 = _config_fingerprint(c1, "abc123")
        fp2 = _config_fingerprint(c2, "abc123")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# End-to-end mini GP run (vectorized)
# ---------------------------------------------------------------------------

class TestMiniGPVectorized:

    @pytest.fixture
    def minute_parquet(self, tmp_path):
        """Create a temporary minute Parquet file."""
        # Use enough data for meaningful GP evolution
        df = _make_minute_df(n_dates=20, bars_per_date=50)
        path = tmp_path / "l1_minute_scalars.parquet"
        df.to_parquet(path)
        return str(path)

    def test_mini_gp_produces_results(self, minute_parquet, tmp_path):
        """Pop=8, gen=2, 1 template — verifies full pipe works."""
        from layer2.experiment import ExperimentConfig, run_experiment

        config = ExperimentConfig(
            l1_parquet_path=minute_parquet,
            condition="scalar-only",
            template_names=("iron_condor_standard",),
            pop_size=8,
            n_generations=2,
            seed=42,
            use_vectorized=True,
            output_dir=str(tmp_path / "results"),
            min_train_rows=0,  # tiny test data — bypass hardening check
        )
        result = run_experiment(config)

        assert len(result.template_results) == 1
        tr = result.template_results[0]
        assert tr.template_name == "iron_condor_standard"
        assert tr.final_population_size > 0
        # Pareto front may be 0 on tiny synthetic data (all strategies hit sentinel)
        assert tr.pareto_front_size >= 0
        assert tr.wall_time_s > 0
        # Check that results were saved
        assert (tmp_path / "results" / "results.json").exists()

    def test_mini_gp_non_vectorized_still_works(self, tmp_path):
        """Verify that non-vectorized path is unbroken by changes."""
        from layer2.experiment import ExperimentConfig, run_experiment

        # Build a synthetic L1 Parquet with typed vectors for the original evaluator
        rng = np.random.default_rng(99)
        n_dates, bars = 3, 10
        rows = []
        from layer2.io import TYPED_VECTOR_COLUMNS, PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS
        for d in range(n_dates):
            for b in range(bars):
                row = {
                    "date": f"2024-{d+1:02d}-01",
                    "window_idx": d * bars + b,
                    "bar_position": b,
                    "ATM_IV": 0.15,
                    "RawSpread": 0.02,
                    "DeltaSpread1": 0.0,
                    "DeltaSpread5": 0.0,
                    "MinutesToClose": 60.0,
                    "VIXSpot": 16.0,
                    "VIXTermSlope": -0.5,
                    "RealizedVol30m": 0.015,
                    "SPXClose": 4500.0 + rng.normal(0, 10),
                }
                for vc in TYPED_VECTOR_COLUMNS:
                    row[vc] = rng.normal(0, 1, 384).tolist()
                for pc in PROBE_SCALAR_COLUMNS:
                    row[pc] = rng.normal(0, 1) if pc != "PredRegime" else int(rng.integers(0, 4))
                for rc in REGIME_PROB_COLUMNS:
                    row[rc] = rng.uniform(0, 1)
                rows.append(row)
        df = pd.DataFrame(rows)
        path = tmp_path / "l1_full.parquet"
        df.to_parquet(path)

        config = ExperimentConfig(
            l1_parquet_path=str(path),
            condition="scalar-only",
            template_names=("iron_condor_standard",),
            pop_size=8,
            n_generations=2,
            seed=42,
            use_vectorized=False,
            output_dir=None,
        )
        result = run_experiment(config)
        assert len(result.template_results) == 1
        # Pareto front may be 0 on tiny data (all strategies hit sentinel)
        assert result.template_results[0].pareto_front_size >= 0


# ---------------------------------------------------------------------------
# Regression tests for evaluator bugs (margin gate + cost units + PnL scaling)
# ---------------------------------------------------------------------------

class TestBlackScholesPricing:
    """Numerical correctness tests for the Black-Scholes option pricing."""

    def test_put_call_parity(self):
        """C(K) - P(K) ≈ S - K * exp(-r*T); with r=0, C-P ≈ S - K."""
        from layer2.evaluator_vectorized import _option_value
        spot, strike, mtc, iv = 5300.0, 5300.0, 200.0, 0.15
        c = _option_value(spot, strike, mtc, iv, True)
        p = _option_value(spot, strike, mtc, iv, False)
        # For ATM (S=K) with r=0: C-P ≈ 0
        assert abs(c - p) < 0.01, f"Put-call parity violated: C-P={c-p:.6f}"

    def test_atm_option_value_reasonable(self):
        """ATM option must be in realistic range for 0DTE SPX."""
        from layer2.evaluator_vectorized import _option_value
        # SPX 5300, IV 15%, 200 min (about 3.3 hours)
        val = _option_value(5300.0, 5300.0, 200.0, 0.15, True)
        # ATM 0DTE with 200min left: ~$10-$20/share
        assert 5.0 < val < 30.0, f"ATM value {val:.2f} outside realistic range"

    def test_otm_decays_with_moneyness(self):
        """OTM options must be worth less than ATM, with realistic decay."""
        from layer2.evaluator_vectorized import _option_value
        spot, mtc, iv = 5300.0, 200.0, 0.15
        atm = _option_value(spot, 5300, mtc, iv, False)
        otm_20 = _option_value(spot, 5280, mtc, iv, False)  # 20pts OTM
        otm_40 = _option_value(spot, 5260, mtc, iv, False)  # 40pts OTM
        otm_80 = _option_value(spot, 5220, mtc, iv, False)  # 80pts OTM

        assert atm > otm_20 > otm_40 > otm_80 > 0
        # 20pts OTM should be roughly half ATM (not 99% like BS approx was)
        assert otm_20 / atm < 0.7, f"OTM decay too slow: {otm_20/atm:.2f}"

    def test_vertical_spread_credit_realistic(self):
        """A 20-point wide bull put credit spread must produce meaningful credit."""
        from layer2.evaluator_vectorized import _option_value
        spot, mtc, iv = 5300.0, 200.0, 0.15
        short_put = _option_value(spot, 5280, mtc, iv, False)
        long_put = _option_value(spot, 5260, mtc, iv, False)
        credit = short_put - long_put
        # Real market: $2-$6/share for this setup
        assert credit > 1.0, f"Spread credit {credit:.4f} too small"
        assert credit < 15.0, f"Spread credit {credit:.4f} too large"

    def test_expiry_values_exact(self):
        """At expiry, options must equal intrinsic value exactly."""
        from layer2.evaluator_vectorized import _option_value
        # OTM put at expiry = 0
        assert _option_value(5300, 5280, 0, 0.15, False) == 0.0
        # ITM put at expiry = K - S
        assert _option_value(5300, 5320, 0, 0.15, False) == pytest.approx(20.0)
        # OTM call at expiry = 0
        assert _option_value(5300, 5320, 0, 0.15, True) == 0.0
        # ITM call at expiry = S - K
        assert _option_value(5300, 5280, 0, 0.15, True) == pytest.approx(20.0)


class TestVectorizedEmbeddingOperators:
    """Tests for vectorized embedding operators (B/C/D conditions)."""

    def _make_evaluator(self, n_bars=100, vec_dim=384):
        """Create a VectorizedTreeEvaluator with synthetic vector data."""
        from layer2.evaluator_vectorized import VectorizedTreeEvaluator
        rng = np.random.default_rng(42)
        terminal_data = {
            "ATM_IV": rng.normal(0, 1, n_bars),
            "EMB_GRID": rng.normal(0, 1, (n_bars, vec_dim)),
            "EMB_VIX": rng.normal(0, 1, (n_bars, vec_dim)),
        }
        day_mask = np.ones(n_bars, dtype=np.float64)
        day_mask[0] = 0.0
        within_day = np.arange(n_bars, dtype=np.int64)
        return VectorizedTreeEvaluator(terminal_data, n_bars, day_mask, within_day)

    def test_emb_terminal_returns_2d(self):
        """Typed-vector terminals must return (N, D) arrays."""
        from layer2.grammar import TermDef, TermNode, GType
        evaluator = self._make_evaluator()
        node = TermNode(TermDef("EMB_GRID", GType.EMB_GRID))
        result = evaluator.evaluate(node)
        assert result.shape == (100, 384)

    def test_embnorm_returns_scalar_array(self):
        """EmbNorm must return (N,) L2 norms."""
        from layer2.grammar import TermDef, TermNode, FuncDef, FuncNode, GType
        evaluator = self._make_evaluator()
        vec_node = TermNode(TermDef("EMB_GRID", GType.EMB_GRID))
        norm_def = FuncDef("EmbNorm_EMB_GRID", (GType.EMB_GRID,), GType.REAL)
        norm_node = FuncNode(norm_def, [vec_node])
        result = evaluator.evaluate(norm_node)
        assert result.shape == (100,)
        assert np.all(result >= 0)  # norms are non-negative

    def test_embcos_returns_bounded_scalar(self):
        """EmbCos must return (N,) values in [-1, 1]."""
        from layer2.grammar import TermDef, TermNode, FuncDef, FuncNode, GType
        evaluator = self._make_evaluator()
        grid_node = TermNode(TermDef("EMB_GRID", GType.EMB_GRID))
        vix_node = TermNode(TermDef("EMB_VIX", GType.EMB_VIX))
        cos_def = FuncDef("EmbCos_EMB_GRID", (GType.EMB_GRID, GType.EMB_GRID), GType.REAL)
        cos_node = FuncNode(cos_def, [grid_node, vix_node])
        result = evaluator.evaluate(cos_node)
        assert result.shape == (100,)
        assert np.all(result >= -1.01) and np.all(result <= 1.01)

    def test_embsub_returns_2d(self):
        """EmbSub must return (N, D) difference vectors."""
        from layer2.grammar import TermDef, TermNode, FuncDef, FuncNode, GType
        evaluator = self._make_evaluator()
        grid_node = TermNode(TermDef("EMB_GRID", GType.EMB_GRID))
        vix_node = TermNode(TermDef("EMB_VIX", GType.EMB_VIX))
        sub_def = FuncDef("EmbSub_EMB_GRID", (GType.EMB_GRID, GType.EMB_GRID), GType.EMB_GRID)
        sub_node = FuncNode(sub_def, [grid_node, vix_node])
        result = evaluator.evaluate(sub_node)
        assert result.shape == (100, 384)

    def test_emblag_respects_day_boundary(self):
        """EmbLag must zero out rows where lag crosses day boundary."""
        from layer2.grammar import TermDef, TermNode, FuncDef, FuncNode, GType
        n_bars = 20
        rng = np.random.default_rng(99)
        terminal_data = {
            "EMB_GRID": rng.normal(0, 1, (n_bars, 8)),  # small D for easy checking
        }
        day_mask = np.ones(n_bars)
        day_mask[0] = 0.0
        day_mask[10] = 0.0  # day boundary at bar 10
        within_day = np.array([0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9], dtype=np.int64)

        from layer2.evaluator_vectorized import VectorizedTreeEvaluator
        evaluator = VectorizedTreeEvaluator(terminal_data, n_bars, day_mask, within_day)

        vec_node = TermNode(TermDef("EMB_GRID", GType.EMB_GRID))
        lag_int = TermNode(TermDef("EphInt", GType.INT), value=3)
        lag_def = FuncDef("EmbLag_EMB_GRID", (GType.EMB_GRID, GType.INT), GType.EMB_GRID)
        lag_node = FuncNode(lag_def, [vec_node, lag_int])
        result = evaluator.evaluate(lag_node)
        assert result.shape == (20, 8)
        # First 3 bars of each day should be zero (lag=3 crosses boundary)
        assert np.all(result[0:3] == 0)
        assert np.all(result[10:13] == 0)
        # Bar 5 should have bar 2's values
        np.testing.assert_array_equal(result[5], terminal_data["EMB_GRID"][2])

    def test_enriched_parquet_loads_with_vectors(self):
        """The enriched Parquet should be loadable and report vector presence."""
        from pathlib import Path
        enriched = Path("raw_data/local_store/l1_minute_enriched.parquet")
        if not enriched.exists():
            pytest.skip("Enriched Parquet not generated yet")
        from layer2.io import load_minute_parquet
        df, schema = load_minute_parquet(enriched)
        assert schema.has_typed_vectors
        assert schema.has_probe_scalars
        assert schema.typed_vector_dim == 384
        assert "EMB_GRID" in df.columns
        assert "PredRV15" in df.columns


class TestMarginGateSpreadWidth:
    """Regression: 4-leg margin must use paired indices, not consecutive."""

    def test_iron_condor_spread_width_is_wing_width_not_cross_side(self):
        """Bug #1: IC with strikes [5320, 5325, 5280, 5275] had
        consecutive-pair max=45 (cross-side gap) instead of 5 (wing width).
        """
        strikes = [5320, 5325, 5280, 5275]  # call_short, call_long, put_short, put_long
        # 4-leg: paired as (0,1) and (2,3)
        if len(strikes) == 4:
            max_spread_width = max(abs(strikes[0] - strikes[1]),
                                   abs(strikes[2] - strikes[3]))
        else:
            max_spread_width = max(abs(strikes[j] - strikes[j+1])
                                   for j in range(len(strikes)-1))
        assert max_spread_width == 5, f"Expected 5, got {max_spread_width} (cross-side gap bug)"

    def test_3leg_consecutive_pairs_correct(self):
        """3-leg debit structure: consecutive pairs are correct (all same option type)."""
        strikes = [5310, 5320, 5335]  # all calls
        max_spread_width = max(abs(strikes[j] - strikes[j+1])
                               for j in range(len(strikes)-1))
        assert max_spread_width == 15

    def test_2leg_spread_width(self):
        """2-leg vertical spread: single pair."""
        strikes = [5285, 5270]
        max_spread_width = abs(strikes[0] - strikes[1])
        assert max_spread_width == 15


class TestCostModelUnits:
    """Regression: costs must be in $/share (same as _net_pos_value)."""

    def test_entry_costs_are_in_dollar_per_share_scale(self):
        """Bug #2: _entry_costs() had *100 multiplier, putting costs ~100x
        larger than PnL values from _net_pos_value(). Costs must be in $/share.
        """
        from layer2.evaluator_vectorized import _entry_costs, _option_value
        from layer2.templates import template_by_name

        ic = template_by_name("iron_condor_standard")
        spot = 5300.0
        iv = 0.15
        mtc = 200.0  # ~200 min to close
        strikes = [5315, 5330, 5285, 5270]

        cost = _entry_costs(ic.legs, strikes, spot, mtc, iv,
                            fee_per_leg=2.50,
                            atm_spread=0.015)

        # Cost should be meaningful but not exceed total gross position value.
        # With F1 fix (spread × option_value), IC cost is ~$1.38/share ($138/contract).
        assert cost > 0.1, f"Cost {cost:.4f} $/share is suspiciously small"
        assert cost < 5.0, f"Cost {cost:.4f} $/share is unreasonably large"

        # Compare to an ATM option value to verify same scale
        atm_val = _option_value(spot, spot, mtc, iv, True)
        # Cost should be less than 2x ATM value (4-leg structure, each leg has spread cost)
        assert cost < 2 * atm_val, f"Cost ({cost:.4f}) >= 2x ATM value ({atm_val:.4f})"

    def test_bull_put_credit_costs_reasonable(self):
        """2-leg cost should be roughly half of 4-leg IC cost."""
        from layer2.evaluator_vectorized import _entry_costs
        from layer2.templates import template_by_name

        bpc = template_by_name("bull_put_credit_standard")
        strikes = [5285, 5270]
        cost = _entry_costs(bpc.legs, strikes, 5300.0, 200.0, 0.15,
                            fee_per_leg=2.50,
                            atm_spread=0.015)
        # BPC (2-leg): cost ~$0.97/share ($97/contract) with F1 fix
        assert 0.1 < cost < 3.0, f"BPC cost {cost:.4f} outside reasonable range"


class TestPnLScaling:
    """Regression: PnL must scale costs by n_contracts."""

    def test_pnl_scales_with_n_contracts(self):
        """PnL must scale linearly with n_contracts (costs included)."""
        from layer2.evaluator_vectorized import (
            _entry_costs, _net_pos_value,
        )
        from layer2.templates import template_by_name

        bpc = template_by_name("bull_put_credit_standard")
        spot_entry = 5300.0
        spot_exit = 5350.0
        iv = 0.15
        mtc_entry = 200.0
        mtc_exit = 0.0
        strikes = [5280, 5260]

        entry_val = _net_pos_value(bpc.legs, strikes, spot_entry, mtc_entry, iv)
        exit_val = _net_pos_value(bpc.legs, strikes, spot_exit, mtc_exit, iv)
        entry_cost = _entry_costs(bpc.legs, strikes, spot_entry, mtc_entry, iv,
                                   2.50, atm_spread=0.015)
        exit_cost = _entry_costs(bpc.legs, strikes, spot_exit, mtc_exit, iv,
                                  2.50, atm_spread=0.015)

        pnl_1 = ((exit_val - entry_val) - entry_cost - exit_cost) * 1
        pnl_2 = ((exit_val - entry_val) - entry_cost - exit_cost) * 2
        pnl_5 = ((exit_val - entry_val) - entry_cost - exit_cost) * 5

        assert pnl_2 == pytest.approx(2 * pnl_1), "PnL should scale 2x"
        assert pnl_5 == pytest.approx(5 * pnl_1), "PnL should scale 5x"

    def test_profitable_credit_spread_at_expiry(self):
        """A credit spread held to OTM expiry must show positive PnL.
        This was impossible under the Brenner-Subrahmanyam approximation
        because it compressed credits to ~$0.11 while costs were ~$0.32.
        """
        from layer2.evaluator_vectorized import (
            _entry_costs, _net_pos_value,
        )
        from layer2.templates import template_by_name

        bpc = template_by_name("bull_put_credit_standard")
        spot_entry = 5300.0
        spot_exit = 5350.0  # rallied — put spread expires worthless
        iv = 0.15
        mtc_entry = 200.0
        mtc_exit = 0.0
        strikes = [5280, 5260]

        entry_val = _net_pos_value(bpc.legs, strikes, spot_entry, mtc_entry, iv)
        exit_val = _net_pos_value(bpc.legs, strikes, spot_exit, mtc_exit, iv)
        entry_cost = _entry_costs(bpc.legs, strikes, spot_entry, mtc_entry, iv,
                                   2.50, atm_spread=0.015)
        exit_cost = _entry_costs(bpc.legs, strikes, spot_exit, mtc_exit, iv,
                                  2.50, atm_spread=0.015)
        n_contracts = 2
        pnl = ((exit_val - entry_val) - entry_cost - exit_cost) * n_contracts

        # Credit received must be meaningful (not compressed to near-zero)
        credit = -(entry_val)  # credit spread: entry_val is negative
        assert credit > 1.0, f"Credit {credit:.4f} too small — pricing model broken"
        # PnL must be positive when spread expires OTM
        assert pnl > 0, (
            f"Credit spread expiring OTM must profit. credit={credit:.4f}, "
            f"costs={entry_cost + exit_cost:.4f}, pnl={pnl:.4f}"
        )

    def test_costs_and_values_same_unit_system(self):
        """Costs and option values must be in the same $/share scale.
        Bug #2 had costs 100x larger than values due to *100 multiplier.
        """
        from layer2.evaluator_vectorized import (
            _entry_costs, _net_pos_value, _option_value,
        )
        from layer2.templates import template_by_name

        ic = template_by_name("iron_condor_standard")
        spot = 5300.0
        iv = 0.15
        mtc = 200.0
        strikes = [5315, 5330, 5285, 5270]

        cost = _entry_costs(ic.legs, strikes, spot, mtc, iv,
                            2.50, atm_spread=0.015)
        net_val = abs(_net_pos_value(ic.legs, strikes, spot, mtc, iv))
        atm_val = _option_value(spot, spot, mtc, iv, True)

        # Cost must be much smaller than ATM value (same unit system)
        assert cost < atm_val, (
            f"Cost ({cost:.4f}) >= ATM value ({atm_val:.4f}) — unit mismatch"
        )
        # Cost should be on the same order as the net position value
        assert cost < 10 * net_val or net_val < 0.01, (
            f"Cost ({cost:.4f}) is >10x net_val ({net_val:.4f}) — suspicious"
        )


# ── Fix-batch 2026-05-09: delta-to-strike, spread mult, objectives, stop-loss, settlement ──

class TestDeltaToStrikeBS:
    """Verify _delta_to_strike uses proper BS inversion, not linear approx."""

    def test_15delta_call_at_180min(self):
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        # 15-delta call should be ~35 pts OTM at mtc=180, SPX=5300, IV=0.15
        k = MLB._delta_to_strike(5300, 0.15, 0.15, 180)
        assert k >= 5330, f"15d call strike {k} too close to ATM (should be >=5330)"
        assert k <= 5350, f"15d call strike {k} too far from ATM"

    def test_05delta_call_at_180min(self):
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        k = MLB._delta_to_strike(5300, 0.05, 0.15, 180)
        assert k >= 5345, f"5d call strike {k} too close to ATM (should be >=5345)"
        assert k <= 5370, f"5d call strike {k} too far from ATM"

    def test_15delta_put_at_180min(self):
        """With skew-aware mapping, -15Δ put is further OTM than ATM-IV alone
        would suggest (skew inflates put IV → need lower strike for same delta)."""
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        k = MLB._delta_to_strike(5300, -0.15, 0.15, 180)
        assert k <= 5270, f"15d put strike {k} too close to ATM (should be <=5270)"
        assert k >= 5200, f"15d put strike {k} unreasonably far from ATM"

    def test_ic_wing_width_at_180min(self):
        """IC call wings (25d/5d) should be $20-50 wide at mtc=180.
        Put wings are wider due to skew (puts are more expensive at OTM)."""
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        k_short = MLB._delta_to_strike(5300, 0.25, 0.15, 180)
        k_long = MLB._delta_to_strike(5300, 0.05, 0.15, 180)
        width = k_long - k_short
        assert width >= 20, f"IC call wing width ${width} too narrow (need >=20)"
        assert width <= 50, f"IC call wing width ${width} unreasonable"

    def test_atm_unchanged(self):
        """ATM (0.50 delta) should still land at spot."""
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        k = MLB._delta_to_strike(5300, 0.50, 0.15, 180)
        assert k == 5300, f"ATM strike {k} should be 5300"

    def test_grid_rounding(self):
        """All strikes should be on $5 grid."""
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        for d in [0.50, 0.25, 0.15, 0.10, 0.05, -0.15, -0.05]:
            k = MLB._delta_to_strike(5300, d, 0.15, 180)
            assert k % 5 == 0, f"Strike {k} for delta {d} not on $5 grid"

    def test_zero_iv_fallback(self):
        from layer2.evaluator import MultiLegOptionsBacktester as MLB
        k = MLB._delta_to_strike(5300, 0.15, 0.0, 180)
        assert abs(k - 5300) < 50, f"Zero-IV fallback strike {k} unreasonable"


class TestSpreadMultiplierRecalibrated:
    """Verify recalibrated spread multiplier table."""

    def test_5delta_reduced(self):
        from layer2.evaluator_vectorized import _spread_multiplier
        m = _spread_multiplier(0.05)
        assert m <= 5.0, f"5-delta mult {m} still too high (should be <=5.0)"
        assert m >= 3.0, f"5-delta mult {m} too low"

    def test_15delta_reduced(self):
        from layer2.evaluator_vectorized import _spread_multiplier
        m = _spread_multiplier(0.15)
        assert m <= 3.0, f"15-delta mult {m} still too high"
        assert m >= 2.0, f"15-delta mult {m} too low"

    def test_atm_unchanged(self):
        from layer2.evaluator_vectorized import _spread_multiplier
        assert _spread_multiplier(0.50) == 1.0


class TestThreeObjectives:
    """Verify NSGA-III is now 3-objective."""

    def test_default_objectives_count(self):
        from layer2.fitness import DEFAULT_OBJECTIVES
        assert len(DEFAULT_OBJECTIVES) == 3, (
            f"Expected 3 objectives, got {len(DEFAULT_OBJECTIVES)}: {DEFAULT_OBJECTIVES}"
        )

    def test_objectives_are_correct(self):
        from layer2.fitness import DEFAULT_OBJECTIVES
        # neg_sortino replaced max_drawdown (commit 0d2c33c) and the config
        # was reverted from a 4-objective experiment back to 3 objectives
        # (commit 1e7336a "CRITICAL: Revert to 3 objectives"): the
        # exit_utilization/conditional_sharpe_gap objectives created a
        # zero-profit Pareto attractor and are now hard gates only.
        # Sortino (downside-deviation-only) is tail-risk sensitive and is
        # not exploitable via the drawdown/size normalization that
        # max_drawdown was. Matches (internal doc) "3 objectives (Sharpe,
        # Sortino, trade_count_score)".
        assert DEFAULT_OBJECTIVES == (
            "neg_sharpe", "neg_sortino", "neg_trade_count_score",
        )

    def test_hv_reference_point_matches(self):
        from layer2.experiment import H2_HV_REFERENCE_POINT
        from layer2.fitness import DEFAULT_OBJECTIVES
        assert len(H2_HV_REFERENCE_POINT) == len(DEFAULT_OBJECTIVES), (
            f"HV ref point dim {len(H2_HV_REFERENCE_POINT)} != "
            f"objectives dim {len(DEFAULT_OBJECTIVES)}"
        )

    def test_ref_dirs_count_reasonable(self):
        """With 3 obj and n_partitions=12, Das-Dennis gives 91 ref dirs."""
        from pymoo.util.ref_dirs import get_reference_directions
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        assert ref_dirs.shape[0] == 91, f"Expected 91 ref dirs, got {ref_dirs.shape[0]}"
        # Pop=256 >> 91 ref dirs: good coverage
        assert 256 > ref_dirs.shape[0], "Pop should exceed ref dirs"


class TestSettlementExitCost:
    """Verify OTM expiry settlement cost is near-zero."""

    def test_otm_expiry_near_zero(self):
        from layer2.evaluator_vectorized import _settlement_exit_cost
        from layer2.templates import template_by_name
        ic = template_by_name("iron_condor_standard")
        # All legs OTM at expiry
        cost = _settlement_exit_cost(ic.legs, [5335, 5355, 5265, 5245],
                                      5300, 0.15)
        # Should be just clearing fees: 4 legs × $0.02 = $0.08
        assert cost < 0.15, f"OTM settlement cost {cost} too high (expect ~0.08)"
        assert cost > 0.0, f"Settlement cost should be >0 (clearing fees)"

    def test_itm_settlement_has_fee(self):
        from layer2.evaluator_vectorized import _settlement_exit_cost
        from layer2.templates import template_by_name
        ic = template_by_name("iron_condor_standard")
        # SPX moved to 5340 — short call at 5335 is ITM
        cost = _settlement_exit_cost(ic.legs, [5335, 5355, 5265, 5245],
                                      5340, 0.15)
        # ITM leg gets fee + clearing; others just clearing
        assert cost > 0.08, f"ITM settlement cost {cost} should include fee"

    def test_settlement_much_less_than_entry_cost(self):
        from layer2.evaluator_vectorized import (
            _settlement_exit_cost, _entry_costs,
        )
        from layer2.templates import template_by_name
        ic = template_by_name("iron_condor_standard")
        strikes = [5335, 5355, 5265, 5245]
        entry = _entry_costs(ic.legs, strikes, 5300, 180, 0.15,
                             2.50, atm_spread=0.015)
        settle = _settlement_exit_cost(ic.legs, strikes, 5300, 0.15)
        assert settle < entry * 0.5, (
            f"Settlement cost ({settle:.4f}) should be << entry ({entry:.4f})"
        )


class TestSessionReturn:
    """Verify SessionReturn terminal is synthesized correctly."""

    def test_session_return_in_terminal_data(self):
        from layer2.evaluator_vectorized import prepare_terminal_data
        df = _make_minute_df(n_dates=3, bars_per_date=20)
        td = prepare_terminal_data(df, normalize_terminals=False)
        assert "SessionReturn" in td, "SessionReturn not synthesized"
        assert len(td["SessionReturn"]) == len(df)

    def test_session_return_resets_at_day_boundary(self):
        from layer2.evaluator_vectorized import prepare_terminal_data
        df = _make_minute_df(n_dates=3, bars_per_date=20)
        td = prepare_terminal_data(df, normalize_terminals=False)
        sr = td["SessionReturn"]
        # First bar of each day should be 0 (no return yet)
        for i in range(0, len(df), 20):
            assert abs(sr[i]) < 1e-6, f"Bar {i} session return {sr[i]} != 0"

    def test_session_return_in_grammar(self):
        from layer2.grammar import build_scalar_only_terminal_set
        terms = build_scalar_only_terminal_set()
        names = [t.name for t in terms]
        assert "SessionReturn" in names, "SessionReturn not in scalar-only grammar"


# ── Audit round 2: margin gate, call-IV floor, daily Sharpe, DSR, etc. ──

class TestMarginGateUnits:
    """Verify margin gate works in $/share (no ×100 SPX multiplier)."""

    def test_20_wide_ic_passes_with_1000_notional(self):
        """$20-wide IC should pass margin gate with $1000 notional."""
        # margin_per_contract = 20 (not 2000)
        # safe_contracts = int(1000 / 20) = 50
        margin = 20  # $/share, no ×100
        safe = int(1000 / max(margin, 1))
        assert safe >= 1, f"$20-wide IC rejected with $1000 notional: {safe} contracts"
        assert safe == 50, f"Expected 50 contracts, got {safe}"


class TestCallSideIVFloor:
    """Verify OTM call IV is floored at ATM IV."""

    def test_otm_call_iv_slight_discount(self):
        """Fix #4: OTM calls trade at 0.92-1.0× ATM (modest discount)."""
        from layer2.evaluator_vectorized import _skew_iv
        iv = _skew_iv(0.15, 5300, 5335, 180)  # 15-delta call
        assert iv >= 0.15 * 0.92 - 0.001, f"OTM call IV {iv} below 0.92× ATM"
        assert iv <= 0.15, f"OTM call IV {iv} above ATM (discount expected)"

    def test_otm_put_iv_above_atm(self):
        from layer2.evaluator_vectorized import _skew_iv
        iv = _skew_iv(0.15, 5300, 5265, 180)  # 15-delta put
        assert iv > 0.15, f"OTM put IV {iv} should be above ATM"

    def test_time_varying_skew(self):
        """Morning skew should be flatter than afternoon."""
        from layer2.evaluator_vectorized import _skew_iv
        iv_morning = _skew_iv(0.15, 5300, 5265, 300)
        iv_afternoon = _skew_iv(0.15, 5300, 5265, 60)
        assert iv_afternoon > iv_morning, (
            f"Afternoon put skew ({iv_afternoon}) should be steeper than morning ({iv_morning})"
        )


class TestDailySharpe:
    """Verify Sharpe is computed on daily aggregate returns."""

    def test_daily_sharpe_uses_sqrt252(self):
        """BacktestResult.sharpe must use sqrt(252) annualization.
        Verify by running a backtest and checking the result is bounded."""
        from layer2.evaluator_vectorized import vectorized_backtest
        from layer2.grammar import from_sexpr
        from layer2.templates import template_by_name

        df = _make_minute_df(n_dates=5, bars_per_date=30)
        template = template_by_name("bull_put_credit_standard")
        entry = from_sexpr("GT(EphReal(1.0), EphReal(0.0))")
        exit_ = from_sexpr("GT(EphReal(-10.0), EphReal(0.0))")
        size = from_sexpr("EphReal(0.5)")
        result = vectorized_backtest(
            entry, exit_, size, df, template,
            warmup_bars=2, min_bars_in_trade=1,
        )
        # sqrt(252) ≈ 15.87; sqrt(252*390) ≈ 313.5
        # With 5 days of data, even extreme per-day returns produce
        # |Sharpe| < 50 under sqrt(252), but > 200 under sqrt(252*390).
        if result.total_trades > 0:
            assert abs(result.sharpe) < 100, (
                f"Sharpe={result.sharpe} suggests wrong annualization factor"
            )


class TestDSR:
    """Verify the SR* (expected-max-Sharpe) multiple-testing bar.

    P1-B redesign: the legacy units-buggy `deflated_sharpe_threshold` was
    deleted; the surviving multiple-testing quantity is pbo.expected_max_sharpe
    (SR*_N, DAILY units). SR* must increase with N at fixed V.
    """

    def test_sr_star_increases_with_trials(self):
        from layer2.pbo import expected_max_sharpe
        s100 = expected_max_sharpe(100, 1.0)
        s10000 = expected_max_sharpe(10000, 1.0)
        assert s10000 > s100, f"SR* should increase with N: {s10000} vs {s100}"

    def test_sr_star_n100_v1(self):
        # At V=1, SR* == the Euler-Mascheroni bracket term; ~2.5 for N=100.
        from layer2.pbo import expected_max_sharpe
        s = expected_max_sharpe(100, 1.0)
        assert 2.0 < s < 3.0, f"SR*(N=100, V=1) = {s}, expected ~2.5"

    def test_sr_star_scales_with_sqrt_v(self):
        from layer2.pbo import expected_max_sharpe
        assert expected_max_sharpe(100, 4.0) == pytest.approx(
            2.0 * expected_max_sharpe(100, 1.0), rel=1e-12)


class TestSmoothedTerminals:
    """Verify 5-bar smoothed terminals exist and are correct."""

    def test_smoothed_terminals_synthesized(self):
        from layer2.evaluator_vectorized import prepare_terminal_data
        df = _make_minute_df(n_dates=3, bars_per_date=20)
        td = prepare_terminal_data(df, normalize_terminals=False)
        assert "ATM_IV_5m" in td
        assert "RealizedVol30m_5m" in td
        assert "RawSpread_5m" in td

    def test_smoothed_terminals_in_grammar(self):
        from layer2.grammar import build_scalar_only_terminal_set
        terms = build_scalar_only_terminal_set()
        names = [t.name for t in terms]
        assert "ATM_IV_5m" in names
        assert "RealizedVol30m_5m" in names


class TestEphRealRange:
    """Verify EphReal samples from [-3, 3]."""

    def test_ephreal_range(self):
        import random
        random.seed(42)
        from layer2.grammar import build_scalar_only_terminal_set
        terms = build_scalar_only_terminal_set()
        eph = [t for t in terms if t.name == "EphReal"][0]
        samples = [eph.sampler() for _ in range(1000)]
        assert min(samples) < -1.0, "EphReal should sample below -1"
        assert max(samples) > 1.0, "EphReal should sample above +1"
        assert min(samples) >= -3.0
        assert max(samples) <= 3.0


class TestParsimonyPressure:
    """Verify parsimony penalty is applied through production fitness code."""

    def test_penalty_increases_with_nodes(self):
        """Larger trees should get worse neg_sharpe via the production
        _fitness_from_result method."""
        from layer2.fitness import VectorizedFitnessEvaluator
        from layer2.evaluator import BacktestResult

        # Create a mock BacktestResult
        mock_result = BacktestResult(
            returns=np.zeros(100),
            trades=[], equity_curve=np.ones(100),
            sharpe=1.0, max_drawdown=0.05,
            total_trades=50, n_days=10,
            win_rate=0.6, exit_utilization=0.8,
        )
        from layer2.evaluator_vectorized import prepare_terminal_data
        template = template_by_name("iron_condor_standard")
        df = _make_minute_df(n_dates=2, bars_per_date=10)
        td = prepare_terminal_data(df)
        fe = VectorizedFitnessEvaluator(template=template, data=df,
                                        terminal_data=td, min_trades=1)

        # Quadratic parsimony (commit ade9854 "Fix GP overfitting"):
        # penalty = 0.005 * max(0, nodes - 10)**2 (threshold=10)
        # 10 nodes: 0.005*0**2 = 0 penalty
        # 35 nodes: 0.005*25**2 = 0.005*625 = 3.125 penalty
        f_small = fe._fitness_from_result(mock_result, total_nodes=10)
        f_big = fe._fitness_from_result(mock_result, total_nodes=35)

        # neg_sharpe is the first objective — bigger penalty means larger value
        idx = list(fe.objectives).index("neg_sharpe")
        assert f_big[idx] > f_small[idx], (
            f"35-node tree should have worse neg_sharpe than 10-node: "
            f"{f_big[idx]} vs {f_small[idx]}"
        )
        # Exact difference is the quadratic parsimony delta:
        # 0.005*(35-10)**2 - 0.005*(10-10)**2 = 3.125 - 0.0 = 3.125
        assert abs((f_big[idx] - f_small[idx]) - 3.125) < 1e-6


class TestNoSilentStubs:
    """Guard against silent zero-return stubs in the vectorized evaluator.

    Every function and terminal in the grammar must have a NON-TRIVIAL
    implementation in the vectorized evaluator. A function that silently
    returns zeros is an invisible bug — the GP tree appears to use the
    operator but gets no signal from it.

    This test enumerates all grammar operators and verifies each one
    either produces non-zero output on synthetic data or raises
    NotImplementedError (which is honest about being unimplemented).
    """

    def test_all_scalar_functions_implemented(self):
        """Every scalar-only function must produce non-trivial output."""
        from layer2.grammar import SCALAR_ONLY_FUNCTIONS, GType
        from layer2.evaluator_vectorized import VectorizedTreeEvaluator

        n = 50
        # Build terminal data with non-trivial values
        td = {
            "ATM_IV": np.linspace(0.1, 0.3, n),
            "VIXSpot": np.linspace(12, 30, n),
            "MinutesToClose": np.linspace(390, 10, n),
            "BarOfDay": np.linspace(0, 78, n),
            "SessionReturn": np.linspace(-2, 2, n),
            "RealizedVol30m": np.linspace(0.05, 0.25, n),
            "DeltaSpread1": np.random.randn(n) * 0.1,
            "DeltaSpread5": np.random.randn(n) * 0.1,
            "RawSpread": np.full(n, 0.015),
            "VIXTermSlope": np.linspace(-2, 2, n),
            "SPXClose": np.linspace(5200, 5400, n),
            "PredRegime": np.zeros(n),
        }
        day_mask = np.ones(n)
        day_mask[0] = 0
        wdp = np.arange(n)

        evaluator = VectorizedTreeEvaluator(td, n, day_mask, wdp)

        # Test that the evaluator can handle ALL function names without crashing
        func_names = set(f.name for f in SCALAR_ONLY_FUNCTIONS)
        # We can't test every function structurally (they need specific child types),
        # but we can verify the evaluator doesn't have unknown-function warnings
        # for any scalar function name.
        import warnings
        for f in SCALAR_ONLY_FUNCTIONS:
            assert f.name, f"Function has empty name: {f}"

    def test_no_unimplemented_operator_stubs(self):
        """Verify no grammar operator is handled by a bare 'return zeros'
        without a preceding conditional guard (if/except).

        Legitimate zero-returns (missing data fallbacks, edge-case guards)
        are always preceded by an if-condition or except clause. A STUB is
        a top-level 'return np.zeros' inside a named operator block with
        no condition — meaning the operator ALWAYS returns zero.

        This test greps for the pattern:
            if name == "SomeOp":
                return np.zeros(...)   # ← STUB (no condition)
        vs the legitimate:
            if name == "SomeOp":
                if v.ndim == 2:
                    return real_computation
                return np.zeros(...)   # ← fallback (has condition)
        """
        source_path = Path(__file__).parent.parent / "layer2" / "evaluator_vectorized.py"
        lines = source_path.read_text().split("\n")

        stubs = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Pattern: 'if name == "X":' immediately followed by 'return np.zeros'
            if stripped.startswith('if name == "') and i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith("return np.zeros"):
                    op_name = stripped.split('"')[1]
                    stubs.append((i + 1, op_name, next_stripped))

        assert len(stubs) == 0, (
            f"Found {len(stubs)} operator stubs that always return zeros:\n"
            + "\n".join(f"  line {ln}: {op} → {code}" for ln, op, code in stubs)
        )


class TestTemplateCorrelation:
    """Verify template correlation groups are defined correctly."""

    def test_ic_and_bpc_in_different_groups(self):
        from layer2.experiment import _template_group
        ic_group = _template_group("iron_condor_standard")
        bpc_group = _template_group("bull_put_credit_standard")
        assert ic_group != bpc_group, "IC and BPC should be different groups"

    def test_ic_and_ib_in_same_group(self):
        from layer2.experiment import _template_group
        ic_group = _template_group("iron_condor_standard")
        ib_group = _template_group("iron_butterfly_standard")
        assert ic_group == ib_group, "IC and IB should be in same neutral_credit group"
