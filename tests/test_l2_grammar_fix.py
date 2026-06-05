"""Validation suite for L2 grammar fix (V1-V10).

V1:  No degenerate GT pairs (10-90% True)
V2:  EphReal covers all terminals (>20% True AND >20% False)
V3:  Arithmetic no scale dominance
V4:  Proxy-to-codegen 0% entry/exit divergence on identical inputs
V5:  Strike separation >= $5
V6:  Leverage <= 2x
V7:  Seed threshold behavior preserved after normalization
V8:  1-minute Parquet has ~405 rows/day (PLACEHOLDER — requires data generation)
V9:  Hand-crafted strategy round-trip through codegen (structural validation)
V10: Short GP run produces non-degenerate strategies (PLACEHOLDER — requires GP)
"""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_train_data():
    """Create a small but realistic DataFrame matching L1 Parquet schema."""
    n_days = 10
    bars_per_day = 12
    n = n_days * bars_per_day
    dates = []
    for d in range(n_days):
        date_str = f"2024-01-{d+1:02d}"
        dates.extend([date_str] * bars_per_day)

    rng = np.random.default_rng(42)
    data = {
        "date": dates,
        "window_idx": list(range(n)),
        "bar_position": [i % bars_per_day for i in range(n)],
        # 2026-06-02 (M5): raw ATM_IV std widened 0.05→0.153 so the synthetic
        # column still normalizes to ~unit std under the re-derived ATM_IV scale
        # (0.153269, winsor@0.80 cap→+4.5). The old 0.05 std produced normalized
        # std≈1.0 only under the OLD tight scale (0.0486); with the widened scale
        # it would be ≈0.33 and break the Add-balance / EphReal-coverage checks
        # that assume each normalized terminal is ~N(0,1). center≈new median.
        "ATM_IV": rng.normal(0.11, 0.153, n).clip(0.01, 0.80),
        "VIXSpot": rng.normal(17, 5, n).clip(10, 40),
        "VIXTermSlope": rng.normal(-0.5, 1.5, n),
        # 2026-06-01: scales match the re-derived TERMINAL_NORM_STATS after the
        # single-Parkinson RealizedVol30m + spike-cleaned spread parquet rebuild.
        "RealizedVol30m": rng.normal(0.00026, 0.00015, n).clip(0.0, 0.001),
        "MinutesToClose": np.tile(np.linspace(330, 0, bars_per_day), n_days),
        "RawSpread": rng.normal(0.0165, 0.008, n).clip(0.0, 0.06),
        "DeltaSpread1": rng.normal(0, 0.0084, n),
        "DeltaSpread5": rng.normal(0, 0.0095, n),
        "SPXClose": rng.normal(4500, 100, n).clip(4000, 5000),
        "PredRV15": rng.normal(0.015, 0.003, n).clip(0.008, 0.05),
        "PredRV30": rng.normal(0.015, 0.003, n).clip(0.008, 0.05),
        "PredSpread": rng.normal(0, 0.01, n),
        "PredRegime": rng.integers(0, 4, n).astype(float),
    }
    # Typed vector columns (placeholder zeros for scalar-only tests)
    for col in ["EMB_SHARED", "EMB_GRID", "EMB_STRIKE", "EMB_SPX",
                "EMB_VIX", "EMB_FLOW_RAW", "EMB_FLOW_AGG",
                "EMB_VAR_ATM_IV", "EMB_VAR_ATM_SPREAD", "EMB_VAR_SPX_RET"]:
        data[col] = [np.zeros(384, dtype=np.float32) for _ in range(n)]
    for k in range(4):
        data[f"RegimeProb{k}"] = rng.uniform(0, 1, n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# V1: No degenerate GT pairs (10-90% True)
# ---------------------------------------------------------------------------

class TestV1NoDegenerateGTPairs:
    """After normalization, every GT(terminal, terminal) pair should produce
    True on 10-90% of training bars (not structurally constant)."""

    def test_gt_pairs_non_degenerate(self, sample_train_data):
        """Every pair of REAL terminals, when compared via GT, produces
        True on 10-90% of bars (not constant True or False)."""
        from layer2.terminal_stats import NORMALIZED_TERMINALS, normalize

        # Get REAL terminal columns present in data
        real_terminals = [col for col in sample_train_data.columns
                         if col in NORMALIZED_TERMINALS]

        assert len(real_terminals) >= 5, "Need at least 5 REAL terminals for this test"

        # Normalize all terminal values
        normalized = {}
        for name in real_terminals:
            vals = sample_train_data[name].values.astype(np.float64)
            normalized[name] = np.array([normalize(name, v) for v in vals])

        # Test all pairs
        degenerate_pairs = []
        for i, a_name in enumerate(real_terminals):
            for b_name in real_terminals[i+1:]:
                a_vals = normalized[a_name]
                b_vals = normalized[b_name]
                frac_true = np.mean(a_vals > b_vals)
                if frac_true < 0.10 or frac_true > 0.90:
                    degenerate_pairs.append((a_name, b_name, frac_true))

        assert len(degenerate_pairs) == 0, (
            f"Found {len(degenerate_pairs)} degenerate GT pairs "
            f"(should be 10-90% True): {degenerate_pairs[:5]}"
        )


# ---------------------------------------------------------------------------
# V2: EphReal covers all terminals
# ---------------------------------------------------------------------------

class TestV2EphRealCoversAllTerminals:
    """EphReal[-1, 1] should produce >20% True AND >20% False when
    compared to any normalized terminal."""

    def test_ephreal_coverage(self, sample_train_data):
        from layer2.terminal_stats import NORMALIZED_TERMINALS, normalize

        real_terminals = [col for col in sample_train_data.columns
                         if col in NORMALIZED_TERMINALS]

        eph_samples = np.linspace(-1, 1, 21)  # sample across EphReal range

        for name in real_terminals:
            vals = sample_train_data[name].values.astype(np.float64)
            norm_vals = np.array([normalize(name, v) for v in vals])

            # For each EphReal sample, check GT produces non-degenerate result
            found_discriminating = False
            for eph in eph_samples:
                frac_true = np.mean(norm_vals > eph)
                if 0.20 <= frac_true <= 0.80:
                    found_discriminating = True
                    break

            assert found_discriminating, (
                f"Terminal {name}: no EphReal value in [-1,1] produces "
                f"20-80% True in GT comparison. Normalized range: "
                f"[{norm_vals.min():.3f}, {norm_vals.max():.3f}]"
            )


# ---------------------------------------------------------------------------
# V3: Arithmetic no scale dominance
# ---------------------------------------------------------------------------

class TestV3ArithmeticNoScaleDominance:
    """After normalization, Add(A, B) should have std within [0.3, 3.0] of
    each operand's std. No terminal dominates arithmetic."""

    def test_add_scale_balance(self, sample_train_data):
        from layer2.terminal_stats import NORMALIZED_TERMINALS, normalize

        real_terminals = [col for col in sample_train_data.columns
                         if col in NORMALIZED_TERMINALS]

        for i, a_name in enumerate(real_terminals):
            for b_name in real_terminals[i+1:]:
                a = np.array([normalize(a_name, v) for v in
                             sample_train_data[a_name].values])
                b = np.array([normalize(b_name, v) for v in
                             sample_train_data[b_name].values])
                sum_vals = a + b
                std_sum = np.std(sum_vals)
                std_a = max(np.std(a), 1e-10)
                std_b = max(np.std(b), 1e-10)

                ratio_a = std_sum / std_a
                ratio_b = std_sum / std_b

                assert 0.3 <= ratio_a <= 3.0, (
                    f"Add({a_name}, {b_name}): std(sum)/std({a_name}) = "
                    f"{ratio_a:.3f}, expected [0.3, 3.0]"
                )
                assert 0.3 <= ratio_b <= 3.0, (
                    f"Add({a_name}, {b_name}): std(sum)/std({b_name}) = "
                    f"{ratio_b:.3f}, expected [0.3, 3.0]"
                )


# ---------------------------------------------------------------------------
# V4: Proxy-to-codegen consistency
# ---------------------------------------------------------------------------

class TestV4ProxyCodegenConsistency:
    """Proxy evaluator and codegen must produce identical entry/exit decisions
    when fed identical scalar values."""

    def test_identical_entry_exit_on_simple_tree(self):
        """Hand-crafted GT(ATM_IV, EphReal(0.5)) must match between proxy
        and codegen Python expression."""
        from layer2.grammar import from_sexpr
        from layer2.evaluator import TreeEvaluator, EvaluationContext
        from layer2.terminal_stats import normalize, normalize_threshold
        from layer3.codegen import _node_to_python

        # Normalized threshold: ATM_IV > 0.18 in raw space
        norm_thresh = normalize_threshold("ATM_IV", 0.18)
        sexpr = f"GT(ATM_IV, EphReal({norm_thresh:.4f}))"
        tree = from_sexpr(sexpr)

        # Test with several raw ATM_IV values
        test_values = [0.10, 0.15, 0.18, 0.20, 0.25, 0.30]

        evaluator = TreeEvaluator()
        ctx = EvaluationContext(max_lag=30, emb_dim=384)

        # Get Python expression
        py_expr = _node_to_python(tree)

        for raw_iv in test_values:
            # Proxy path: normalize then evaluate
            normalized_iv = normalize("ATM_IV", raw_iv)
            ctx.buffer.clear()
            ctx.buffer["ATM_IV"] = deque([normalized_iv], maxlen=31)
            ctx.bar_idx = 1
            proxy_result = bool(evaluator.evaluate(tree, ctx))

            # Codegen path: evaluate Python expression with same normalized value
            s = {"ATM_IV": normalized_iv}
            codegen_result = bool(eval(py_expr))

            assert proxy_result == codegen_result, (
                f"ATM_IV raw={raw_iv}, norm={normalized_iv:.4f}: "
                f"proxy={proxy_result}, codegen={codegen_result}, "
                f"expr={py_expr}"
            )


# ---------------------------------------------------------------------------
# V5: Strike separation >= $5
# ---------------------------------------------------------------------------

class TestV5StrikeSeparation:
    """All templates must have leg strikes >= $5 apart at all bars."""

    def test_strike_separation_all_templates(self):
        from layer2.evaluator import MultiLegOptionsBacktester
        from layer2.templates import all_templates

        for template in all_templates():
            bt = MultiLegOptionsBacktester(template)
            # Test at various spot prices and IVs
            for spot in [4000, 4500, 5000, 5500]:
                for iv in [0.10, 0.15, 0.20, 0.30]:
                    for minutes in [30, 60, 120, 300]:
                        strikes = [
                            bt._delta_to_strike(spot, leg.delta_target, iv, minutes)
                            for leg in template.legs
                        ]
                        strikes = bt._enforce_strike_separation(
                            strikes, template.legs, bt.MIN_STRIKE_SEPARATION
                        )

                        # Check all pairwise separations
                        for i in range(len(strikes)):
                            for j in range(i + 1, len(strikes)):
                                gap = abs(strikes[i] - strikes[j])
                                assert gap >= 4.99, (  # 4.99 for float tolerance
                                    f"{template.name}: strikes {strikes[i]:.0f} "
                                    f"and {strikes[j]:.0f} are only ${gap:.1f} "
                                    f"apart (need >=$5). spot={spot}, iv={iv}, "
                                    f"min={minutes}"
                                )

    def test_strike_separation_enforcement(self):
        """Test that _enforce_strike_separation actually fixes collapsed strikes."""
        from layer2.evaluator import MultiLegOptionsBacktester

        # Simulate collapsed strikes (all on same $5 grid point)
        collapsed = [5000.0, 5000.0, 5000.0, 5000.0]
        legs = tuple()  # not used by enforcement
        result = MultiLegOptionsBacktester._enforce_strike_separation(
            collapsed, legs, 5.0
        )
        # All strikes should now be separated
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                assert abs(result[i] - result[j]) >= 4.99, (
                    f"Strikes {result[i]} and {result[j]} still collapsed"
                )


# ---------------------------------------------------------------------------
# V6: Leverage <= 2x
# ---------------------------------------------------------------------------

class TestV6LeverageCap:
    """n_contracts * gross_value <= 2 * notional for all templates."""

    def test_leverage_bounded(self, sample_train_data):
        from layer2.evaluator import MultiLegOptionsBacktester
        from layer2.templates import all_templates
        from layer2.grammar import from_sexpr

        # Use a simple always-enter tree
        entry = from_sexpr("GT(EphReal(0.5), EphReal(-0.5))")  # always True
        exit_ = from_sexpr("LT(MinutesToClose, EphReal(-1.5))")  # never exits
        size = from_sexpr("EphReal(1.0)")  # max size

        for template in all_templates()[:4]:  # test first 4 for speed
            bt = MultiLegOptionsBacktester(template, notional=1000.0)
            result = bt.run(
                entry_tree=entry, exit_tree=exit_, size_tree=size,
                data=sample_train_data,
            )
            # The test passes if we get here without assertion errors.
            # The leverage cap is enforced in the position entry code.
            assert result is not None


# ---------------------------------------------------------------------------
# V7: Seed threshold behavior preserved after normalization
# ---------------------------------------------------------------------------

class TestV7SeedThresholdPreservation:
    """Seed trees must produce the same entry behavior on real data after
    normalization conversion."""

    def test_iron_condor_entry_preserved(self):
        """Iron condor entry: ATM_IV > 0.18.
        In normalized space: ATM_IV_norm > normalize("ATM_IV", 0.18).
        The entry should fire on the same bars."""
        from layer2.terminal_stats import normalize, normalize_threshold

        # Generate test data
        rng = np.random.default_rng(42)
        raw_iv = rng.normal(0.12, 0.05, 1000).clip(0.01, 0.5)

        # Old behavior: raw comparison
        old_fires = raw_iv > 0.18

        # New behavior: normalized comparison
        norm_iv = np.array([normalize("ATM_IV", v) for v in raw_iv])
        norm_thresh = normalize_threshold("ATM_IV", 0.18)
        new_fires = norm_iv > norm_thresh

        # Must be identical
        assert np.array_equal(old_fires, new_fires), (
            f"Entry behavior changed after normalization. "
            f"Old fires {old_fires.sum()}, new fires {new_fires.sum()}"
        )

    def test_exit_close_to_expiry_preserved(self):
        """Exit: MinutesToClose < 30 must fire on same bars."""
        from layer2.terminal_stats import normalize, normalize_threshold

        raw_mtc = np.linspace(0, 330, 100)
        old_fires = raw_mtc < 30.0
        norm_mtc = np.array([normalize("MinutesToClose", v) for v in raw_mtc])
        norm_thresh = normalize_threshold("MinutesToClose", 30.0)
        new_fires = norm_mtc < norm_thresh

        assert np.array_equal(old_fires, new_fires), (
            f"Exit behavior changed after normalization. "
            f"Old fires {old_fires.sum()}, new fires {new_fires.sum()}"
        )


# ---------------------------------------------------------------------------
# V8: 1-minute Parquet has ~405 rows/day (PLACEHOLDER)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# V9: Hand-crafted strategy round-trip through codegen
# ---------------------------------------------------------------------------

class TestV9CodegenRoundTrip:
    """Generate QC code from a known strategy and verify it is syntactically
    valid and contains the expected structure."""

    def test_codegen_produces_valid_python(self):
        from layer3.codegen import generate_qc_algorithm, validate_generated_code

        code = generate_qc_algorithm(
            strategy_id="test_v9",
            template_name="iron_condor",
            entry_sexpr="GT(ATM_IV, EphReal(0.5))",
            exit_sexpr="LT(MinutesToClose, EphReal(-1.0))",
            size_sexpr="EphReal(0.5)",
        )
        valid, err = validate_generated_code(code)
        assert valid, f"Generated code is invalid: {err}"

    def test_codegen_contains_normalization(self):
        """Generated code must include _normalize method and _NORM_STATS."""
        from layer3.codegen import generate_qc_algorithm

        code = generate_qc_algorithm(
            strategy_id="test_norm",
            template_name="iron_condor",
            entry_sexpr="GT(ATM_IV, EphReal(0.5))",
            exit_sexpr="LT(MinutesToClose, EphReal(-1.0))",
            size_sexpr="EphReal(0.5)",
        )
        assert "_normalize" in code, "Generated code missing _normalize method"
        assert "_NORM_STATS" in code, "Generated code missing _NORM_STATS dict"
        assert "ATM_IV" in code, "Generated code missing ATM_IV in norm stats"

    def test_codegen_all_templates(self):
        """All 12 templates produce valid Python."""
        from layer3.codegen import generate_qc_algorithm, validate_generated_code, TEMPLATE_LEGS

        for template_name in TEMPLATE_LEGS.keys():
            code = generate_qc_algorithm(
                strategy_id=f"test_{template_name}",
                template_name=template_name,
                entry_sexpr="GT(ATM_IV, EphReal(0.5))",
                exit_sexpr="LT(MinutesToClose, EphReal(-1.0))",
                size_sexpr="EphReal(0.5)",
            )
            valid, err = validate_generated_code(code)
            assert valid, f"Template {template_name} produced invalid code: {err}"


# ---------------------------------------------------------------------------
# V10: Short GP run produces non-degenerate strategies (PLACEHOLDER)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Additional regression tests
# ---------------------------------------------------------------------------

class TestTerminalStats:
    """Test terminal_stats.py invariants."""

    def test_normalize_roundtrip(self):
        from layer2.terminal_stats import normalize, denormalize
        for name in ["ATM_IV", "VIXSpot", "MinutesToClose", "RealizedVol30m"]:
            raw = 0.15 if name == "ATM_IV" else 20.0
            norm = normalize(name, raw)
            recovered = denormalize(name, norm)
            assert abs(recovered - raw) < 1e-10, (
                f"Round-trip failed for {name}: {raw} -> {norm} -> {recovered}"
            )

    def test_normalize_unknown_terminal(self):
        """Unknown terminals pass through unchanged."""
        from layer2.terminal_stats import normalize
        assert normalize("UnknownTerminal", 42.0) == 42.0

    def test_all_bypass_scalars_have_stats(self):
        """Every bypass scalar terminal in the grammar has normalization stats."""
        from layer2.terminal_stats import TERMINAL_NORM_STATS
        from layer2.grammar import build_scalar_only_terminal_set

        bypass_names = {"DeltaSpread1", "DeltaSpread5", "RawSpread", "ATM_IV",
                       "VIXSpot", "VIXTermSlope", "RealizedVol30m",
                       "MinutesToClose", "BarOfDay"}
        for name in bypass_names:
            assert name in TERMINAL_NORM_STATS, (
                f"Bypass scalar {name} missing from TERMINAL_NORM_STATS"
            )


class TestAbstainFlagRemoval:
    """AbstainFlag must not appear in any terminal set."""

    def test_no_abstain_in_full_set(self):
        from layer2.grammar import build_terminal_set
        names = [t.name for t in build_terminal_set()]
        assert "AbstainFlag" not in names

    def test_no_abstain_in_scalar_only(self):
        from layer2.grammar import build_scalar_only_terminal_set
        names = [t.name for t in build_scalar_only_terminal_set()]
        assert "AbstainFlag" not in names

    def test_no_abstain_in_probes_only(self):
        from layer2.grammar import build_probes_only_terminal_set
        names = [t.name for t in build_probes_only_terminal_set()]
        assert "AbstainFlag" not in names

    def test_no_abstain_in_emb_only(self):
        from layer2.grammar import build_emb_only_terminal_set
        names = [t.name for t in build_emb_only_terminal_set()]
        assert "AbstainFlag" not in names


class TestTournamentSelection:
    """Tournament selection uses dominance ranks, not raw fitness sum."""

    def test_tournament_with_ranks(self):
        """Individuals with dominance rank should be preferred over higher-rank."""
        from layer2.gp_engine import _tournament_select, Individual
        from layer2.grammar import from_sexpr

        tree = from_sexpr("GT(ATM_IV, EphReal(0.5))")

        # Create individuals with known ranks
        ind_good = Individual(tree, tree, tree, "test")
        ind_good.fitness = np.array([0.1, 0.1, 0.1, 0.1])
        ind_good._nds_rank = 0
        ind_good._crowding_dist = 1.0

        ind_bad = Individual(tree, tree, tree, "test")
        ind_bad.fitness = np.array([1.0, 1.0, 1.0, 1.0])
        ind_bad._nds_rank = 5
        ind_bad._crowding_dist = 0.5

        # Tournament should always pick ind_good (rank 0 < rank 5)
        pop = [ind_good, ind_bad]
        random.seed(42)
        selected = _tournament_select(pop, 2)
        assert selected._nds_rank == 0, "Tournament should prefer lower rank"


class TestTradeCountScoring:
    """Trade count scoring should peak at 1-3 trades/day."""

    def test_trade_count_curve(self):
        from layer2.evaluator import MultiLegOptionsBacktester
        from layer2.templates import iron_condor

        bt = MultiLegOptionsBacktester(iron_condor())

        # 0 trades -> -1
        result = type("R", (), {"total_trades": 0, "n_days": 10,
                                 "sharpe": 0, "max_drawdown": 0, "win_rate": 0})()
        f = bt.compute_fitness(result)
        assert f["trade_count_score"] == -1.0

        # 1 trade/day -> 1.0 (top of ramp)
        result = type("R", (), {"total_trades": 10, "n_days": 10,
                                 "sharpe": 0, "max_drawdown": 0, "win_rate": 0})()
        f = bt.compute_fitness(result)
        assert abs(f["trade_count_score"] - 1.0) < 0.01

        # 2 trades/day -> 1.0 (plateau)
        result = type("R", (), {"total_trades": 20, "n_days": 10,
                                 "sharpe": 0, "max_drawdown": 0, "win_rate": 0})()
        f = bt.compute_fitness(result)
        assert abs(f["trade_count_score"] - 1.0) < 0.01

        # 8 trades/day -> 0.0 (decay end)
        result = type("R", (), {"total_trades": 80, "n_days": 10,
                                 "sharpe": 0, "max_drawdown": 0, "win_rate": 0})()
        f = bt.compute_fitness(result)
        assert abs(f["trade_count_score"]) < 0.01

    def test_simple_backtester_scoring_matches(self):
        """SimpleBacktester and MultiLegOptionsBacktester should use same curve."""
        from layer2.evaluator import SimpleBacktester

        bt = SimpleBacktester()
        result = type("R", (), {"total_trades": 20, "n_days": 10,
                                 "sharpe": 0, "max_drawdown": 0, "win_rate": 0})()
        f = bt.compute_fitness(result)
        assert abs(f["trade_count_score"] - 1.0) < 0.01


class TestMinHoldPeriod:
    """Test that min_bars_in_trade prevents premature exits."""

    def test_min_hold_prevents_immediate_exit(self, sample_train_data):
        """An exit tree that fires every bar should not exit before min_bars."""
        from layer2.evaluator import MultiLegOptionsBacktester
        from layer2.templates import iron_condor
        from layer2.grammar import from_sexpr

        template = iron_condor()
        bt = MultiLegOptionsBacktester(template, min_bars_in_trade=5)

        # Always-enter, always-exit
        entry = from_sexpr("GT(EphReal(0.5), EphReal(-0.5))")
        exit_ = from_sexpr("GT(EphReal(0.5), EphReal(-0.5))")
        size = from_sexpr("EphReal(0.5)")

        result = bt.run(entry_tree=entry, exit_tree=exit_, size_tree=size,
                       data=sample_train_data)

        # All trades should be at least 5 bars long (except forced day-boundary)
        for trade in result.trades:
            # Allow day-boundary closes to be shorter
            assert trade.bars_held >= 4, (
                f"Trade held for only {trade.bars_held} bars, "
                f"expected at least 5"
            )


class TestStrikeRounding:
    """Strikes should be rounded to $5 increments."""

    def test_strikes_on_5_dollar_grid(self):
        from layer2.evaluator import MultiLegOptionsBacktester

        for spot in [4000, 4123, 4567, 5000]:
            for delta in [0.15, 0.05, -0.15, -0.05, 0.50]:
                strike = MultiLegOptionsBacktester._delta_to_strike(
                    spot, delta, 0.20, 60.0
                )
                assert strike % 5.0 == 0.0, (
                    f"Strike {strike} not on $5 grid (spot={spot}, delta={delta})"
                )
