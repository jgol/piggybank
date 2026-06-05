"""Tests for layer3/codegen.py and layer2/grammar.from_sexpr().

Tests:
  1. from_sexpr round-trip on all Condition A strategies
  2. codegen produces syntactically valid Python for all templates
  3. Tree-walker correctly maps each operator
  4. Template leg generation for all 8 templates
  5. Edge cases: deeply nested trees, ephemeral constants, cross operators
"""
import ast
import json
import glob
import math
import pytest
from pathlib import Path

from layer2.grammar import from_sexpr, to_str, FuncNode, TermNode, GType
from layer3.codegen import (
    generate_qc_algorithm,
    validate_generated_code,
    _node_to_python,
    TEMPLATE_LEGS,
)


# ---------------------------------------------------------------------------
# from_sexpr tests
# ---------------------------------------------------------------------------

class TestFromSexpr:
    """Test s-expression parsing and round-trip fidelity."""

    def test_bare_terminal(self):
        node = from_sexpr("ATM_IV")
        assert isinstance(node, TermNode)
        assert node.name == "ATM_IV"
        assert node.ret_type == GType.REAL

    def test_ephreal_with_value(self):
        node = from_sexpr("EphReal(0.18)")
        assert isinstance(node, TermNode)
        assert node.name == "EphReal"
        assert node.value == pytest.approx(0.18)

    def test_ephreal_negative(self):
        node = from_sexpr("EphReal(-0.5)")
        assert isinstance(node, TermNode)
        assert node.value == pytest.approx(-0.5)

    def test_ephint(self):
        node = from_sexpr("EphInt(5)")
        assert isinstance(node, TermNode)
        assert node.value == 5
        assert node.ret_type == GType.INT

    def test_regime_literal(self):
        node = from_sexpr("LOW_VOL")
        assert isinstance(node, TermNode)
        assert node.ret_type == GType.REGIME

    def test_side_literal(self):
        node = from_sexpr("CALL")
        assert isinstance(node, TermNode)
        assert node.ret_type == GType.SIDE

    def test_simple_function(self):
        node = from_sexpr("GT(ATM_IV, EphReal(0.18))")
        assert isinstance(node, FuncNode)
        assert node.name == "GT"
        assert len(node.children) == 2
        assert node.children[0].name == "ATM_IV"
        assert node.children[1].value == pytest.approx(0.18)

    def test_nested_function(self):
        node = from_sexpr("GT(Add(ATM_IV, VIXSpot), EphReal(0.5))")
        assert isinstance(node, FuncNode)
        assert node.name == "GT"
        assert node.children[0].name == "Add"
        assert node.children[0].children[0].name == "ATM_IV"

    def test_ifthenelse(self):
        node = from_sexpr("IfThenElse(GT(ATM_IV, EphReal(0.2)), VIXSpot, RealizedVol30m)")
        assert node.name == "IfThenElse"
        assert len(node.children) == 3

    def test_lag_delta(self):
        node = from_sexpr("Lag(ATM_IV, EphInt(5))")
        assert node.name == "Lag"
        assert node.children[1].value == 5

    def test_cross_above(self):
        node = from_sexpr("CrossAbove(ATM_IV, VIXSpot)")
        assert node.name == "CrossAbove"
        assert len(node.children) == 2

    def test_roundtrip_simple(self):
        s = "GT(ATM_IV, EphReal(0.18))"
        assert to_str(from_sexpr(s)) == s

    def test_roundtrip_complex(self):
        s = "IfThenElse(AND(GT(ATM_IV, EphReal(0.2)), LT(VIXSpot, EphReal(20.0))), Add(RawSpread, DeltaSpread1), Sqrt(RealizedVol30m))"
        assert to_str(from_sexpr(s)) == s

    def test_roundtrip_negative_ephreal(self):
        s = "GT(Delta(ATM_IV, EphInt(5)), EphReal(-0.165))"
        assert to_str(from_sexpr(s)) == s

    @pytest.mark.skipif(
        not Path("results/gp_A_local_20260505").exists(),
        reason="Condition A results not available",
    )
    def test_roundtrip_all_condition_a(self):
        """Round-trip ALL trees from Condition A output."""
        total = 0
        for path in glob.glob("results/gp_A_local_20260505/strategies_*.jsonl"):
            with open(path) as f:
                for line in f:
                    strat = json.loads(line)
                    for key in ["entry_tree", "exit_tree", "size_tree"]:
                        s = strat[key]
                        assert to_str(from_sexpr(s)) == s, f"Round-trip failed: {s[:80]}..."
                        total += 1
        assert total > 5000, f"Expected >5000 trees, got {total}"

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            from_sexpr("BOGUS_TERMINAL")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            from_sexpr("")

    def test_unbalanced_parens_raises(self):
        with pytest.raises((ValueError, IndexError)):
            from_sexpr("GT(ATM_IV, EphReal(0.18)")


# ---------------------------------------------------------------------------
# Tree-to-Python expression tests
# ---------------------------------------------------------------------------

class TestNodeToPython:
    """Test the tree-walker produces correct Python expressions."""

    def test_gt(self):
        node = from_sexpr("GT(ATM_IV, EphReal(0.18))")
        py = _node_to_python(node)
        assert 'ATM_IV' in py
        assert "0.18" in py
        assert ">" in py

    def test_div_analytic_quotient(self):
        node = from_sexpr("Div(ATM_IV, VIXSpot)")
        py = _node_to_python(node)
        assert "math.sqrt(1.0 +" in py

    def test_sqrt_protected(self):
        node = from_sexpr("Sqrt(ATM_IV)")
        py = _node_to_python(node)
        # Protected sqrt = sqrt(|x|), matching BOTH proxy evaluators
        # (evaluator.py:716, evaluator_vectorized.py:331). P0-1: previously this
        # asserted sqrt(max(0,x)), which silently zeroed negatives and diverged
        # from the proxy on ~half of bars; the semantic-equivalence oracle
        # (test_codegen_semantic::test_sqrt) is the authority.
        assert "math.sqrt(abs(" in py

    def test_lag_terminal(self):
        node = from_sexpr("Lag(ATM_IV, EphInt(5))")
        py = _node_to_python(node)
        assert 'self._lag("ATM_IV", 5)' == py

    def test_delta_terminal(self):
        node = from_sexpr("Delta(RawSpread, EphInt(3))")
        py = _node_to_python(node)
        assert 'self._delta("RawSpread", 3)' == py

    def test_lag_non_terminal_uses_buffer(self):
        """Lag on non-terminal buffers the computed expression."""
        node = from_sexpr("Lag(Add(ATM_IV, VIXSpot), EphInt(5))")
        py = _node_to_python(node)
        assert "_lag_expr" in py
        assert "ATM_IV" in py

    def test_delta_non_terminal_uses_buffer(self):
        """Delta on non-terminal buffers the computed expression."""
        node = from_sexpr("Delta(Add(ATM_IV, VIXSpot), EphInt(5))")
        py = _node_to_python(node)
        assert "_delta_expr" in py

    def test_cross_above(self):
        node = from_sexpr("CrossAbove(ATM_IV, VIXSpot)")
        py = _node_to_python(node)
        assert "self._cross_above" in py

    def test_ifthenelse(self):
        node = from_sexpr("IfThenElse(GT(ATM_IV, EphReal(0.2)), VIXSpot, RealizedVol30m)")
        py = _node_to_python(node)
        assert " if " in py
        assert " else " in py

    def test_boolean_ops(self):
        node = from_sexpr("AND(GT(ATM_IV, EphReal(0.1)), NOT(LT(VIXSpot, EphReal(15.0))))")
        py = _node_to_python(node)
        assert " and " in py
        assert "<= 0.5)" in py  # NOT uses float(x) <= 0.5 to match proxy > 0.5 semantics

    def test_expression_is_valid_python(self):
        """The generated expression must be valid Python syntax (within a function body)."""
        node = from_sexpr(
            "GT(Delta(IfThenElse(LT(Lag(DeltaSpread1, EphInt(30)), "
            "Mul(AbstainFlag, VIXSpot)), ATM_IV, AbstainFlag), EphInt(1)), "
            "Sub(Sqrt(ATM_IV), Lag(Sqrt(RealizedVol30m), EphInt(30))))"
        )
        py = _node_to_python(node)
        # Wrap in a function so it's parseable
        code = f"def test_fn(self, s):\n    return {py}"
        ast.parse(code)  # should not raise


# ---------------------------------------------------------------------------
# Code generation tests
# ---------------------------------------------------------------------------

class TestCodegen:
    """Test full QC algorithm generation."""

    def test_generates_valid_python(self):
        code = generate_qc_algorithm(
            strategy_id="test_001",
            template_name="iron_condor_standard",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
            size_sexpr="EphReal(1.0)",
        )
        valid, err = validate_generated_code(code)
        assert valid, f"Generated code has syntax error: {err}"

    def test_class_name_in_output(self):
        code = generate_qc_algorithm(
            strategy_id="my_strat",
            template_name="iron_condor_standard",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
            size_sexpr="EphReal(1.0)",
        )
        assert "class GP_my_strat" in code

    def test_all_templates_generate(self):
        """Every template produces valid code."""
        for template_name in TEMPLATE_LEGS:
            code = generate_qc_algorithm(
                strategy_id=f"test_{template_name}",
                template_name=template_name,
                entry_sexpr="GT(ATM_IV, EphReal(0.18))",
                exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
                size_sexpr="EphReal(1.0)",
            )
            valid, err = validate_generated_code(code)
            assert valid, f"{template_name}: {err}"

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            generate_qc_algorithm(
                strategy_id="test",
                template_name="nonexistent",
                entry_sexpr="ATM_IV",
                exit_sexpr="ATM_IV",
                size_sexpr="ATM_IV",
            )

    def test_dates_in_output(self):
        code = generate_qc_algorithm(
            strategy_id="test",
            template_name="bull_put_credit_standard",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
            size_sexpr="EphReal(1.0)",
            start_date="2024-01-01",
            end_date="2025-06-30",
        )
        assert "SetStartDate(2024, 1, 1)" in code
        assert "SetEndDate(2025, 6, 30)" in code

    def test_leg_count_matches_template(self):
        """iron_condor_standard should have 4 _find_contract calls in open + 1 def."""
        code = generate_qc_algorithm(
            strategy_id="test_ic",
            template_name="iron_condor_standard",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
            size_sexpr="EphReal(1.0)",
        )
        # iron_condor orders exactly 4 legs: count the leg-ORDER calls
        # (`cN = self._find_contract(...)`), robust to the sizing path also calling
        # _find_contract per leg (b-gross alignment, 2026-05-31) and to comments.
        import re
        leg_order_calls = re.findall(r"c\d\s*=\s*self\._find_contract\(", code)
        assert len(leg_order_calls) == 4, (
            f"iron_condor must order 4 legs, found {len(leg_order_calls)} _find_contract leg calls"
        )

    def test_ratio_put_backspread_ratio_in_output(self):
        """Ratio put backspread has ratio=2 on the long OTM leg."""
        code = generate_qc_algorithm(
            strategy_id="test_rpb",
            template_name="ratio_put_backspread",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
            size_sexpr="EphReal(1.0)",
            delta_sexpr="SessionReturn",
        )
        assert "n_contracts * 2" in code

    def test_backward_compat_old_names_generate(self):
        """Old template names (without _standard) still work via backward-compat aliases."""
        for old_name in ["iron_condor", "iron_butterfly",
                         "bull_put_credit", "bear_call_credit"]:
            code = generate_qc_algorithm(
                strategy_id=f"test_{old_name}",
                template_name=old_name,
                entry_sexpr="GT(ATM_IV, EphReal(0.18))",
                exit_sexpr="LT(MinutesToClose, EphReal(30.0))",
                size_sexpr="EphReal(1.0)",
            )
            valid, err = validate_generated_code(code)
            assert valid, f"{old_name}: {err}"

    @pytest.mark.skipif(
        not Path("results/gp_A_local_20260505").exists(),
        reason="Condition A results not available",
    )
    def test_all_condition_a_strategies_generate_valid_code(self):
        """Every Condition A strategy generates syntactically valid QC Python."""
        total = 0
        for path in sorted(glob.glob("results/gp_A_local_20260505/strategies_*.jsonl")):
            with open(path) as f:
                for i, line in enumerate(f):
                    strat = json.loads(line)
                    code = generate_qc_algorithm(
                        strategy_id=f"{strat['template_name']}_{i}",
                        template_name=strat["template_name"],
                        entry_sexpr=strat["entry_tree"],
                        exit_sexpr=strat["exit_tree"],
                        size_sexpr=strat["size_tree"],
                    )
                    valid, err = validate_generated_code(code)
                    assert valid, f"{strat['template_name']}_{i}: {err}"
                    total += 1
        assert total == 1828, f"Expected 1828, got {total}"


# ---------------------------------------------------------------------------
# Template leg tests
# ---------------------------------------------------------------------------

class TestTemplateLegs:
    """Verify template leg definitions match layer2/templates.py."""

    def test_iron_condor_standard_has_4_legs(self):
        assert len(TEMPLATE_LEGS["iron_condor_standard"]) == 4

    def test_iron_butterfly_standard_has_4_legs(self):
        assert len(TEMPLATE_LEGS["iron_butterfly_standard"]) == 4

    def test_bull_put_credit_standard_has_2_legs(self):
        assert len(TEMPLATE_LEGS["bull_put_credit_standard"]) == 2

    def test_narrow_variants_leg_counts(self):
        assert len(TEMPLATE_LEGS["iron_condor_narrow"]) == 4
        assert len(TEMPLATE_LEGS["iron_butterfly_narrow"]) == 4
        assert len(TEMPLATE_LEGS["bull_put_credit_narrow"]) == 2
        assert len(TEMPLATE_LEGS["bear_call_credit_narrow"]) == 2

    def test_wide_variants_leg_counts(self):
        assert len(TEMPLATE_LEGS["iron_condor_wide"]) == 4
        assert len(TEMPLATE_LEGS["bull_put_credit_wide"]) == 2
        assert len(TEMPLATE_LEGS["bear_call_credit_wide"]) == 2

    def test_ratio_put_backspread_has_2_legs(self):
        """ratio_put_backspread: sell 1 near-ATM put, buy 2 OTM puts."""
        from layer3.codegen import _build_dynamic_legs
        legs = _build_dynamic_legs("ratio_put_backspread", 0.5)
        assert len(legs) == 2
        assert legs[1].ratio == 2
        assert legs[1].qty_sign == +1  # long OTM

    def test_all_scalar_templates_defined(self):
        expected = {
            "iron_condor_standard", "iron_butterfly_standard",
            "bull_put_credit_standard", "bear_call_credit_standard",
            "iron_condor_narrow", "iron_butterfly_narrow",
            "bull_put_credit_narrow", "bear_call_credit_narrow",
            "iron_condor_wide", "bull_put_credit_wide", "bear_call_credit_wide",
            "bull_call_debit", "bear_put_debit",
        }
        assert expected.issubset(set(TEMPLATE_LEGS.keys()))

    def test_no_enc_templates_in_legs(self):
        """Encoder-augmented _enc templates removed (2026-05-11)."""
        for name in TEMPLATE_LEGS:
            assert not name.endswith("_enc"), f"_enc template {name} still in TEMPLATE_LEGS"

    def test_backward_compat_old_names_in_legs(self):
        """Old names (without _standard) still map to correct legs."""
        assert TEMPLATE_LEGS["iron_condor"] == TEMPLATE_LEGS["iron_condor_standard"]
        assert TEMPLATE_LEGS["iron_butterfly"] == TEMPLATE_LEGS["iron_butterfly_standard"]
        assert TEMPLATE_LEGS["bull_put_credit"] == TEMPLATE_LEGS["bull_put_credit_standard"]
        assert TEMPLATE_LEGS["bear_call_credit"] == TEMPLATE_LEGS["bear_call_credit_standard"]

    def test_narrow_deltas_differ_from_standard(self):
        """Narrow variants should have different (closer to ATM) deltas than standard."""
        ic_std = TEMPLATE_LEGS["iron_condor_standard"]
        ic_nar = TEMPLATE_LEGS["iron_condor_narrow"]
        # Narrow short call is at 35d, standard at 25d
        assert abs(ic_nar[0].delta_target) > abs(ic_std[0].delta_target)

    def test_wide_deltas_differ_from_standard(self):
        """Wide variants should have different (farther from ATM) deltas than standard."""
        ic_std = TEMPLATE_LEGS["iron_condor_standard"]
        ic_wide = TEMPLATE_LEGS["iron_condor_wide"]
        # Wide short call is at 15d, standard at 25d
        assert abs(ic_wide[0].delta_target) < abs(ic_std[0].delta_target)


# ---------------------------------------------------------------------------
# BS delta inversion in codegen (proxy-to-QC alignment)
# ---------------------------------------------------------------------------

class TestBSDeltaInversion:
    """Verify the BS delta inversion embedded in generated QC code."""

    def test_bs_delta_to_strike_low_vol(self):
        """At IV=0.15 on 0DTE (200 min left), 25-delta put is ~0.3-0.8% OTM.
        0DTE tau is tiny (200/98280 = 0.002), so strikes are very close to ATM.
        The old hardcoded table (2% OTM) was 4x too far for 0DTE."""
        from layer2.evaluator import _norm_ppf
        iv, spx, mtc = 0.15, 5000, 200
        tau = mtc / (252.0 * 390.0)
        sigma_sqrt_tau = iv * math.sqrt(tau)
        d1 = _norm_ppf(1.0 - 0.25)
        strike = spx * math.exp(-d1 * sigma_sqrt_tau + 0.5 * sigma_sqrt_tau ** 2)
        strike = round(strike / 5.0) * 5.0
        otm_pct = (spx - strike) / spx
        assert 0.001 < otm_pct < 0.015, f"25d put 0DTE at IV=0.15: {otm_pct:.3%} OTM"

    def test_bs_delta_to_strike_high_vol(self):
        """At IV=0.40 on 0DTE, 25-delta put should be ~1-2% OTM.
        KEY REGRESSION: old hardcoded table always gave 2% regardless of IV.
        At high vol the correct strike is further OTM than at low vol."""
        from layer2.evaluator import _norm_ppf
        iv, spx, mtc = 0.40, 5000, 200
        tau = mtc / (252.0 * 390.0)
        sigma_sqrt_tau = iv * math.sqrt(tau)
        d1 = _norm_ppf(1.0 - 0.25)
        strike = spx * math.exp(-d1 * sigma_sqrt_tau + 0.5 * sigma_sqrt_tau ** 2)
        strike = round(strike / 5.0) * 5.0
        otm_pct = (spx - strike) / spx
        # High vol: further OTM than low vol
        assert 0.005 < otm_pct < 0.03, f"25d put 0DTE at IV=0.40: {otm_pct:.3%} OTM"

    def test_high_vol_further_otm_than_low_vol(self):
        """At higher IV, the same delta corresponds to a more OTM strike.
        This is what the old hardcoded table got wrong — it was IV-invariant."""
        from layer2.evaluator import _norm_ppf
        spx, mtc = 5000, 200

        def _bs_strike(iv, delta):
            tau = mtc / (252.0 * 390.0)
            sst = iv * math.sqrt(tau)
            d1 = _norm_ppf(1.0 - delta)
            return spx * math.exp(-d1 * sst + 0.5 * sst ** 2)

        strike_low = _bs_strike(0.15, 0.25)
        strike_high = _bs_strike(0.40, 0.25)
        # Higher IV → strike further from spot
        assert strike_high < strike_low, (
            f"High-vol strike ({strike_high:.0f}) should be further OTM than "
            f"low-vol ({strike_low:.0f})"
        )

    def test_bs_delta_to_strike_edges(self):
        """Extreme IV values produce no NaN/Inf, strike within bounds."""
        from layer2.evaluator import _norm_ppf
        spx = 5000
        for iv in [0.01, 0.05, 0.50, 1.0]:
            tau = 200 / (252.0 * 390.0)
            sigma_sqrt_tau = max(iv * math.sqrt(tau), 1e-8)
            d1 = _norm_ppf(1.0 - 0.25)
            strike = spx * math.exp(-d1 * sigma_sqrt_tau + 0.5 * sigma_sqrt_tau ** 2)
            strike = round(strike / 5.0) * 5.0
            assert not math.isnan(strike), f"NaN at IV={iv}"
            assert not math.isinf(strike), f"Inf at IV={iv}"
            assert spx * 0.70 < strike < spx * 1.30, f"Strike {strike} out of bounds at IV={iv}"

    def test_generated_code_no_hardcoded_table(self):
        """Generated code must NOT contain the old delta_to_otm lookup table."""
        code = generate_qc_algorithm(
            "test_no_table", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        assert "delta_to_otm" not in code, "Hardcoded delta_to_otm table still in generated code"

    def test_generated_code_has_norm_ppf(self):
        """Generated code must include _norm_ppf function."""
        code = generate_qc_algorithm(
            "test_ppf", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        assert "def _norm_ppf" in code

    def test_generated_code_has_bs_delta(self):
        """Generated code must include _bs_delta_to_strike method."""
        code = generate_qc_algorithm(
            "test_bsd", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        assert "_bs_delta_to_strike" in code

    def test_find_contract_uses_bs_strike_not_qc_greeks(self):
        """P0-7: _find_contract must select strikes via _bs_delta_to_strike (the
        proxy's delta->strike inversion on the self-computed ATM IV), NOT via
        QC-native c.Greeks.Delta. QC's IV is ~1.9x the collector's (P0-4), so
        selecting by QC Greeks would execute a DIFFERENT strike than the proxy
        modeled/evolved — an execution-axis training-serving skew."""
        import ast
        code = generate_qc_algorithm(
            "test_p07", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        tree = ast.parse(code)
        fc = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_find_contract"), None)
        assert fc is not None, "_find_contract must exist in generated code"
        # Check the AST (not raw text) so the docstring's explanatory mention of
        # "Greeks" doesn't count — only actual `.Greeks` attribute ACCESS in code.
        greeks_access = [n for n in ast.walk(fc)
                         if isinstance(n, ast.Attribute) and n.attr == "Greeks"]
        assert not greeks_access, (
            "_find_contract must NOT access QC-native .Greeks for strike selection "
            "(P0-7 execution-axis skew)"
        )
        bs_calls = [n for n in ast.walk(fc)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_bs_delta_to_strike"]
        assert bs_calls, (
            "_find_contract must compute the target strike via _bs_delta_to_strike"
        )

    def test_codegen_strike_matches_proxy_delta_to_strike(self):
        """P0-7 fit-for-purpose: the strike codegen executes (via _bs_delta_to_strike
        on a given ATM IV) must EQUAL the strike the proxy modeled
        (evaluator._delta_to_strike) for the same inputs — else QC trades a
        different option than was evolved. Verified byte-identical (same $5 strike)
        by extracting and executing the GENERATED method and comparing to the proxy
        directly.

        The grid deliberately spans mtc < 60 (the afternoon near-expiry window
        where 0DTE actually trades): the skew tau-damping (tau_ref=60min) and the
        intraday slope branches at mtc=120/240 only diverge there, so a grid that
        stopped at mtc>=60 would hide a real tau_ref/skew mismatch. Also probes the
        slope-branch boundaries, far-OTM low deltas (clamp stress), high IV, and the
        degenerate iv<=0 / mtc<=0 fallbacks."""
        import ast
        import textwrap
        from layer2.evaluator import MultiLegOptionsBacktester as _MB
        code = generate_qc_algorithm(
            "test_p07_num", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")

        def _extract(name):
            tree = ast.parse(code)
            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef) and n.name == name:
                    return textwrap.dedent(ast.get_source_segment(code, n))
            raise AssertionError(f"{name} not found in generated code")

        ns = {"math": math}
        exec(_extract("_norm_ppf"), ns)          # module-level helper used by the method
        exec(_extract("_bs_delta_to_strike"), ns)
        bs = ns["_bs_delta_to_strike"]           # method body has no self.* refs

        worst = 0.0
        for delta in (0.02, 0.05, 0.10, 0.25, 0.40, 0.45,
                      -0.02, -0.05, -0.10, -0.25, -0.40, -0.45):
            for spx in (3500.0, 4000.0, 5000.0, 6000.0, 6500.0):
                # 0.0005 pins the proxy _skew_iv `atm_iv<0.001 -> 0.01` guard
                # (unreachable in production but kept for in-isolation faithfulness).
                for iv in (0.0005, 0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70):
                    # mtc < 60 + the 120/240 slope-branch boundaries are the regimes
                    # a too-coarse grid would miss (where tau_ref/skew diverge).
                    for mtc in (1.0, 5.0, 15.0, 30.0, 45.0, 60.0, 90.0,
                                120.0, 121.0, 200.0, 240.0, 241.0, 380.0, 390.0):
                        proxy_k = _MB._delta_to_strike(spx, delta, iv, mtc)
                        qc_k = bs(None, delta, spx, iv, mtc)
                        worst = max(worst, abs(proxy_k - qc_k))
        # Degenerate fallbacks must agree too.
        for delta, spx, iv, mtc in ((0.25, 5000.0, 0.0, 120.0),
                                    (0.25, 5000.0, 0.15, 0.0),
                                    (-0.25, 5000.0, 0.0, 0.0)):
            worst = max(worst, abs(_MB._delta_to_strike(spx, delta, iv, mtc)
                                   - bs(None, delta, spx, iv, mtc)))
        assert worst == 0.0, (
            f"codegen strike must equal proxy _delta_to_strike exactly; max "
            f"divergence ${worst:.0f} (>=$5 means QC executes a different strike)"
        )

    def test_generated_code_syntax_valid(self):
        """Generated code must parse as valid Python."""
        code = generate_qc_algorithm(
            "test_syn", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        ast.parse(code)  # raises SyntaxError if invalid

    def test_strike_separation_in_generated_code(self):
        """Multi-leg templates must enforce $5 minimum strike separation."""
        code = generate_qc_algorithm(
            "test_sep", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        assert "Strike separation" in code, "Strike separation check missing from IC codegen"

    def test_entry_diagnostic_in_generated_code(self):
        """Generated code must include entry evaluation diagnostic logging."""
        code = generate_qc_algorithm(
            "test_diag", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))", "EphReal(1.0)")
        assert "Entry eval bar" in code, "Entry diagnostic logging missing"


# ---------------------------------------------------------------------------
# Cubic spline interpolation tests (proxy-matching ATM_IV/PutCallSkew/Spread)
# ---------------------------------------------------------------------------

class TestSplineInterpolation:
    """Verify cubic spline interpolation in generated code matches proxy methodology."""

    def _gen_code(self):
        return generate_qc_algorithm(
            "test_spline", "iron_condor_standard",
            "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))",
            "EphReal(1.0)",
        )

    def test_generated_code_has_cubic_spline_eval(self):
        """Generated code must include _cubic_spline_eval method."""
        code = self._gen_code()
        assert "_cubic_spline_eval" in code, "Missing _cubic_spline_eval in generated code"

    def test_generated_code_has_linear_interp(self):
        """Generated code must include _linear_interp fallback method."""
        code = self._gen_code()
        assert "_linear_interp" in code, "Missing _linear_interp in generated code"

    def test_generated_code_has_dedup_by_delta(self):
        """Generated code must include _dedup_by_delta helper."""
        code = self._gen_code()
        assert "_dedup_by_delta" in code, "Missing _dedup_by_delta in generated code"

    def test_generated_code_no_linear_correction(self):
        """Generated code must NOT apply the old 0.386*x+0.034 correction in code."""
        code = self._gen_code()
        # The pattern "0.386 * atm_iv" should NOT appear (only in comments)
        # Look for actual code applying the correction
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # skip comments
            assert "0.386 *" not in stripped, (
                f"Old linear correction still applied in code: {stripped}"
            )

    def test_generated_code_uses_spline_for_atm_iv(self):
        """ATM IV must come from spline interpolation, not single-contract lookup."""
        code = self._gen_code()
        assert "_put_atm_iv" in code, "Missing put-side ATM IV spline variable"
        assert "_call_atm_iv" in code, "Missing call-side ATM IV spline variable"
        assert "(_put_atm_iv + _call_atm_iv) / 2.0" in code, (
            "Missing ATM IV averaging of put/call splines"
        )

    def test_generated_code_uses_spline_for_put_call_skew(self):
        """PutCallSkew must use spline-interpolated 25-delta IVs."""
        code = self._gen_code()
        assert "_put_25d_iv" in code, "Missing put 25d IV spline variable"
        assert "_call_25d_iv" in code, "Missing call 25d IV spline variable"

    def test_generated_code_uses_spline_for_grid_reliability(self):
        """GridReliability must use spline-based coverage, not BS strike inversion."""
        code = self._gen_code()
        assert "_grid_put_deltas" in code, "Missing grid put delta targets"
        assert "_grid_call_deltas" in code, "Missing grid call delta targets"
        # Old approach used _bs_delta_to_strike for grid — should no longer be
        # present in the GridReliability section
        grid_section = code[code.index("GridReliability"):]
        # Should not call _bs_delta_to_strike in the grid section
        grid_end = grid_section.index("Apply normalization")
        grid_code = grid_section[:grid_end]
        assert "_bs_delta_to_strike" not in grid_code, (
            "GridReliability still uses BS delta inversion instead of spline"
        )

    def test_generated_code_uses_spline_for_spread(self):
        """RawSpread must come from spline interpolation at ATM."""
        code = self._gen_code()
        assert "_put_atm_spread" in code, "Missing put-side ATM spread spline"
        assert "_call_atm_spread" in code, "Missing call-side ATM spread spline"

    def test_spline_syntax_valid_all_templates(self):
        """All templates with spline produce syntactically valid code."""
        for tmpl in TEMPLATE_LEGS:
            code = generate_qc_algorithm(
                f"test_spline_{tmpl}", tmpl,
                "GT(ATM_IV, EphReal(0.1))", "LT(MinutesToClose, EphReal(30))",
                "EphReal(1.0)",
            )
            valid, err = validate_generated_code(code)
            assert valid, f"{tmpl}: {err}"

    def test_spline_knot_interpolation(self):
        """Spline must return exact values at knot points."""
        # Standalone implementation matching the template
        def _cubic_spline_eval(xs, ys, x_target):
            n = len(xs)
            if n < 2:
                return ys[0] if ys else 0.0
            if n == 2:
                t = (x_target - xs[0]) / max(xs[1] - xs[0], 1e-12)
                return ys[0] + t * (ys[1] - ys[0])
            if n == 3:
                h0 = xs[1] - xs[0]; h1 = xs[2] - xs[1]
                t0 = ((x_target-xs[1])*(x_target-xs[2]))/((xs[0]-xs[1])*(xs[0]-xs[2]))
                t1 = ((x_target-xs[0])*(x_target-xs[2]))/((xs[1]-xs[0])*(xs[1]-xs[2]))
                t2 = ((x_target-xs[0])*(x_target-xs[1]))/((xs[2]-xs[0])*(xs[2]-xs[1]))
                return max(0.0, min(5.0, ys[0]*t0 + ys[1]*t1 + ys[2]*t2))
            h = [xs[i+1]-xs[i] for i in range(n-1)]
            for i in range(len(h)):
                if h[i] < 1e-12: h[i] = 1e-12
            m = n - 2
            diag=[0.0]*m; sub=[0.0]*m; sup=[0.0]*m; rhs=[0.0]*m
            for i in range(m):
                j = i+1
                diag[i] = 2.0*(h[j-1]+h[j])
                rhs[i] = 6.0*((ys[j+1]-ys[j])/h[j]-(ys[j]-ys[j-1])/h[j-1])
                if i > 0: sub[i] = h[j-1]
                if i < m-1: sup[i] = h[j]
            for i in range(1, m):
                if abs(diag[i-1])<1e-15: diag[i-1]=1e-15
                w = sub[i]/diag[i-1]
                diag[i] -= w*sup[i-1]; rhs[i] -= w*rhs[i-1]
            M = [0.0]*n
            if abs(diag[m-1])<1e-15: diag[m-1]=1e-15
            M[m] = rhs[m-1]/diag[m-1]
            for i in range(m-2, -1, -1):
                if abs(diag[i])<1e-15: diag[i]=1e-15
                M[i+1] = (rhs[i]-sup[i]*M[i+2])/diag[i]
            if x_target <= xs[0]: k = 0
            elif x_target >= xs[-1]: k = n-2
            else:
                k = 0
                for i in range(n-1):
                    if xs[i]<=x_target<=xs[i+1]: k=i; break
            dx = x_target-xs[k]; hk=h[k]
            a = (M[k+1]-M[k])/(6.0*hk); b = M[k]/2.0
            c = (ys[k+1]-ys[k])/hk-hk*(2.0*M[k]+M[k+1])/6.0; d = ys[k]
            return max(0.0, min(5.0, a*dx**3+b*dx**2+c*dx+d))

        # Test 1: exact at knots
        xs = [0.05, 0.10, 0.25, 0.40, 0.50]
        ys = [0.25, 0.20, 0.15, 0.12, 0.11]
        for x, y in zip(xs, ys):
            result = _cubic_spline_eval(xs, ys, x)
            assert abs(result - y) < 1e-12, (
                f"Spline not exact at knot x={x}: got {result}, expected {y}"
            )

        # Test 2: monotonicity (IV decreasing with increasing delta for typical skew)
        prev = None
        for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            v = _cubic_spline_eval(xs, ys, t)
            if prev is not None:
                assert v <= prev + 0.005, (
                    f"Spline non-monotonic: {prev:.4f} -> {v:.4f} at delta={t}"
                )
            prev = v

        # Test 3: output in valid range
        for t in [0.01, 0.03, 0.50, 0.60, 0.70]:
            v = _cubic_spline_eval(xs, ys, t)
            assert 0.0 <= v <= 5.0, f"Spline out of range at {t}: {v}"

    def test_linear_interp_2_points(self):
        """Linear interp fallback works with 2 points."""
        def _linear_interp(xs, ys, x_target):
            n = len(xs)
            if n == 0: return 0.0
            if n == 1: return ys[0]
            if x_target <= xs[0]:
                t = (x_target-xs[0])/max(xs[1]-xs[0],1e-12)
                return ys[0]+t*(ys[1]-ys[0])
            if x_target >= xs[-1]:
                t = (x_target-xs[-2])/max(xs[-1]-xs[-2],1e-12)
                return ys[-2]+t*(ys[-1]-ys[-2])
            for i in range(n-1):
                if xs[i]<=x_target<=xs[i+1]:
                    t = (x_target-xs[i])/max(xs[i+1]-xs[i],1e-12)
                    return ys[i]+t*(ys[i+1]-ys[i])
            return ys[-1]

        xs = [0.10, 0.40]
        ys = [0.20, 0.12]
        # At endpoints
        assert abs(_linear_interp(xs, ys, 0.10) - 0.20) < 1e-12
        assert abs(_linear_interp(xs, ys, 0.40) - 0.12) < 1e-12
        # At midpoint
        assert abs(_linear_interp(xs, ys, 0.25) - 0.16) < 1e-12
        # Extrapolation
        v = _linear_interp(xs, ys, 0.50)
        assert abs(v - (0.12 + 0.1/0.3 * (0.12-0.20))) < 1e-12
