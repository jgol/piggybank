"""Tests for codegen v5 fixes: dynamic delta_sexpr, buffer persistence, MAX_LAG, credit guard."""
import ast
import unittest


class TestDynamicDeltaSexpr(unittest.TestCase):
    """Test that delta_sexpr generates runtime delta evaluation in QC code."""

    def test_dynamic_delta_generates_eval_delta(self):
        from layer3.codegen import generate_qc_algorithm, validate_generated_code
        code = generate_qc_algorithm(
            strategy_id="test_dyn",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            delta_sexpr="Lag(BarOfDay, EphInt(5))",
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        ok, err = validate_generated_code(code)
        self.assertTrue(ok, f"Syntax error: {err}")
        self.assertIn("_eval_delta", code)
        self.assertIn("_short_delta", code)
        self.assertIn("Delta tree ->", code)

    def test_dynamic_delta_all_templates(self):
        from layer3.codegen import generate_qc_algorithm, validate_generated_code
        for tmpl in ["bull_put_credit", "bear_call_credit", "iron_condor",
                      "iron_butterfly", "ratio_put_backspread"]:
            code = generate_qc_algorithm(
                strategy_id=f"test_{tmpl}",
                template_name=tmpl,
                entry_sexpr="GT(ATM_IV, VIXChange)",
                exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
                size_sexpr="EphReal(0.5)",
                delta_sexpr="SessionReturn",
                start_date="2025-01-01",
                end_date="2025-06-01",
            )
            ok, err = validate_generated_code(code)
            self.assertTrue(ok, f"{tmpl} syntax error: {err}")
            self.assertIn("_eval_delta", code, f"{tmpl} missing _eval_delta")

    def test_static_delta_no_eval_delta_in_open(self):
        """Static delta_value should use fixed legs, not _eval_delta in open_position."""
        from layer3.codegen import generate_qc_algorithm
        code = generate_qc_algorithm(
            strategy_id="test_static",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            delta_value=0.5,
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        # _eval_delta method exists (always generated) but open_position uses fixed deltas
        self.assertNotIn("_delta_val = self._eval_delta", code)

    def test_delta_sexpr_overrides_delta_value(self):
        """When both delta_sexpr and delta_value are provided, delta_sexpr wins."""
        from layer3.codegen import generate_qc_algorithm
        code = generate_qc_algorithm(
            strategy_id="test_override",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            delta_value=0.3,
            delta_sexpr="SessionReturn",
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        # Dynamic delta takes precedence. Since the 2026-05-31 eval-schedule
        # parity fix, _on_bar evaluates the delta tree up-front every bar and the
        # open uses the cached value (_sig_delta) rather than re-evaluating it on
        # the fill bar (a second call would double-append the delta Lag/Delta
        # buffer). The presence of _sig_delta confirms dynamic delta is wired.
        self.assertIn("_delta_val = self._sig_delta", code)
        self.assertIn("self._sig_delta = self._eval_delta(s)", code)

    def test_v1_no_delta_still_works(self):
        """V1 mode (no delta_value, no delta_sexpr) should produce valid code."""
        from layer3.codegen import generate_qc_algorithm, validate_generated_code
        code = generate_qc_algorithm(
            strategy_id="test_v1",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        ok, err = validate_generated_code(code)
        self.assertTrue(ok, f"V1 syntax error: {err}")


class TestBufferPersistence(unittest.TestCase):
    """Per-bar Lag/Delta buffers MUST be cleared at the day boundary (P0-7b,
    2026-05-31), because the proxy masks within_day_pos<lag to 0 — i.e. it resets
    per day (evaluator_vectorized.py:296). Carrying buffers across days made
    Lag^k reach into the prior day, over-firing entries vs the proxy. This test
    was inverted from its pre-P0-7b form, which asserted the (wrong) opposite."""

    def test_buffers_cleared_at_day_boundary_for_proxy_parity(self):
        from layer3.codegen import generate_qc_algorithm
        code = generate_qc_algorithm(
            strategy_id="test_buf",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        # Day-boundary clear of the per-bar Lag/Delta buffers is present...
        self.assertIn("_lag_buf.clear()", code)
        # ...and the old "keep buffers across days" design is gone.
        self.assertNotIn("DO NOT clear rolling buffers", code)


class TestMaxLag(unittest.TestCase):
    """Test that MAX_LAG spans a full trading day."""

    def test_max_lag_at_least_405(self):
        from layer3.codegen import generate_qc_algorithm
        code = generate_qc_algorithm(
            strategy_id="test_lag",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        self.assertIn("MAX_LAG = 450", code)


class TestCreditGuardRemoved(unittest.TestCase):
    """Test that credit guard logs but does not Liquidate."""

    def test_no_liquidate_on_credit_divergence(self):
        from layer3.codegen import generate_qc_algorithm
        code = generate_qc_algorithm(
            strategy_id="test_guard",
            template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, VIXChange)",
            exit_sexpr="CrossAbove(ATM_IV, GridReliability)",
            size_sexpr="EphReal(0.5)",
            start_date="2025-01-01",
            end_date="2025-06-01",
        )
        # Credit guard section should log but NOT liquidate
        self.assertIn("DO NOT reject", code)
        # The regime guard block should contain Debug() for logging
        # but no _close_position() or direct Liquidate() call.
        # Check: the code between "regime_ok" check and "entry_signal" should
        # NOT contain _close_position or Liquidate.
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "regime guard" in line.lower() and "skipping entry" in line.lower():
                # This is the guard log line — there should be no Liquidate
                # in the surrounding 5 lines
                nearby = lines[max(0, i-2):i+3]
                for nl in nearby:
                    if "Liquidate" in nl and "def " not in nl and "_close_position" not in nl:
                        self.fail(f"Found Liquidate near regime guard at line {i}")


class TestPipelineLoaderFlat(unittest.TestCase):
    """Test that pipeline loader handles flat fold directory structure."""

    def test_load_from_flat_folds(self):
        """Pipeline should find strategies at results_dir/fold_N/strategies_*.jsonl."""
        import tempfile, json, os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create flat structure: tmpdir/fold_1/strategies_bull_put_credit.jsonl
            fold_dir = os.path.join(tmpdir, "fold_1")
            os.makedirs(fold_dir)
            strat = {
                "entry_tree": "GT(ATM_IV, VIXChange)",
                "exit_tree": "CrossAbove(ATM_IV, GridReliability)",
                "size_tree": "EphReal(0.5)",
                "delta_tree": "SessionReturn",
                "fitness": [-2.0, -1.5, -1.0],
                "val_sharpe": 2.0,
                "n_trades_val": 50,
            }
            with open(os.path.join(fold_dir, "strategies_bull_put_credit.jsonl"), "w") as f:
                f.write(json.dumps(strat) + "\n")

            from layer3.pipeline_v2 import load_gp_strategies
            strategies = load_gp_strategies(tmpdir, "bull_put_credit")
            self.assertEqual(len(strategies), 1)
            first = list(strategies.values())[0]
            self.assertEqual(first["delta_sexpr"], "SessionReturn")


if __name__ == "__main__":
    unittest.main()
