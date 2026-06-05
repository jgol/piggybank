"""Tests for the strategy interpreter (tree → plain language)."""
import pytest


class TestTerminalInterpretation:
    """Terminal nodes produce readable descriptions."""

    def test_named_terminal(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("ATM_IV")
        assert interpret_node(node) == "ATM implied volatility"

    def test_ephreal(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("EphReal(0.5)")
        assert "0.50" in interpret_node(node)

    def test_ephint(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("EphInt(5)")
        assert "5 bars" in interpret_node(node)

    def test_side_literal(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("CALL")
        assert interpret_node(node) == "call"


class TestComparisonInterpretation:
    """GT/LT with EphReal show denormalized values."""

    def test_gt_with_ephreal_shows_raw_value(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("GT(ATM_IV, EphReal(1.0))")
        result = interpret_node(node)
        assert "ATM implied volatility" in result
        assert ">" in result
        # Should show denormalized value and sigma
        assert "σ" in result

    def test_lt_with_ephreal(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("LT(MinutesToClose, EphReal(-0.5))")
        result = interpret_node(node)
        assert "minutes to close" in result
        assert "<" in result


class TestLogicalOperators:
    """AND/OR/NOT produce readable logic."""

    def test_and(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("AND(GT(ATM_IV, EphReal(0.5)), LT(MinutesToClose, EphReal(0.0)))")
        result = interpret_node(node)
        assert "AND" in result
        assert "ATM implied volatility" in result
        assert "minutes to close" in result

    def test_or(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("OR(GT(VIXSpot, EphReal(1.0)), LT(SessionReturn, EphReal(-1.0)))")
        result = interpret_node(node)
        assert "OR" in result

    def test_not(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("NOT(GT(ATM_IV, EphReal(0.0)))")
        result = interpret_node(node)
        assert "NOT" in result


class TestTemporalOperators:
    """Lag/Delta/CrossAbove produce readable descriptions."""

    def test_lag(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("Lag(ATM_IV, EphInt(5))")
        result = interpret_node(node)
        assert "lagged" in result
        assert "5 bars" in result

    def test_delta(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("Delta(ATM_IV, EphInt(5))")
        result = interpret_node(node)
        assert "change in" in result

    def test_cross_above(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("CrossAbove(ATM_IV, VIXSpot)")
        result = interpret_node(node)
        assert "crosses above" in result


class TestArithmeticOperators:

    def test_add(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("Add(ATM_IV, VIXSpot)")
        result = interpret_node(node)
        assert "+" in result

    def test_mul(self):
        from layer2.strategy_interpreter import interpret_node
        from layer2.grammar import from_sexpr
        node = from_sexpr("Mul(VIXTermSlope, EphReal(0.3))")
        result = interpret_node(node)
        assert "×" in result


class TestFullStrategyInterpretation:
    """Complete strategy interpretation."""

    def test_interpret_strategy(self):
        from layer2.strategy_interpreter import interpret_strategy
        result = interpret_strategy(
            entry_sexpr="AND(GT(ATM_IV, EphReal(0.5)), LT(SessionReturn, EphReal(-1.0)))",
            exit_sexpr="OR(LT(MinutesToClose, EphReal(-1.0)), GT(RealizedVol30m, EphReal(2.0)))",
            size_sexpr="EphReal(0.5)",
            template_name="bull_put_credit",
        )
        assert "ENTER when:" in result
        assert "EXIT when:" in result
        assert "SIZE:" in result
        assert "bull_put_credit" in result

    def test_interpret_with_delta(self):
        from layer2.strategy_interpreter import interpret_strategy
        result = interpret_strategy(
            entry_sexpr="GT(ATM_IV, EphReal(0.0))",
            exit_sexpr="LT(MinutesToClose, EphReal(-1.0))",
            size_sexpr="EphReal(0.5)",
            delta_sexpr="Add(VIXSpot, EphReal(0.3))",
            template_name="iron_condor",
        )
        assert "DELTA:" in result

    def test_size_percentage(self):
        from layer2.strategy_interpreter import interpret_strategy
        result = interpret_strategy(
            entry_sexpr="GT(EphReal(1.0), EphReal(0.0))",
            exit_sexpr="LT(EphReal(-1.0), EphReal(0.0))",
            size_sexpr="EphReal(0.5)",
        )
        assert "50%" in result
