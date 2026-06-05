"""Semantic equivalence test: L2 evaluator vs codegen-generated Python.

Feeds identical bar data through both the L2 TreeEvaluator and the generated
Python expression, checking they produce the same result. This catches
semantic divergences that syntax-only tests miss (e.g., wrong operator mapping,
off-by-one in lag, NaN handling differences).
"""
import json
import glob
import math
import random
import pytest
from collections import deque
from pathlib import Path
from typing import Any, Dict

from layer2.grammar import from_sexpr, to_str, Node, FuncNode, TermNode, GType
from layer2.evaluator import EvaluationContext, TreeEvaluator
from layer2.terminal_stats import normalize, TERMINAL_NORM_STATS
from layer3.codegen import _node_to_python


# ---------------------------------------------------------------------------
# Mini QC-like runtime that executes codegen expressions
# ---------------------------------------------------------------------------

class CodegenRuntime:
    """Minimal runtime environment for evaluating codegen-generated expressions.

    Mirrors the self.* methods available in the generated QCAlgorithm class:
    _lag, _delta, _cross_above, _cross_below, _safe, plus the `s` dict.
    """

    MAX_LAG = 30

    def __init__(self):
        self._buffers: Dict[str, deque] = {}
        self._prev_eval: Dict[str, float] = {}
        self._curr_eval: Dict[str, float] = {}

    def push_bar(self, scalars: Dict[str, float]):
        """Push scalar values into rolling buffers (mirrors _on_bar buffer update)."""
        for name, val in scalars.items():
            if name not in self._buffers:
                self._buffers[name] = deque(maxlen=self.MAX_LAG + 1)
            self._buffers[name].append(val)

    def rotate_eval_cache(self):
        """Rotate cross-detection cache (call after tree evaluation)."""
        self._prev_eval = dict(self._curr_eval)
        self._curr_eval = {}

    def _lag(self, name, k):
        k = max(0, min(int(k), self.MAX_LAG))
        buf = self._buffers.get(name)
        if not buf or len(buf) == 0:
            return 0.0
        idx = len(buf) - 1 - k
        return float(buf[idx]) if idx >= 0 else 0.0

    def _delta(self, name, k):
        k = max(1, min(int(k), self.MAX_LAG))
        return self._lag(name, 0) - self._lag(name, k)

    def _lag_expr(self, key, current_val, k):
        """Lag a computed expression: store in buffer, return k-lagged."""
        current_val = float(current_val) if current_val == current_val else 0.0
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.MAX_LAG + 1)
        self._buffers[key].append(current_val)
        return self._lag(key, max(0, min(int(k), self.MAX_LAG)))

    def _delta_expr(self, key, current_val, k):
        """Delta of a computed expression: store, return current - lagged."""
        current_val = float(current_val) if current_val == current_val else 0.0
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.MAX_LAG + 1)
        self._buffers[key].append(current_val)
        return self._lag(key, 0) - self._lag(key, max(1, min(int(k), self.MAX_LAG)))

    def _cross_above(self, ka, kb, a, b):
        self._curr_eval[ka] = a
        self._curr_eval[kb] = b
        pa = self._prev_eval.get(ka)
        pb = self._prev_eval.get(kb)
        if pa is None or pb is None:
            return False
        return a > b and pa <= pb

    def _cross_below(self, ka, kb, a, b):
        self._curr_eval[ka] = a
        self._curr_eval[kb] = b
        pa = self._prev_eval.get(ka)
        pb = self._prev_eval.get(kb)
        if pa is None or pb is None:
            return False
        return a < b and pa >= pb

    def _safe(self, x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return float(x)
        return 0.0

    def eval_expr(self, expr_str: str, scalars: Dict[str, float]) -> Any:
        """Evaluate a codegen expression string in this runtime's context."""
        s = scalars  # noqa: F841 — referenced by the expression via s["..."]
        # The expression uses `self` for _lag, _delta, etc.
        self_ref = self  # noqa: F841
        # Rewrite self.xxx to self_ref.xxx so eval() can find it
        code = expr_str.replace("self.", "self_ref.")
        return eval(code)


# ---------------------------------------------------------------------------
# Synthetic bar data generators
# ---------------------------------------------------------------------------

def _random_bar(rng: random.Random) -> Dict[str, float]:
    """Generate a realistic-looking bar of scalar data (raw, pre-normalization)."""
    return {
        "ATM_IV": rng.uniform(0.05, 0.50),
        "VIXSpot": rng.uniform(12.0, 45.0),
        "VIXTermSlope": rng.uniform(-3.0, 5.0),
        "RealizedVol30m": rng.uniform(0.005, 0.060),
        "RawSpread": rng.uniform(0.0, 0.30),
        "DeltaSpread1": rng.uniform(-0.10, 0.10),
        "DeltaSpread5": rng.uniform(-0.15, 0.15),
        "MinutesToClose": rng.uniform(5.0, 330.0),
        "BarOfDay": rng.uniform(0.0, 78.0),
        "AbstainFlag": 0.0,  # removed from grammar but existing strategies still reference it
    }


def _feed_bars(ctx: EvaluationContext, runtime: CodegenRuntime,
               bars: list, rotate_cross: bool = True):
    """Feed bars into both L2 context and codegen runtime in parallel."""
    for bar in bars:
        ctx.update(bar)
        runtime.push_bar(bar)
        if rotate_cross:
            # L2 evaluator rotates in _eval_function for CrossAbove/Below
            # Codegen rotates at end of _on_bar
            # We do NOT rotate here — we rotate after eval to match real flow
            pass


def _safe_compare(l2_result, cg_result) -> bool:
    """Compare L2 evaluator result with codegen result, handling type differences."""
    # Both bool
    if isinstance(l2_result, bool) and isinstance(cg_result, bool):
        return l2_result == cg_result
    # Both numeric
    if isinstance(l2_result, (int, float)) and isinstance(cg_result, (int, float)):
        # NaN handling
        if math.isnan(float(l2_result)) and math.isnan(float(cg_result)):
            return True
        if math.isnan(float(l2_result)) or math.isnan(float(cg_result)):
            return False
        return abs(float(l2_result) - float(cg_result)) < 1e-10
    # L2 returns bool, codegen returns int 0/1 — both acceptable for bool trees
    if isinstance(l2_result, bool) and isinstance(cg_result, (int, float)):
        return l2_result == bool(cg_result)
    if isinstance(l2_result, (int, float)) and isinstance(cg_result, bool):
        return bool(l2_result) == cg_result
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSemanticEquivalence:
    """Verify L2 evaluator and codegen produce identical results on same inputs."""

    def _run_equivalence(self, sexpr: str, n_bars: int = 40, seed: int = 42):
        """Run one tree through both paths and compare on every bar after warmup."""
        node = from_sexpr(sexpr)
        py_expr = _node_to_python(node)

        rng = random.Random(seed)
        ctx = EvaluationContext(max_lag=30)
        runtime = CodegenRuntime()
        evaluator = TreeEvaluator()

        mismatches = []
        for bar_idx in range(n_bars):
            bar = _random_bar(rng)
            # ctx.update() normalizes terminal values internally AND rotates cross caches
            ctx.update(bar)
            # Normalize bar for codegen runtime (matching what evaluator does internally)
            norm_bar = {}
            for k, v in bar.items():
                if k in TERMINAL_NORM_STATS:
                    norm_bar[k] = normalize(k, v)
                else:
                    norm_bar[k] = v
            runtime.push_bar(norm_bar)

            # Evaluate in both systems — both see normalized values
            l2_result = evaluator.evaluate(node, ctx)
            cg_result = runtime.eval_expr(py_expr, norm_bar)

            if not _safe_compare(l2_result, cg_result):
                mismatches.append({
                    "bar": bar_idx,
                    "l2": l2_result,
                    "cg": cg_result,
                    "bar_data": {k: round(v, 4) for k, v in bar.items()},
                })

            # Only rotate codegen runtime cache (L2 ctx rotates in update())
            runtime.rotate_eval_cache()

        return mismatches

    # -- Simple operators --

    def test_gt_constant(self):
        m = self._run_equivalence("GT(ATM_IV, EphReal(0.18))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_lt_terminals(self):
        m = self._run_equivalence("LT(RealizedVol30m, DeltaSpread1)")
        assert not m, f"Mismatches: {m[:3]}"

    def test_add(self):
        m = self._run_equivalence("Add(ATM_IV, VIXSpot)")
        assert not m, f"Mismatches: {m[:3]}"

    def test_sub(self):
        m = self._run_equivalence("Sub(VIXTermSlope, RealizedVol30m)")
        assert not m, f"Mismatches: {m[:3]}"

    def test_mul(self):
        m = self._run_equivalence("Mul(ATM_IV, EphReal(0.5))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_div_analytic_quotient(self):
        m = self._run_equivalence("Div(ATM_IV, VIXSpot)")
        assert not m, f"Mismatches: {m[:3]}"

    def test_sqrt(self):
        m = self._run_equivalence("Sqrt(RealizedVol30m)")
        assert not m, f"Mismatches: {m[:3]}"

    def test_not(self):
        m = self._run_equivalence("NOT(GT(ATM_IV, EphReal(0.2)))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_and(self):
        m = self._run_equivalence("AND(GT(ATM_IV, EphReal(0.1)), LT(VIXSpot, EphReal(20.0)))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_or(self):
        m = self._run_equivalence("OR(GT(ATM_IV, EphReal(0.3)), LT(RealizedVol30m, EphReal(0.01)))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_ifthenelse(self):
        m = self._run_equivalence("IfThenElse(GT(ATM_IV, EphReal(0.2)), VIXSpot, RealizedVol30m)")
        assert not m, f"Mismatches: {m[:3]}"

    # -- Temporal operators --

    def test_lag_terminal(self):
        m = self._run_equivalence("Lag(ATM_IV, EphInt(5))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_lag_large(self):
        m = self._run_equivalence("Lag(RawSpread, EphInt(20))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_delta_terminal(self):
        m = self._run_equivalence("Delta(RawSpread, EphInt(3))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_lag_non_terminal_buffers_correctly(self):
        """Lag on non-terminal: codegen now buffers computed expressions and
        returns the actual lagged value (proxy evaluator returns current value
        as a no-op — this is a known proxy limitation, not a codegen bug)."""
        from layer3.codegen import _node_to_python
        from layer2.grammar import from_sexpr
        py = _node_to_python(from_sexpr("Lag(Add(ATM_IV, VIXSpot), EphInt(5))"))
        assert "_lag_expr" in py, f"Expected _lag_expr call, got: {py}"

    def test_delta_non_terminal_buffers_correctly(self):
        """Delta on non-terminal: codegen now buffers computed expressions and
        returns actual delta (proxy evaluator returns 0.0 as a no-op)."""
        from layer3.codegen import _node_to_python
        from layer2.grammar import from_sexpr
        py = _node_to_python(from_sexpr("Delta(Add(ATM_IV, VIXSpot), EphInt(5))"))
        assert "_delta_expr" in py, f"Expected _delta_expr call, got: {py}"

    # -- Cross operators --

    def test_cross_above(self):
        m = self._run_equivalence("CrossAbove(ATM_IV, RealizedVol30m)", n_bars=50)
        assert not m, f"Mismatches: {m[:3]}"

    def test_cross_below(self):
        m = self._run_equivalence("CrossBelow(VIXSpot, MinutesToClose)", n_bars=50)
        assert not m, f"Mismatches: {m[:3]}"

    # -- Nested / complex trees --

    def test_nested_ifthenelse(self):
        m = self._run_equivalence(
            "IfThenElse(AND(GT(ATM_IV, EphReal(0.15)), NOT(LT(VIXSpot, EphReal(15.0)))), "
            "Sub(Sqrt(ATM_IV), Lag(RealizedVol30m, EphInt(5))), "
            "Mul(DeltaSpread1, EphReal(2.0)))"
        )
        assert not m, f"Mismatches: {m[:3]}"

    def test_gt_with_delta_and_lag(self):
        """GT(Delta(terminal, k), Lag(FuncNode, k)) — Delta on terminal matches,
        Lag on FuncNode now correctly buffers in codegen (diverges from L2 proxy
        which returns current value). Test codegen produces valid Python."""
        from layer3.codegen import _node_to_python
        from layer2.grammar import from_sexpr
        py = _node_to_python(from_sexpr(
            "GT(Delta(ATM_IV, EphInt(5)), Lag(Sqrt(RealizedVol30m), EphInt(10)))"))
        assert "_delta" in py
        assert "_lag_expr" in py

    def test_cross_with_nested_args(self):
        m = self._run_equivalence(
            "CrossAbove(Add(ATM_IV, VIXTermSlope), Delta(DeltaSpread1, EphInt(10)))",
            n_bars=50,
        )
        assert not m, f"Mismatches: {m[:3]}"

    def test_deeply_nested(self):
        """Deeply nested tree with Lag(FuncNode) — codegen now correctly lags
        the computed Sqrt(RealizedVol30m) via buffer, L2 proxy returns current.
        Validate codegen produces valid Python with _lag_expr."""
        from layer3.codegen import _node_to_python
        from layer2.grammar import from_sexpr
        py = _node_to_python(from_sexpr(
            "GT(IfThenElse(LT(Lag(DeltaSpread1, EphInt(10)), Mul(AbstainFlag, VIXSpot)), "
            "ATM_IV, AbstainFlag), "
            "Sub(Sqrt(ATM_IV), Lag(Sqrt(RealizedVol30m), EphInt(20))))"))
        assert "_lag_expr" in py
        assert "_lag(" in py  # Lag(DeltaSpread1, 10) uses terminal _lag

    def test_div_chain(self):
        """Chain of Div ops — tests analytic quotient composition."""
        m = self._run_equivalence(
            "Div(Div(ATM_IV, VIXSpot), Div(RealizedVol30m, DeltaSpread1))"
        )
        assert not m, f"Mismatches: {m[:3]}"

    # -- Real Condition A strategies --

    @pytest.mark.skipif(
        not Path("results/gp_A_local_20260505").exists(),
        reason="Condition A results not available",
    )
    def test_real_strategies_sample(self):
        """Test 10 random real Condition A strategies for semantic equivalence."""
        all_strats = []
        for path in glob.glob("results/gp_A_local_20260505/strategies_*.jsonl"):
            with open(path) as f:
                for line in f:
                    all_strats.append(json.loads(line))

        rng = random.Random(42)
        sample = rng.sample(all_strats, min(10, len(all_strats)))

        for i, strat in enumerate(sample):
            for tree_key in ["entry_tree", "exit_tree", "size_tree"]:
                sexpr = strat[tree_key]
                m = self._run_equivalence(sexpr, n_bars=40, seed=100 + i)
                assert not m, (
                    f"Strategy {strat['template_name']} {tree_key}: "
                    f"{len(m)} mismatches. First: bar={m[0]['bar']}, "
                    f"L2={m[0]['l2']}, CG={m[0]['cg']}"
                )

    @pytest.mark.skipif(
        not Path("results/gp_A_local_20260505").exists(),
        reason="Condition A results not available",
    )
    def test_all_entry_trees_equivalence(self):
        """Semantic equivalence on ALL 1828 entry trees (the most critical tree).

        Entry tree determines whether a position is opened — semantic divergence
        here directly changes which trades the QC backtest executes.
        """
        total = 0
        failed = []
        for path in sorted(glob.glob("results/gp_A_local_20260505/strategies_*.jsonl")):
            template = path.split("strategies_")[1].replace(".jsonl", "")
            with open(path) as f:
                for idx, line in enumerate(f):
                    strat = json.loads(line)
                    m = self._run_equivalence(
                        strat["entry_tree"], n_bars=35, seed=total
                    )
                    if m:
                        failed.append(
                            f"{template}_{idx}: {len(m)} mismatches, "
                            f"first at bar {m[0]['bar']}"
                        )
                    total += 1
        assert not failed, (
            f"{len(failed)}/{total} entry trees have semantic divergence:\n"
            + "\n".join(failed[:10])
        )


class TestEdgeCases:
    """Edge cases that could cause semantic divergence."""

    def test_ephreal_zero(self):
        """EphReal(0.0) — division by zero in Div."""
        m = TestSemanticEquivalence()._run_equivalence("Div(ATM_IV, EphReal(0.0))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_negative_ephreal_in_sqrt(self):
        """Sqrt of negative — protected sqrt(abs(x))."""
        m = TestSemanticEquivalence()._run_equivalence("Sqrt(EphReal(-0.5))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_lag_zero(self):
        """Lag with k=0 should return current value."""
        # EphInt range is [1,30] but tree could have EphInt(1) and Lag clamps
        m = TestSemanticEquivalence()._run_equivalence("Lag(ATM_IV, EphInt(1))")
        assert not m, f"Mismatches: {m[:3]}"

    def test_constant_comparison(self):
        """EphReal vs EphReal — constant comparison."""
        m = TestSemanticEquivalence()._run_equivalence(
            "GT(EphReal(0.5), EphReal(-0.5))"
        )
        assert not m, f"Mismatches: {m[:3]}"

    def test_cross_first_bar(self):
        """CrossAbove on first bar should return False (no previous)."""
        node = from_sexpr("CrossAbove(ATM_IV, VIXSpot)")
        py_expr = _node_to_python(node)

        ctx = EvaluationContext(max_lag=30)
        runtime = CodegenRuntime()
        evaluator = TreeEvaluator()

        bar = _random_bar(random.Random(42))
        ctx.update(bar)
        runtime.push_bar(bar)

        l2 = evaluator.evaluate(node, ctx)
        cg = runtime.eval_expr(py_expr, bar)
        assert l2 == False  # noqa: E712
        assert cg == False  # noqa: E712
