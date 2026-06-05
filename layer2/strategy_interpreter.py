"""Deterministic GP tree → plain-language interpreter.

Translates s-expression trees into human-readable descriptions for:
1. Diagnostician agent context (understanding what a strategy DOES)
2. Human-in-the-loop review (paper trading approval)
3. Audit trail (what was approved and why)

The interpreter is purely deterministic — no LLM involved. It walks the
tree and maps each operator/terminal to a natural-language fragment.
"""
from __future__ import annotations

from typing import Optional

from layer2.grammar import FuncNode, GType, Node, TermNode, from_sexpr, to_str
from layer2.terminal_stats import TERMINAL_NORM_STATS, denormalize


# ---------------------------------------------------------------------------
# Terminal descriptions
# ---------------------------------------------------------------------------

_TERMINAL_DESCRIPTIONS = {
    "ATM_IV": "ATM implied volatility",
    "VIXSpot": "VIX level",
    "VIXTermSlope": "VIX term structure slope",
    "VIXChange": "overnight VIX change",
    "VIXMean5d": "5-day mean VIX",
    "RealizedVol30m": "30-min realized volatility",
    "RealizedVol30m_5m": "smoothed realized volatility",
    "RawSpread": "ATM bid-ask spread",
    "RawSpread_5m": "smoothed ATM spread",
    "DeltaSpread1": "1-bar spread change",
    "DeltaSpread5": "5-bar spread change",
    "MinutesToClose": "minutes to close",
    "BarOfDay": "bar of day",
    "SessionReturn": "intraday return",
    "SessionPosition": "session position (0=low, 1=high)",
    "OvernightGap": "overnight gap",
    "GridReliability": "IV grid reliability",
    "PutCallSkew": "put-call skew",
    "ThetaUrgency": "theta urgency (1/√time)",
    "SPXReturn3d": "3-day SPX return",
    "RV5d": "5-day realized volatility",
    "IVRVGap5d": "5-day IV-RV gap (variance premium)",
    "ATM_IV_5m": "smoothed ATM IV",
    "PredRV15": "predicted 15-min RV",
    "PredRV30": "predicted 30-min RV",
    "PredSpread": "predicted spread change",
    "PredGammaAccel": "predicted gamma acceleration",
    "PredSmileConvexity": "predicted smile convexity change",
    "PredJump": "predicted jump probability",
    "PredFlowToxicity": "predicted flow toxicity",
    "RegimeAboveLow": "regime above low-vol",
    "RegimeIsHigh": "regime is high-vol",
    "RegimeIsPremium": "regime is crisis",
}


def _describe_terminal(node: TermNode) -> str:
    """Describe a terminal node in plain language."""
    name = node.name
    if name == "EphReal":
        raw_val = node.value
        return f"{raw_val:.2f}"
    if name == "EphInt":
        return f"{node.value} bars"
    if name in ("CALL", "PUT", "NEUTRAL"):
        return name.lower()
    if name in ("LOW_VOL", "MID_VOL", "HIGH_VOL", "HIGH_VOL_PREMIUM"):
        return name.replace("_", " ").lower()
    desc = _TERMINAL_DESCRIPTIONS.get(name, name)
    return desc


def _describe_ephreal_in_context(value: float, terminal_name: str) -> str:
    """Describe an EphReal value in the context of the terminal it's compared to.

    If the terminal has normalization stats, denormalize the EphReal to
    show the raw-scale value for interpretability.
    """
    stats = TERMINAL_NORM_STATS.get(terminal_name)
    if stats is not None:
        raw = denormalize(terminal_name, value)
        center, scale, method = stats
        # Format based on the terminal's natural scale
        if terminal_name in ("ATM_IV", "RealizedVol30m", "RawSpread", "PutCallSkew"):
            return f"{raw:.4f} ({value:+.2f}σ)"
        if terminal_name in ("VIXSpot", "VIXMean5d"):
            return f"{raw:.1f} ({value:+.2f}σ)"
        if terminal_name in ("MinutesToClose",):
            return f"{raw:.0f} min ({value:+.2f}σ)"
        if terminal_name in ("SessionReturn", "OvernightGap", "SPXReturn3d"):
            return f"{raw:.4f} ({value:+.2f}σ)"
        return f"{raw:.3f} ({value:+.2f}σ)"
    return f"{value:.2f}"


def _find_comparison_terminal(node: FuncNode) -> Optional[str]:
    """For GT/LT nodes, find the named terminal being compared to an EphReal."""
    if len(node.children) != 2:
        return None
    c0, c1 = node.children
    if isinstance(c0, TermNode) and c0.name not in ("EphReal", "EphInt"):
        return c0.name
    if isinstance(c1, TermNode) and c1.name not in ("EphReal", "EphInt"):
        return c1.name
    return None


# ---------------------------------------------------------------------------
# Tree interpreter
# ---------------------------------------------------------------------------

def interpret_node(node: Node, depth: int = 0) -> str:
    """Recursively interpret a GP tree node into plain language."""
    if isinstance(node, TermNode):
        return _describe_terminal(node)

    if not isinstance(node, FuncNode):
        return str(node)

    name = node.name
    children = node.children

    if name == "GT" and len(children) == 2:
        # Check for EphReal comparison to provide context
        term_name = _find_comparison_terminal(node)
        left = interpret_node(children[0], depth + 1)
        right = interpret_node(children[1], depth + 1)
        if term_name and isinstance(children[1], TermNode) and children[1].name == "EphReal":
            right = _describe_ephreal_in_context(children[1].value, term_name)
        elif term_name and isinstance(children[0], TermNode) and children[0].name == "EphReal":
            left = _describe_ephreal_in_context(children[0].value, term_name)
        return f"{left} > {right}"

    if name == "LT" and len(children) == 2:
        term_name = _find_comparison_terminal(node)
        left = interpret_node(children[0], depth + 1)
        right = interpret_node(children[1], depth + 1)
        if term_name and isinstance(children[1], TermNode) and children[1].name == "EphReal":
            right = _describe_ephreal_in_context(children[1].value, term_name)
        elif term_name and isinstance(children[0], TermNode) and children[0].name == "EphReal":
            left = _describe_ephreal_in_context(children[0].value, term_name)
        return f"{left} < {right}"

    if name == "AND" and len(children) == 2:
        left = interpret_node(children[0], depth + 1)
        right = interpret_node(children[1], depth + 1)
        if depth > 0:
            return f"({left} AND {right})"
        return f"{left} AND {right}"

    if name == "OR" and len(children) == 2:
        left = interpret_node(children[0], depth + 1)
        right = interpret_node(children[1], depth + 1)
        if depth > 0:
            return f"({left} OR {right})"
        return f"{left} OR {right}"

    if name == "NOT" and len(children) == 1:
        return f"NOT ({interpret_node(children[0], depth + 1)})"

    if name == "CrossAbove" and len(children) == 2:
        return f"{interpret_node(children[0], depth+1)} crosses above {interpret_node(children[1], depth+1)}"

    if name == "CrossBelow" and len(children) == 2:
        return f"{interpret_node(children[0], depth+1)} crosses below {interpret_node(children[1], depth+1)}"

    if name == "Lag" and len(children) == 2:
        return f"{interpret_node(children[0], depth+1)} lagged {interpret_node(children[1], depth+1)}"

    if name == "Delta" and len(children) == 2:
        return f"change in {interpret_node(children[0], depth+1)} over {interpret_node(children[1], depth+1)}"

    if name == "Add" and len(children) == 2:
        return f"({interpret_node(children[0], depth+1)} + {interpret_node(children[1], depth+1)})"

    if name == "Sub" and len(children) == 2:
        return f"({interpret_node(children[0], depth+1)} - {interpret_node(children[1], depth+1)})"

    if name == "Mul" and len(children) == 2:
        return f"({interpret_node(children[0], depth+1)} × {interpret_node(children[1], depth+1)})"

    if name == "Div" and len(children) == 2:
        return f"({interpret_node(children[0], depth+1)} / {interpret_node(children[1], depth+1)})"

    if name == "Sqrt" and len(children) == 1:
        return f"√({interpret_node(children[0], depth+1)})"

    if name == "IfThenElse" and len(children) == 3:
        cond = interpret_node(children[0], depth + 1)
        then = interpret_node(children[1], depth + 1)
        else_ = interpret_node(children[2], depth + 1)
        return f"if {cond} then {then} else {else_}"

    if name == "IfSide" and len(children) == 3:
        cond = interpret_node(children[0], depth + 1)
        then = interpret_node(children[1], depth + 1)
        else_ = interpret_node(children[2], depth + 1)
        return f"if {cond} then {then} else {else_}"

    # Fallback for unknown operators
    args = ", ".join(interpret_node(c, depth + 1) for c in children)
    return f"{name}({args})"


def interpret_strategy(
    entry_sexpr: str,
    exit_sexpr: str,
    size_sexpr: str = "EphReal(0.5)",
    delta_sexpr: Optional[str] = None,
    template_name: str = "",
    stop_mult: Optional[float] = None,
) -> str:
    """Interpret a complete strategy (entry + exit + size + delta) into
    plain-language description.

    Args:
        entry_sexpr: Entry condition s-expression.
        exit_sexpr: Exit condition s-expression.
        size_sexpr: Position size s-expression.
        delta_sexpr: Delta tree s-expression (Level B).
        template_name: Template name for context.
        stop_mult: Evolved stop-loss base credit-multiple (gp_engine stop gene);
            0.0 ⇒ hold-to-expiry. None ⇒ omit the STOP line.

    Returns:
        Multi-line plain-language description.
    """
    entry_tree = from_sexpr(entry_sexpr)
    exit_tree = from_sexpr(exit_sexpr)
    size_tree = from_sexpr(size_sexpr)

    lines = []
    if template_name:
        lines.append(f"Template: {template_name}")
    lines.append(f"ENTER when: {interpret_node(entry_tree)}")
    lines.append(f"EXIT when: {interpret_node(exit_tree)}")

    # Size interpretation
    size_desc = interpret_node(size_tree)
    if size_desc.startswith("0.") or size_desc.startswith("1."):
        lines.append(f"SIZE: {float(size_desc)*100:.0f}% of max position")
    else:
        lines.append(f"SIZE: {size_desc}")

    if delta_sexpr:
        delta_tree = from_sexpr(delta_sexpr)
        lines.append(f"DELTA: {interpret_node(delta_tree)}")

    if stop_mult is not None:
        if float(stop_mult) <= 0.0:
            lines.append("STOP: none — hold to expiry (no intraday stop-loss)")
        else:
            lines.append(
                f"STOP: exit when intraday loss exceeds {float(stop_mult):.2f}x the "
                f"credit received (evolved base; vol/time adjustments apply on top)"
            )

    return "\n".join(lines)
