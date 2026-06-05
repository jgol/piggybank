"""Grammar-constrained mutation validation + statistical testing.

Validates LLM-proposed mutations against grammar.py rules, applies
valid mutations to create modified trees, and runs Wilcoxon signed-rank
test across walk-forward folds to determine statistical significance.

Usage:
    from layer3.mutation_validator import validate_and_apply_mutation

    result = validate_and_apply_mutation(
        original_sexpr="LT(OvernightGap, Mul(VIXChange, BarOfDay))",
        mutation=tree_mutation,
        grammar=grammar,
    )
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from layer3.schemas import TreeMutation, MutationType, TreeSlot


# ---------------------------------------------------------------------------
# Grammar validation
# ---------------------------------------------------------------------------

# Valid REAL terminals per condition — imported from canonical grammar source
def _build_valid_terminals() -> Dict[str, frozenset]:
    """Build per-condition terminal sets from grammar.py (single source of truth)."""
    from layer2.grammar import (
        build_scalar_only_terminal_set,
        build_probes_only_terminal_set,
        build_emb_only_terminal_set,
        build_terminal_set,
        GType,
    )
    result = {}
    for condition, builder in [
        ("scalar-only", build_scalar_only_terminal_set),
        ("probes-only", build_probes_only_terminal_set),
        ("emb-only", build_emb_only_terminal_set),
        ("real-l1", build_terminal_set),
    ]:
        terms = builder()
        result[condition] = frozenset(
            t.name for t in terms
            if t.ret_type == GType.REAL and t.name not in ("EphReal",)
        )
    return result

_VALID_TERMINALS_BY_CONDITION: Optional[Dict[str, frozenset]] = None

def _get_valid_terminals(condition: str = "scalar-only") -> frozenset:
    """Get valid REAL terminals for a condition (lazy-loaded)."""
    global _VALID_TERMINALS_BY_CONDITION
    if _VALID_TERMINALS_BY_CONDITION is None:
        _VALID_TERMINALS_BY_CONDITION = _build_valid_terminals()
    return _VALID_TERMINALS_BY_CONDITION.get(condition, _VALID_TERMINALS_BY_CONDITION["scalar-only"])

# Valid functions by return type
VALID_BOOL_FUNCTIONS = frozenset({
    "GT", "LT", "AND", "OR", "NOT", "CrossAbove", "CrossBelow",
    "InRegime", "RegimeIs",
})
VALID_REAL_FUNCTIONS = frozenset({
    "Add", "Sub", "Mul", "Div", "Sqrt", "Lag", "Delta", "IfThenElse",
})

# Valid EphInt values (lag periods)
VALID_EPHINT_VALUES = frozenset({1, 3, 5, 10, 15, 30})

# Max tree size
MAX_NODES = 15


def validate_terminal_swap(mutation: TreeMutation,
                           condition: str = "scalar-only") -> Tuple[bool, str]:
    """Check if a terminal swap is grammar-legal for the given condition."""
    from_term = mutation.from_node.strip()
    to_term = mutation.to_node.strip()

    valid_terms = _get_valid_terminals(condition)
    if from_term not in valid_terms and not from_term.startswith("EphReal"):
        return False, f"from_node '{from_term}' is not a valid REAL terminal for {condition}"
    if to_term not in valid_terms and not to_term.startswith("EphReal"):
        return False, f"to_node '{to_term}' is not a valid REAL terminal for {condition}"

    return True, "valid terminal swap"


def validate_function_swap(mutation: TreeMutation) -> Tuple[bool, str]:
    """Check if a function swap preserves type signature."""
    from_func = mutation.from_node.strip()
    to_func = mutation.to_node.strip()

    # Both must be valid functions
    all_funcs = VALID_BOOL_FUNCTIONS | VALID_REAL_FUNCTIONS
    if from_func not in all_funcs:
        return False, f"from_node '{from_func}' is not a valid function"
    if to_func not in all_funcs:
        return False, f"to_node '{to_func}' is not a valid function"

    # Must be same return type
    from_is_bool = from_func in VALID_BOOL_FUNCTIONS
    to_is_bool = to_func in VALID_BOOL_FUNCTIONS
    if from_is_bool != to_is_bool:
        return False, f"type mismatch: {from_func} ({'BOOL' if from_is_bool else 'REAL'}) → {to_func} ({'BOOL' if to_is_bool else 'REAL'})"

    # Arity check — group functions by argument count
    unary_funcs = {"NOT", "Sqrt", "InRegime"}
    binary_funcs = {"GT", "LT", "AND", "OR", "CrossAbove", "CrossBelow",
                    "Add", "Sub", "Mul", "Div", "Lag", "Delta", "RegimeIs"}
    ternary_funcs = {"IfThenElse", "IfSide"}

    def _arity_of(f: str) -> int:
        if f in unary_funcs: return 1
        if f in binary_funcs: return 2
        if f in ternary_funcs: return 3
        return -1  # unknown

    from_arity = _arity_of(from_func)
    to_arity = _arity_of(to_func)
    if from_arity != to_arity:
        return False, f"arity mismatch: {from_func} (arity {from_arity}) → {to_func} (arity {to_arity})"

    return True, "valid function swap"


def validate_ephemeral_perturbation(mutation: TreeMutation) -> Tuple[bool, str]:
    """Check if an ephemeral perturbation is valid."""
    to_val = mutation.to_node.strip()

    # EphReal: must be in [-3.0, 3.0]
    if to_val.startswith("EphReal("):
        try:
            val = float(to_val[8:-1])
            if -3.0 <= val <= 3.0:
                return True, f"valid EphReal perturbation: {val}"
            return False, f"EphReal({val}) out of range [-3.0, 3.0]"
        except ValueError:
            return False, f"cannot parse EphReal value: {to_val}"

    # EphInt: must be in valid set
    if to_val.startswith("EphInt("):
        try:
            val = int(to_val[7:-1])
            if val in VALID_EPHINT_VALUES:
                return True, f"valid EphInt perturbation: {val}"
            return False, f"EphInt({val}) not in valid set {sorted(VALID_EPHINT_VALUES)}"
        except ValueError:
            return False, f"cannot parse EphInt value: {to_val}"

    return False, f"cannot identify ephemeral type: {to_val}"


def validate_structural(mutation: TreeMutation) -> Tuple[bool, str]:
    """Check structural mutation (insert/remove node).

    Structural mutations are harder to validate without the full tree.
    We do basic checks here; full validation happens when the tree is rebuilt.
    """
    to_node = mutation.to_node.strip()

    # If inserting a new subtree, check it parses
    if "(" in to_node:
        # Basic bracket matching
        open_count = to_node.count("(")
        close_count = to_node.count(")")
        if open_count != close_count:
            return False, f"unbalanced parentheses in to_node: {open_count} open, {close_count} close"

        # Check root function is valid
        root_func = to_node.split("(")[0]
        all_funcs = VALID_BOOL_FUNCTIONS | VALID_REAL_FUNCTIONS
        if root_func not in all_funcs and root_func not in ("EphReal", "EphInt"):
            return False, f"unknown root function in structural mutation: {root_func}"

    return True, "structural mutation (full validation at tree rebuild)"


def validate_mutation(mutation: TreeMutation,
                      condition: str = "scalar-only") -> Tuple[bool, str]:
    """Validate a single mutation against grammar rules.

    Args:
        mutation: the proposed mutation
        condition: GP condition ("scalar-only", "probes-only", "emb-only", "real-l1")
                   determines which terminal set is valid

    Returns (is_valid, reason).
    """
    if mutation.mutation_type == MutationType.TERMINAL_SWAP:
        return validate_terminal_swap(mutation, condition=condition)
    elif mutation.mutation_type == MutationType.FUNCTION_SWAP:
        return validate_function_swap(mutation)
    elif mutation.mutation_type == MutationType.EPHEMERAL_PERTURBATION:
        return validate_ephemeral_perturbation(mutation)
    elif mutation.mutation_type == MutationType.STRUCTURAL:
        return validate_structural(mutation)
    else:
        return False, f"unknown mutation type: {mutation.mutation_type}"


# ---------------------------------------------------------------------------
# Tree modification (apply mutation to s-expression)
# ---------------------------------------------------------------------------

def apply_terminal_swap(sexpr: str, from_term: str, to_term: str) -> str:
    """Replace first occurrence of from_term with to_term in s-expression.

    Uses word-boundary matching to avoid partial replacements
    (e.g., replacing "ATM_IV" inside "ATM_IV_5m").
    """
    pattern = r'\b' + re.escape(from_term) + r'\b'
    return re.sub(pattern, to_term, sexpr, count=1)


def apply_ephemeral_perturbation(sexpr: str, from_val: str, to_val: str) -> str:
    """Replace ephemeral constant value."""
    return sexpr.replace(from_val, to_val, 1)


def apply_mutation_to_sexpr(sexpr: str, mutation: TreeMutation) -> Optional[str]:
    """Apply a validated mutation to an s-expression string.

    Returns modified s-expression, or None if application fails.
    """
    if mutation.mutation_type == MutationType.TERMINAL_SWAP:
        result = apply_terminal_swap(sexpr, mutation.from_node, mutation.to_node)
        if result == sexpr:
            return None  # no change made (from_term not found)
        return result

    elif mutation.mutation_type == MutationType.EPHEMERAL_PERTURBATION:
        result = apply_ephemeral_perturbation(sexpr, mutation.from_node, mutation.to_node)
        if result == sexpr:
            return None
        return result

    elif mutation.mutation_type == MutationType.FUNCTION_SWAP:
        # Replace function name (first occurrence)
        result = sexpr.replace(mutation.from_node + "(", mutation.to_node + "(", 1)
        if result == sexpr:
            return None
        return result

    elif mutation.mutation_type == MutationType.STRUCTURAL:
        # Structural mutations need manual application — return the to_node
        # as the new subtree (caller must integrate it into the tree)
        # For now, if to_node is a complete s-expression, wrap the existing tree
        if mutation.to_node.startswith("AND(") or mutation.to_node.startswith("OR("):
            # Wrapping: AND(existing, new_condition)
            # Extract the new condition from to_node
            return f"AND({sexpr}, {mutation.to_node.split(',', 1)[1].rstrip(')')})"
        return None  # can't apply structural mutation automatically

    return None


# ---------------------------------------------------------------------------
# Statistical validation (Wilcoxon signed-rank)
# ---------------------------------------------------------------------------

def wilcoxon_test(
    original_sharpes: List[float],
    modified_sharpes: List[float],
    alpha: float = 0.10,
) -> Tuple[bool, float, float]:
    """Wilcoxon signed-rank test: is modified significantly better than original?

    Args:
        original_sharpes: per-fold val Sharpe of original strategy
        modified_sharpes: per-fold val Sharpe of modified strategy
        alpha: significance threshold (pre-registered: 0.10, one-sided)

    Returns:
        (significant, p_value, mean_improvement)
    """
    from scipy import stats

    orig = np.array(original_sharpes)
    mod = np.array(modified_sharpes)
    diffs = mod - orig

    mean_improvement = float(np.mean(diffs))

    # Need at least 3 non-zero differences for Wilcoxon
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) < 3:
        # Too few differences — can't run test
        return False, 1.0, mean_improvement

    # One-sided test: is modified > original?
    try:
        stat, p_one_sided = stats.wilcoxon(
            nonzero_diffs, alternative="greater"
        )
        p_value = float(p_one_sided)
    except ValueError:
        # All differences are zero or test can't be computed
        return False, 1.0, mean_improvement

    significant = p_value < alpha
    return significant, p_value, mean_improvement
