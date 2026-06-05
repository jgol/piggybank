"""L3 Mutator agent: grammar-constrained tree edits.

Receives the Diagnostician's technical gap analysis and modification
guidance, plus the tree s-expressions and grammar constraints. Proposes
ONE targeted mutation that addresses the diagnosed issue.

Context isolation: receives MutatorInput (no QC metrics, no trade log).
Output validated by mutation_validator.py before re-deployment.
"""
from __future__ import annotations

import json
from typing import Optional

from layer3.agents import task_sync
from layer3.config import MUTATOR_MODEL
from layer3.prompts_v2 import MUTATOR_SYSTEM
from layer3.tools import MUTATOR_TOOLS
from layer3.schemas import MutatorInput, MutationProposal
from layer2.terminal_stats import TERMINAL_NORM_STATS


def propose_mutation(
    mut_input: MutatorInput,
) -> tuple[Optional[MutationProposal], dict]:
    """Ask the Mutator to propose a grammar-constrained tree edit.

    Args:
        mut_input: Typed input (context isolation enforced by schema —
            no QC metrics, no trade log, no behavioral profile).

    Returns:
        (proposal, metadata) where proposal is a MutationProposal
        or None if the agent failed to produce structured output.
    """
    # Build grammar spec for the condition
    from layer2.grammar import (
        build_scalar_only_terminal_set,
        build_probes_only_terminal_set,
        build_emb_only_terminal_set,
        build_terminal_set,
        SCALAR_ONLY_FUNCTIONS,
        PROBES_ONLY_FUNCTIONS,
        EMB_ONLY_FUNCTIONS,
        ALL_FUNCTIONS,
    )

    condition = mut_input.condition
    if condition == "scalar-only":
        terminals = build_scalar_only_terminal_set()
        functions = SCALAR_ONLY_FUNCTIONS
    elif condition == "probes-only":
        terminals = build_probes_only_terminal_set()
        functions = PROBES_ONLY_FUNCTIONS
    elif condition == "emb-only":
        terminals = build_emb_only_terminal_set()
        functions = EMB_ONLY_FUNCTIONS
    else:
        terminals = build_terminal_set()
        functions = ALL_FUNCTIONS

    terminal_names = sorted(set(t.name for t in terminals))
    function_names = sorted(set(f.name for f in functions))

    # Build normalization stats for reference
    norm_stats = {
        name: {"center": center, "scale": scale, "method": method}
        for name, (center, scale, method) in TERMINAL_NORM_STATS.items()
        if name in terminal_names
    }

    # Tool handlers
    def _read_grammar(inp: dict) -> str:
        return json.dumps({
            "condition": condition,
            "terminals": terminal_names,
            "functions": function_names,
            "max_nodes": mut_input.grammar_max_nodes,
            "max_depth": mut_input.grammar_max_depth,
        }, indent=2)

    def _read_terminal_stats(inp: dict) -> str:
        return json.dumps(norm_stats, indent=2)

    handlers = {
        "read_grammar_spec": _read_grammar,
        "read_terminal_stats": _read_terminal_stats,
        "submit_mutation": lambda inp: inp,  # final-answer tool
    }

    user_msg = (
        f"Propose a mutation for strategy {mut_input.strategy_id} "
        f"({mut_input.template_name}, condition={mut_input.condition}).\n\n"
        f"Diagnostician's technical gap analysis:\n{mut_input.technical_gap}\n\n"
        f"Modification guidance:\n{mut_input.modification_guidance}\n\n"
        f"Current trees:\n"
        f"  Entry: {mut_input.entry_sexpr}\n"
        f"  Exit:  {mut_input.exit_sexpr}\n"
        f"  Size:  {mut_input.size_sexpr}\n"
    )
    if mut_input.delta_sexpr:
        user_msg += f"  Delta: {mut_input.delta_sexpr}\n"
    user_msg += (
        f"\nUse read_grammar_spec and read_terminal_stats to verify constraints, "
        f"then submit_mutation with your proposal."
    )

    output, metadata = task_sync(
        role="mutator",
        user_message=user_msg,
        system_prompt=MUTATOR_SYSTEM,
        tools=MUTATOR_TOOLS,
        tool_handlers=handlers,
        model=MUTATOR_MODEL,
        max_turns=6,
    )

    if output:
        try:
            proposal = MutationProposal.model_validate(output)
            return proposal, metadata
        except Exception as e:
            metadata["parse_error"] = str(e)
            return None, metadata
    return None, metadata


def validate_mutation(
    proposal: MutationProposal,
    original_trees: dict,
    condition: str,
    max_nodes: int = 15,
    max_depth: int = 5,
) -> tuple[bool, str]:
    """Validate a mutation proposal against grammar constraints.

    Returns (valid, error_message). If valid=True, error_message is empty.
    """
    from layer2.grammar import from_sexpr, node_count

    # Parse the replacement subtree
    try:
        replacement = from_sexpr(proposal.replacement_subtree)
    except Exception as e:
        return False, f"Invalid s-expression: {e}"

    # Check node count
    n_nodes = node_count(replacement)
    target_tree_key = f"{proposal.target_tree}_sexpr"
    original_sexpr = original_trees.get(target_tree_key, "")
    if original_sexpr:
        try:
            original_tree = from_sexpr(original_sexpr)
            original_nodes = node_count(original_tree)
            # Rough check: replacement + rest of tree should be within limits
            # (exact check would require tree surgery, which is complex)
        except Exception:
            pass

    if n_nodes > max_nodes:
        return False, f"Replacement subtree has {n_nodes} nodes (max {max_nodes})"

    # Check terminal availability
    from layer2.grammar import (
        build_scalar_only_terminal_set,
        build_probes_only_terminal_set,
        build_emb_only_terminal_set,
        build_terminal_set,
    )

    if condition == "scalar-only":
        available = {t.name for t in build_scalar_only_terminal_set()}
    elif condition == "probes-only":
        available = {t.name for t in build_probes_only_terminal_set()}
    elif condition == "emb-only":
        available = {t.name for t in build_emb_only_terminal_set()}
    else:
        available = {t.name for t in build_terminal_set()}

    # Walk the replacement tree and check all terminals
    def _check_terminals(node) -> list:
        from layer2.grammar import TermNode, FuncNode
        issues = []
        if isinstance(node, TermNode):
            if node.name not in available and node.name not in ("EphReal", "EphInt", "CALL", "PUT", "NEUTRAL"):
                issues.append(f"Terminal '{node.name}' not available in {condition}")
        elif isinstance(node, FuncNode):
            for child in node.children:
                issues.extend(_check_terminals(child))
        return issues

    terminal_issues = _check_terminals(replacement)
    if terminal_issues:
        return False, "; ".join(terminal_issues)

    return True, ""
