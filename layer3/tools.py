"""L3 per-agent tool definitions for the Anthropic API.

Each agent gets a distinct tool subset. Tools are defined as dicts
matching the Anthropic tools=[...] schema. Tool handlers (the actual
implementations) are in tool_handlers.py or inline lambdas — this
module defines only the SCHEMAS that the model sees.

Final-answer tools (submit_assessment, submit_mutation, report_result)
signal the end of the agentic loop — their input IS the structured output.
"""
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# QC Operator tools
# ---------------------------------------------------------------------------

QC_OPERATOR_TOOLS: List[dict] = [
    {
        "name": "deploy_strategy",
        "description": (
            "Deploy a strategy to QuantConnect: create project, upload code, "
            "compile, and launch backtest. Returns project_id and backtest_id "
            "on success, or error details on failure. This is a long-running "
            "operation (10-30 minutes for 0DTE backtests)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "code_path": {"type": "string", "description": "Path to generated QC Python file"},
                "project_name": {"type": "string"},
            },
            "required": ["strategy_id", "code_path", "project_name"],
        },
    },
    {
        "name": "check_backtest_status",
        "description": (
            "Poll a running backtest for progress. Returns completion status "
            "and progress percentage. Call repeatedly until completed=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "backtest_id": {"type": "string"},
            },
            "required": ["project_id", "backtest_id"],
        },
    },
    {
        "name": "read_trade_log",
        "description": (
            "Fetch per-trade data from a completed backtest, including exit "
            "reasons (stop_loss, signal, eod, max_hold), bars held, and "
            "entry credit. Used for behavioral analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "backtest_id": {"type": "string"},
            },
            "required": ["project_id", "backtest_id"],
        },
    },
    {
        "name": "read_compile_errors",
        "description": (
            "Fetch compile error details from a failed compilation. "
            "Returns error messages with line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "integer"},
                "compile_id": {"type": "string"},
            },
            "required": ["project_id", "compile_id"],
        },
    },
    {
        "name": "report_result",
        "description": (
            "FINAL ANSWER: Submit the deployment result back to the supervisor. "
            "Call this when you have either a successful backtest result or "
            "a deployment error to report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "success": {"type": "boolean"},
                "project_id": {"type": "integer"},
                "backtest_id": {"type": "string"},
                "sharpe": {"type": "number"},
                "total_trades": {"type": "integer"},
                "win_rate": {"type": "number"},
                "max_drawdown": {"type": "number"},
                "net_profit": {"type": "number"},
                "total_fees": {"type": "number"},
                "error_type": {"type": "string", "enum": ["COMPILE_ERROR", "RUNTIME_ERROR", "TIMEOUT"]},
                "error_message": {"type": "string"},
                "retryable": {"type": "boolean"},
            },
            "required": ["strategy_id", "success"],
        },
    },
]


# ---------------------------------------------------------------------------
# Diagnostician tools
# ---------------------------------------------------------------------------

DIAGNOSTICIAN_TOOLS: List[dict] = [
    {
        "name": "read_proxy_metrics",
        "description": (
            "Fetch proxy evaluation metrics for the strategy: per-fold Sharpe, "
            "mean val Sharpe, val trades, win rate, drawdown, PBO score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "read_qc_metrics",
        "description": (
            "Fetch QC backtest metrics: Sharpe, total trades, win rate, "
            "max drawdown, net profit, total fees."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "read_trade_log",
        "description": (
            "Fetch per-trade data from QC backtest: exit reasons, bars held, "
            "entry credit. Use this to analyze loss streaks, regime-specific "
            "performance, and stop-loss behavior."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "read_strategy_interpretation",
        "description": (
            "Fetch the plain-language description of what the strategy does: "
            "entry conditions, exit conditions, size logic, delta selection. "
            "Use this to understand the strategy's financial semantics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "request_additional_data",
        "description": (
            "Request additional data from the supervisor. Use when the "
            "initial data is insufficient for diagnosis. The supervisor "
            "will dispatch the QC Operator to fetch the requested data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "request_type": {
                    "type": "string",
                    "enum": ["detailed_trade_log", "diagnostic_run", "terminal_comparison"],
                },
                "parameters": {"type": "object"},
                "reason": {"type": "string"},
            },
            "required": ["request_type", "reason"],
        },
    },
    {
        "name": "submit_assessment",
        "description": (
            "FINAL ANSWER: Submit your three-layer strategy assessment. "
            "This ends the diagnostic session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "technical_gap": {"type": "string"},
                "gap_severity": {"type": "number", "minimum": 0, "maximum": 1},
                "behavioral_profile": {"type": "string"},
                "max_consecutive_losses": {"type": "integer"},
                "profit_concentration": {"type": "number", "minimum": 0, "maximum": 1},
                "regime_vulnerability": {"type": "string"},
                "tradeability": {"type": "string"},
                "tradeability_score": {"type": "number", "minimum": 0, "maximum": 1},
                "action": {"type": "string", "enum": ["DISCARD", "MODIFY", "APPROVE"]},
                "modification_guidance": {"type": "string"},
                "risk_warnings": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["technical_gap", "behavioral_profile", "tradeability", "action", "confidence"],
        },
    },
]


# ---------------------------------------------------------------------------
# Mutator tools
# ---------------------------------------------------------------------------

MUTATOR_TOOLS: List[dict] = [
    {
        "name": "read_grammar_spec",
        "description": (
            "Fetch the grammar specification for this condition: available "
            "terminals, functions, type constraints, max nodes, max depth."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {"type": "string"},
            },
            "required": ["condition"],
        },
    },
    {
        "name": "read_terminal_stats",
        "description": (
            "Fetch normalization constants (center, scale, method) for "
            "all terminals. Use to understand the raw-scale meaning of "
            "EphReal threshold values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "submit_mutation",
        "description": (
            "FINAL ANSWER: Submit your grammar-constrained tree mutation "
            "proposal. The proposal will be validated against the grammar "
            "before acceptance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_tree": {"type": "string", "enum": ["entry", "exit", "size", "delta"]},
                "node_path": {"type": "string"},
                "original_subtree": {"type": "string", "description": "S-expression of current subtree"},
                "replacement_subtree": {"type": "string", "description": "S-expression of replacement"},
                "rationale": {"type": "string"},
            },
            "required": ["target_tree", "original_subtree", "replacement_subtree", "rationale"],
        },
    },
]


# ---------------------------------------------------------------------------
# Supervisor tools (write_todos for planning)
# ---------------------------------------------------------------------------

SUPERVISOR_TOOLS: List[dict] = [
    {
        "name": "write_todos",
        "description": (
            "Create or update the batch task list. Use this to plan which "
            "candidates to evaluate, track progress, and record decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "done", "blocked"]},
                            "notes": {"type": "string"},
                        },
                        "required": ["task", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helper: get tools by role
# ---------------------------------------------------------------------------

def get_tools_for_role(role: str) -> List[dict]:
    """Return the tool definitions for a given agent role."""
    from layer3.agent_portfolio_manager import PORTFOLIO_MANAGER_TOOLS
    role_tools = {
        "supervisor": SUPERVISOR_TOOLS,
        "qc_operator": QC_OPERATOR_TOOLS,
        "diagnostician": DIAGNOSTICIAN_TOOLS,
        "mutator": MUTATOR_TOOLS,
        "portfolio_manager": PORTFOLIO_MANAGER_TOOLS,
    }
    tools = role_tools.get(role)
    if tools is None:
        raise ValueError(f"Unknown role: {role}. Must be one of {list(role_tools.keys())}")
    return tools
