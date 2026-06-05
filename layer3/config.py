"""L3 pipeline configuration — externalized constants.

All hardcoded values from across L3 modules centralized here.
Import from this module instead of using magic numbers.
"""

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
MAX_MUTATION_ITERATIONS = 2
COST_BUDGET_USD = 50.0

# Deterministic thresholds for APPROVE override
# (Diagnostician cannot approve if these fail)
APPROVE_MIN_QC_SHARPE = 0.0
APPROVE_MIN_QC_TRADES = 50
APPROVE_MAX_QC_DRAWDOWN = 0.15

# ---------------------------------------------------------------------------
# QC Scheduler
# ---------------------------------------------------------------------------
QC_POLL_INTERVAL_S = 15
QC_STAGNANT_WARN_S = 600       # 10 minutes no progress → warning
QC_BACKTEST_TIMEOUT_S = 2700   # 45 minutes → abort
QC_COMPILE_TIMEOUT_S = 300     # 5 minutes for compilation
QC_TRANSIENT_RETRY_WAIT_S = 60
QC_RATE_LIMIT_PAUSE_S = 300
QC_MAX_TRANSIENT_RETRIES = 3

# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------
HITL_TIMEOUT_HOURS = 24

# ---------------------------------------------------------------------------
# Agent models
# ---------------------------------------------------------------------------
SUPERVISOR_MODEL = "claude-sonnet-4-20250514"
QC_OPERATOR_MODEL = "claude-sonnet-4-20250514"
DIAGNOSTICIAN_MODEL = "claude-sonnet-4-20250514"
MUTATOR_MODEL = "claude-opus-4-20250514"
PORTFOLIO_MANAGER_MODEL = "claude-sonnet-4-20250514"

# Cost estimation (USD per 1K tokens, approximate)
COST_PER_1K_INPUT = {
    SUPERVISOR_MODEL: 0.003,
    QC_OPERATOR_MODEL: 0.003,
    DIAGNOSTICIAN_MODEL: 0.003,
    MUTATOR_MODEL: 0.015,
    PORTFOLIO_MANAGER_MODEL: 0.003,
}
COST_PER_1K_OUTPUT = {
    SUPERVISOR_MODEL: 0.015,
    QC_OPERATOR_MODEL: 0.015,
    DIAGNOSTICIAN_MODEL: 0.015,
    MUTATOR_MODEL: 0.075,
    PORTFOLIO_MANAGER_MODEL: 0.015,
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
DEFAULT_MAX_PER_TEMPLATE = 10
DEFAULT_BACKTEST_START = "2024-10-01"
DEFAULT_BACKTEST_END = "2026-05-22"
