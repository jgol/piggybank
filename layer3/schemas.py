"""L3 pipeline data structures.

Two schema generations coexist:

LEGACY (dataclasses) — used by gates.py, codegen.py, mutation_validator.py:
- StrategyPacket, FinancialReview, OptimizerOutput, RiskAssessment,
  GateResult, PipelineRecord, Decision, MutationType, TreeSlot

NEW (Pydantic BaseModel) — used by the multi-agent pipeline:
- StrategyAssessment, MutationProposal, BacktestResult, DeploymentError,
  DiagnosticianInput, MutatorInput, PortfolioAssessment,
  PipelineState, StrategyLedgerEntry, ProxyMetrics, QCMetrics, TradeRecord

Legacy schemas are retained for backward compatibility. New schemas use
Pydantic BaseModel for structured output via Anthropic tool-use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Decision(Enum):
    """LLM stage decision outcomes."""
    PROCEED = "proceed"    # pass to next stage unchanged
    MODIFY = "modify"      # needs Strategy Optimizer mutations
    REJECT = "reject"      # removed from pipeline
    DEPLOY = "deploy"      # approved for QC deployment
    REVIEW = "review"      # flag for manual human review


class MutationType(Enum):
    """Grammar-constrained mutation classes (Section 6.1 of spec)."""
    EPHEMERAL_PERTURBATION = "ephemeral_perturbation"  # adjust EphReal/EphInt values
    FUNCTION_SWAP = "function_swap"                     # GT → CrossAbove (same signature)
    TERMINAL_SWAP = "terminal_swap"                     # ATM_IV → VIXSpot (same GType)
    STRUCTURAL = "structural"                           # insert AND/OR, prune subtree


class TreeSlot(Enum):
    """Which tree in the strategy to mutate."""
    ENTRY = "entry"
    EXIT = "exit"
    SIZE = "size"
    DELTA = "delta"


# ---------------------------------------------------------------------------
# Strategy input packet (Section 6.2 of spec)
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    """Per-fold proxy evaluation metrics."""
    fold_id: int
    train_sharpe: float
    val_sharpe: float
    val_drawdown: float
    val_trade_count: int
    val_win_rate: float


@dataclass
class RegimeDecomposition:
    """Per-regime Sharpe breakdown."""
    sharpe_low_vol: float
    sharpe_mid_vol: float
    sharpe_high_vol: float
    pct_trades_low_vol: float
    pct_trades_mid_vol: float
    pct_trades_high_vol: float


@dataclass
class StrategyPacket:
    """Complete input to all LLM stages.

    Assembled from GP JSONL output + proxy re-evaluation.
    """
    # Identity
    strategy_id: str
    template_name: str
    condition: str  # "scalar-only", "probes-only", "emb-only", "real-l1"

    # Tree s-expressions
    entry_sexpr: str
    exit_sexpr: str
    size_sexpr: str
    delta_sexpr: Optional[str] = None
    # GP stop gene (evolved stop-loss base credit-multiple; 0.0 = hold-to-expiry).
    stop_mult: Optional[float] = None

    # Human-readable tree descriptions (for LLM consumption)
    entry_readable: str = ""
    exit_readable: str = ""
    size_readable: str = ""
    delta_readable: str = ""

    # Proxy metrics per fold
    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    # Aggregate proxy metrics
    mean_val_sharpe: float = 0.0
    mean_val_drawdown: float = 0.0
    total_val_trades: int = 0

    # Regime decomposition
    regime: Optional[RegimeDecomposition] = None

    # Fitness vector (as stored in JSONL)
    fitness: List[float] = field(default_factory=list)

    # Terminal usage analysis
    terminals_used: List[str] = field(default_factory=list)
    dominant_terminal: str = ""

    # QC results (populated after deployment, None before)
    qc_sharpe: Optional[float] = None
    qc_net_profit: Optional[float] = None
    qc_drawdown: Optional[float] = None
    qc_total_trades: Optional[int] = None
    qc_win_rate: Optional[float] = None
    qc_total_profit: Optional[float] = None   # sum of winning trades ($)
    qc_total_loss: Optional[float] = None      # sum of losing trades ($, negative)
    qc_total_fees: Optional[float] = None      # total exchange/broker fees ($)
    qc_avg_hold_minutes: Optional[float] = None  # average trade duration

    # Gap analysis (populated after QC comparison)
    proxy_qc_sharpe_gap: Optional[float] = None
    proxy_qc_credit_ratio: Optional[float] = None


# ---------------------------------------------------------------------------
# Stage A: Financial Analyst output
# ---------------------------------------------------------------------------

@dataclass
class FinancialReview:
    """Stage A output: economic coherence assessment."""
    decision: Decision  # PROCEED, MODIFY, or REJECT
    issues: List[str] = field(default_factory=list)
    severity: str = ""  # "none", "minor", "major", "critical"
    rationale: str = ""


# ---------------------------------------------------------------------------
# Stage B: Strategy Optimizer output
# ---------------------------------------------------------------------------

@dataclass
class TreeMutation:
    """A single grammar-constrained mutation proposal."""
    mutation_type: MutationType
    target_tree: TreeSlot  # which tree to mutate
    location: str = ""     # path in tree (e.g., "entry.right.left")
    from_node: str = ""    # current node/value
    to_node: str = ""      # proposed replacement
    rationale: str = ""    # financial reasoning for the change

    # Validation results (populated after grammar check + Wilcoxon)
    grammar_valid: Optional[bool] = None
    wilcoxon_p: Optional[float] = None
    sharpe_improvement: Optional[float] = None  # mean across folds
    accepted: Optional[bool] = None


@dataclass
class OptimizerOutput:
    """Stage B output: list of proposed mutations."""
    mutations: List[TreeMutation] = field(default_factory=list)
    analysis: str = ""  # LLM's reasoning about the strategy
    n_proposed: int = 0
    n_grammar_valid: int = 0
    n_statistically_valid: int = 0


# ---------------------------------------------------------------------------
# Stage C: Risk Manager output
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """Stage C output: final deploy/reject decision."""
    decision: Decision  # DEPLOY, REJECT, or REVIEW
    confidence: float = 0.0  # 0.0 to 1.0
    reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    regime_concern: str = ""  # specific regime risk if any


# ---------------------------------------------------------------------------
# Deterministic gate results
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of a deterministic gate check."""
    gate_id: str        # "D1", "D2", ..., "D8", "G5"
    passed: bool
    value: float = 0.0  # the metric value that was tested
    threshold: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Full pipeline record (audit trail)
# ---------------------------------------------------------------------------

@dataclass
class PipelineRecord:
    """Complete audit trail for one strategy through the L3 pipeline."""
    strategy_id: str
    template_name: str

    # Gate results
    gate_results: List[GateResult] = field(default_factory=list)
    passed_all_gates: bool = False

    # LLM stage outputs
    financial_review: Optional[FinancialReview] = None
    optimizer_output: Optional[OptimizerOutput] = None
    risk_assessment: Optional[RiskAssessment] = None

    # Modified tree (after Stage B, if applicable)
    modified_entry_sexpr: Optional[str] = None
    modified_exit_sexpr: Optional[str] = None
    modified_size_sexpr: Optional[str] = None

    # Final decision
    final_decision: Decision = Decision.REJECT
    final_reason: str = ""

    # Timing
    total_llm_calls: int = 0
    total_llm_tokens: int = 0
    total_wall_time_s: float = 0.0


# ===========================================================================
# NEW SCHEMAS (Pydantic BaseModel) — multi-agent pipeline
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class ProxyMetrics(BaseModel):
    """Proxy evaluation metrics passed to Diagnostician."""
    fold_sharpes: Dict[int, float] = Field(default_factory=dict)
    mean_val_sharpe: float = 0.0
    val_trades: int = 0
    val_win_rate: float = 0.0
    val_drawdown: float = 0.0
    pbo_score: float = 0.0


class QCMetrics(BaseModel):
    """QC backtest metrics passed to Diagnostician."""
    sharpe: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    net_profit: float = 0.0
    total_fees: float = 0.0


class TradeRecord(BaseModel):
    """Single trade from QC backtest with exit reason."""
    time: str = ""
    reason: str = ""  # "stop_loss", "signal", "eod", "max_hold"
    bars_held: int = 0
    entry_credit: float = 0.0


# ---------------------------------------------------------------------------
# QC Operator output
# ---------------------------------------------------------------------------

class BacktestResult(BaseModel):
    """Structured output from QC Operator on successful deployment."""
    strategy_id: str
    project_id: int = 0
    backtest_id: str = ""
    sharpe: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    net_profit: float = 0.0
    total_fees: float = 0.0
    trade_log: List[TradeRecord] = Field(default_factory=list)


class DeploymentError(BaseModel):
    """Structured output from QC Operator on deployment failure."""
    strategy_id: str
    error_type: Literal["COMPILE_ERROR", "RUNTIME_ERROR", "TIMEOUT"]
    message: str = ""
    retryable: bool = False


# ---------------------------------------------------------------------------
# Diagnostician: typed input (context isolation) + output
# ---------------------------------------------------------------------------

class DiagnosticianInput(BaseModel):
    """What the Diagnostician receives. Structurally excludes raw trees.

    Context isolation: no entry_sexpr, exit_sexpr, size_sexpr, delta_sexpr
    fields. The Diagnostician reasons about financial semantics via the
    plain-language interpretation, not tree syntax.

    The Diagnostician DOES receive proxy_metrics, qc_metrics, and trade_log
    because gap analysis requires comparing proxy vs QC data. The isolation
    is about tree syntax (the Mutator's domain), not financial data.
    """
    strategy_id: str
    template_name: str
    interpretation: str  # plain language from strategy_interpreter
    proxy_metrics: ProxyMetrics
    qc_metrics: QCMetrics
    trade_log: List[TradeRecord] = Field(default_factory=list)
    memory_context: str = ""


class StrategyAssessment(BaseModel):
    """Diagnostician output: three-layer assessment.

    Returned via submit_assessment final-answer tool. Numerical fields
    (gap_severity, max_consecutive_losses, profit_concentration) are
    cross-checked deterministically by the supervisor against the actual
    data — hallucinated values trigger a re-call.
    """
    # Layer 1: Technical gap
    technical_gap: str = ""
    gap_severity: float = Field(0.0, ge=0.0, le=1.0)

    # Layer 2: Behavioral profile
    behavioral_profile: str = ""
    max_consecutive_losses: int = 0
    profit_concentration: float = Field(0.0, ge=0.0, le=1.0)
    regime_vulnerability: str = ""

    # Layer 3: Tradeability
    tradeability: str = ""
    tradeability_score: float = Field(0.5, ge=0.0, le=1.0)

    # Decision
    action: Literal["DISCARD", "MODIFY", "APPROVE"]
    modification_guidance: Optional[str] = None
    risk_warnings: List[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Mutator: typed input (context isolation) + output
# ---------------------------------------------------------------------------

class MutatorInput(BaseModel):
    """What the Mutator receives. Structurally excludes QC numbers.

    Context isolation: no qc_sharpe, qc_trades, trade_log, behavioral_profile
    fields. The Mutator works from the Diagnostician's technical gap analysis
    and modification guidance only.
    """
    strategy_id: str
    template_name: str
    condition: str
    entry_sexpr: str
    exit_sexpr: str
    size_sexpr: str
    delta_sexpr: Optional[str] = None
    stop_mult: Optional[float] = None  # GP stop gene; 0.0 = hold-to-expiry
    technical_gap: str = ""
    modification_guidance: str = ""
    available_terminals: List[str] = Field(default_factory=list)
    grammar_max_nodes: int = 15
    grammar_max_depth: int = 5


class MutationProposal(BaseModel):
    """Mutator output: grammar-constrained tree edit.

    Returned via submit_mutation final-answer tool. Validated by
    mutation_validator.py before re-deployment.
    """
    target_tree: Literal["entry", "exit", "size", "delta"]
    node_path: str = ""
    original_subtree: str = ""
    replacement_subtree: str = ""
    rationale: str = ""


# ---------------------------------------------------------------------------
# Pipeline state and ledger
# ---------------------------------------------------------------------------

class StrategyLedgerEntry(BaseModel):
    """Tracks one strategy through the pipeline."""
    strategy_id: str
    template_name: str
    status: Literal[
        "pending", "deploying", "analyzing", "modifying",
        "approved", "discarded", "review"
    ] = "pending"
    iteration: int = 0  # mutation iteration count (max 2)
    qc_results: List[dict] = Field(default_factory=list)
    assessments: List[dict] = Field(default_factory=list)
    mutations: List[dict] = Field(default_factory=list)
    final_reason: str = ""


class PipelineState(BaseModel):
    """Full pipeline state, serialized after every strategy completion."""
    batch_id: str
    gp_results_dir: str = ""
    started_at: str = ""
    ledger: Dict[str, StrategyLedgerEntry] = Field(default_factory=dict)
    qc_slot_1: Optional[str] = None
    qc_slot_2: Optional[str] = None
    queue: List[str] = Field(default_factory=list)
    completed: List[str] = Field(default_factory=list)
    approved: List[str] = Field(default_factory=list)
    discarded: List[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio Manager output
# ---------------------------------------------------------------------------

class PortfolioAssessment(BaseModel):
    """Portfolio Manager output: portfolio-level assessment."""
    n_strategies: int = 0
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    terminal_concentration: Dict[str, int] = Field(default_factory=dict)
    regime_exposure: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    directional_balance: str = ""
    portfolio_sharpe: float = 0.0
    portfolio_max_drawdown: float = 0.0
    portfolio_calmar: float = 0.0
    recommended_allocation: Dict[str, float] = Field(default_factory=dict)
    kill_criteria: Dict[str, str] = Field(default_factory=dict)
    diversification_warnings: List[str] = Field(default_factory=list)
    proceed_recommendation: bool = False
