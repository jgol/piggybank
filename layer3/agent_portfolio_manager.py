"""L3 Portfolio Manager agent: LLM assessment of approved strategy portfolio.

Takes the deterministic portfolio computations (correlation, concentration,
drawdown simulation) and produces allocation recommendations, kill criteria,
and diversification warnings via the Portfolio Manager LLM agent.
"""
from __future__ import annotations

import json
from typing import Optional

from layer3.agents import task_sync
from layer3.config import PORTFOLIO_MANAGER_MODEL
from layer3.prompts_v2 import PORTFOLIO_MANAGER_SYSTEM
from layer3.schemas import PortfolioAssessment


# Portfolio Manager uses a single tool: submit_portfolio_assessment
PORTFOLIO_MANAGER_TOOLS = [
    {
        "name": "submit_portfolio_assessment",
        "description": (
            "FINAL ANSWER: Submit your portfolio assessment with allocation "
            "recommendations, kill criteria, and proceed/wait decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommended_allocation": {
                    "type": "object",
                    "description": "strategy_id → weight (0-1, must sum to ~1)"
                },
                "kill_criteria": {
                    "type": "object",
                    "description": "strategy_id → removal condition string"
                },
                "diversification_warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "proceed_recommendation": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": ["recommended_allocation", "proceed_recommendation", "rationale"],
        },
    },
]


def assess_portfolio(
    deterministic_assessment: PortfolioAssessment,
) -> tuple[Optional[PortfolioAssessment], dict]:
    """Run the Portfolio Manager agent on deterministic assessment data.

    Takes the pre-computed metrics (correlation, concentration, drawdown)
    and asks the LLM to produce allocation, kill criteria, and
    proceed/wait recommendation.

    Returns updated PortfolioAssessment with LLM fields populated.
    """
    # Build user message from deterministic assessment
    user_msg = (
        f"Assess this portfolio of {deterministic_assessment.n_strategies} approved strategies.\n\n"
        f"Correlation matrix:\n{json.dumps(deterministic_assessment.correlation_matrix, indent=2)}\n\n"
        f"Terminal concentration:\n{json.dumps(deterministic_assessment.terminal_concentration, indent=2)}\n\n"
        f"Directional balance: {deterministic_assessment.directional_balance}\n\n"
        f"Portfolio Sharpe: {deterministic_assessment.portfolio_sharpe:.3f}\n"
        f"Portfolio max drawdown: {deterministic_assessment.portfolio_max_drawdown:.4f}\n"
        f"Portfolio Calmar: {deterministic_assessment.portfolio_calmar:.3f}\n\n"
        f"Existing warnings: {deterministic_assessment.diversification_warnings}\n\n"
        f"Submit your assessment with allocation, kill criteria, and proceed/wait recommendation."
    )

    handlers = {
        "submit_portfolio_assessment": lambda inp: inp,
    }

    output, metadata = task_sync(
        role="portfolio_manager",
        user_message=user_msg,
        system_prompt=PORTFOLIO_MANAGER_SYSTEM,
        tools=PORTFOLIO_MANAGER_TOOLS,
        tool_handlers=handlers,
        model=PORTFOLIO_MANAGER_MODEL,
        max_turns=5,
    )

    if output:
        # Merge LLM recommendations into the deterministic assessment
        result = deterministic_assessment.model_copy()
        if output.get("recommended_allocation"):
            result.recommended_allocation = output["recommended_allocation"]
        if output.get("kill_criteria"):
            result.kill_criteria = output["kill_criteria"]
        if output.get("diversification_warnings"):
            result.diversification_warnings = (
                deterministic_assessment.diversification_warnings
                + output["diversification_warnings"]
            )
        if "proceed_recommendation" in output:
            result.proceed_recommendation = output["proceed_recommendation"]
        return result, metadata

    return deterministic_assessment, metadata
