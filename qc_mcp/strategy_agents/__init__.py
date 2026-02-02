"""Strategy Agents - instructions and templates for the strategy creation pipeline."""

# Import instructions for each of the agents from instructions file
from .instructions import (
    SPEC_AGENT_INSTRUCTIONS,
    CODER_AGENT_INSTRUCTIONS,
    EXEC_AGENT_INSTRUCTIONS,
)

# Supplement the instructions with templates strategy generation and API references for QuantConnect to help agents produce more reliable code
from .templates import (
    QC_API_REFERENCE,
    QC_REFERENCE_TEMPLATE,
    FEW_SHOT_EXAMPLE,
)

__all__ = [
    "SPEC_AGENT_INSTRUCTIONS",
    "CODER_AGENT_INSTRUCTIONS", 
    "EXEC_AGENT_INSTRUCTIONS",
    "QC_API_REFERENCE",
    "QC_REFERENCE_TEMPLATE",
    "FEW_SHOT_EXAMPLE",
]