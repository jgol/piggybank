"""Strategy Agents - instructions and templates for the strategy creation pipeline."""

# Import instructions for each of the agents from instructions file

from .instructions import (
    SPEC_AGENT_INSTRUCTIONS,
    CODER_AGENT_INSTRUCTIONS,
    EXEC_AGENT_INSTRUCTIONS,
)
from .templates import QC_REFERENCE_TEMPLATE

__all__ = [
    "SPEC_AGENT_INSTRUCTIONS",
    "CODER_AGENT_INSTRUCTIONS",
    "EXEC_AGENT_INSTRUCTIONS",
    "QC_REFERENCE_TEMPLATE",
]
