"""Utilities package for the strategy pipeline."""

from .mcp_connection import QCMCPConnection, ToolInfo
from .parsing import (
    extract_python_code,
    extract_compile_errors,
    #prepare_code_for_json,
    truncate_text,
)
from .prompts import (
    build_code_prompt,
    build_compile_retry_prompt,
    build_zero_trades_prompt,
    build_exec_prompt,
)

__all__ = [
    # MCP Connection
    "QCMCPConnection",
    "ToolInfo",
    # Parsing
    "extract_python_code",
    "extract_compile_errors",
    #"prepare_code_for_json",
    "truncate_text",
    # Prompts
    "build_code_prompt",
    "build_compile_retry_prompt",
    "build_zero_trades_prompt",
    "build_exec_prompt",
]