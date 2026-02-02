"""Configuration constants for the strategy pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# Model Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

# Agent Limits
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "40"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "800"))

# Retry Limits
MAX_COMPILE_ATTEMPTS = int(os.getenv("MAX_COMPILE_ATTEMPTS", "3"))
MAX_REVISION_ATTEMPTS = int(os.getenv("MAX_REVISION_ATTEMPTS", "3"))

# QuantConnect Project Settings
DEFAULT_PROJECT_NAME = os.getenv("QC_PROJECT_NAME", "SPX_0DTE_Strategy")
DEFAULT_MAIN_FILE = "main.py"

# Relevant MCP tools
QC_TOOLS = [
    "create_project",
    "read_project", 
    "update_project",
    "delete_project",
    "create_file",
    "read_file",
    "update_file_contents",
    "create_compile",
    "read_compile",
    "create_backtest",
    "read_backtest",
    "read_backtest_orders",
    "read_backtest_insights",
]