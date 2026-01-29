


"""
Incremental QuantConnect MCP + OpenAI Agents SDK controller.

What it does:
1) Connects to QuantConnect MCP server (via Docker stdio transport) and lists tools.
2) Filters tools to a minimal allowlist (your selected tools + a few recommended).
3) Runs a single controller Agent (OpenAI Agents SDK) that:
   - first produces a verbal strategy spec (no tools)
   - then generates QuantConnect code
   - then uses MCP tools to create/update project files, compile, backtest, and read results.

Requirements:
- pip install openai-agents mcp python-dotenv
- Docker installed and able to run quantconnect/mcp-server
- Environment variables:
  OPENAI_API_KEY
  QUANTCONNECT_USER_ID
  QUANTCONNECT_API_TOKEN
Optional:
  OPENAI_MODEL
  DOCKER_PLATFORM (e.g., linux/arm64)
  MAX_AGENT_TURNS
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents import Agent, Runner

# Tool wrapper decorator (import path varies a bit by version)
try:
    from agents import function_tool
except Exception:
    # Some installs expose it under agents.tool or similar; adjust if needed.
    # If this fails in your env, paste your `pip show openai-agents` and I’ll tailor the import.
    from agents.tool import function_tool  # type: ignore


# ----------------------------
# Config
# ----------------------------

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # pick a known model; override via env
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "30"))

# Necessary QC MCP tools (your baseline list)
QC_TOOLS = [
    "create_project",
    "read_project",
    "update_project",
    "delete_project",
    "create_file",
    "read_compile",
    "read_file",
    "update_file_contents",
    "create_compile",
    "create_backtest",
    "read_backtest",
    "read_backtest_orders",
    "read_backtest_insights"
]

# Max compile retries
MAX_COMPILE_ATTEMPTS = 3


# ----------------------------
# MCP Connection
# ----------------------------

@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: Dict[str, Any]

class QC_MCP_Connection:
    """
    Connect to QuantConnect MCP server over stdio (Docker).
    """

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._stdio_ctx = None
        self._sess_ctx = None

        self.tools: List[ToolInfo] = []
        self.tools_by_name: Dict[str, ToolInfo] = {}

    async def __aenter__(self):
        user_id = os.getenv("QUANTCONNECT_USER_ID")
        api_token = os.getenv("QUANTCONNECT_API_TOKEN")
        if not user_id or not api_token:
            raise RuntimeError("Missing QUANTCONNECT_USER_ID or QUANTCONNECT_API_TOKEN env vars")

        docker_platform = os.getenv("DOCKER_PLATFORM")  # optional

        args = ["run", "-i", "--rm"]
        if docker_platform:
            args += ["--platform", docker_platform]

        args += [
            "-e", f"QUANTCONNECT_USER_ID={user_id}",
            "-e", f"QUANTCONNECT_API_TOKEN={api_token}",
            "quantconnect/mcp-server",
        ]

        server_params = StdioServerParameters(
            command="docker",
            args=args,
        )

        self._stdio_ctx = stdio_client(server_params)
        read, write = await self._stdio_ctx.__aenter__()

        self._sess_ctx = ClientSession(read, write)
        self.session = await self._sess_ctx.__aenter__()
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = [
            ToolInfo(
                name=t.name,
                description=(t.description or ""),
                input_schema=(t.inputSchema or {}),
            )
            for t in response.tools
        ]
        self.tools_by_name = {t.name: t for t in self.tools}
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Defensive teardown: avoid masking original errors during shutdown
        for ctx in (self._sess_ctx, self._stdio_ctx):
            if ctx is None:
                continue
            try:
                await ctx.__aexit__(exc_type, exc, tb)
            except asyncio.CancelledError:
                # Cancellation should propagate
                raise
            except Exception:
                # Broken pipe / reset / taskgroup shutdown noise is common here
                pass

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self.session:
            raise RuntimeError("MCP session not initialized")

        result = await self.session.call_tool(name, arguments=arguments)

        if not result.content:
            return ""

        chunks: List[str] = []
        for block in result.content:
            # Prefer text if present
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
                continue

            # Fallback: try common JSON-ish attributes
            # Many MCP clients expose either `.data` or `.json` or a `.type` with a payload
            data = getattr(block, "data", None)
            if data is not None:
                try:
                    chunks.append(json.dumps(data))
                    continue
                except Exception:
                    pass

            # As a last resort, stringify the block
            try:
                # Some blocks support `model_dump_json` (pydantic)
                if hasattr(block, "model_dump_json"):
                    chunks.append(block.model_dump_json())
                else:
                    chunks.append(str(block))
            except Exception:
                pass

        return "\n".join(chunks)

# ----------------------------
# Tool Filtering
# ----------------------------

def build_tool_bank(mcp: QC_MCP_Connection) -> List[str]:
    """
    Returns (allowed_tool_names, missing_tool_names).
    Only tools that exist on the MCP server are allowed.
    """
    relevant_tools = QC_TOOLS
    located: List[str] = []

    print("\nTOOLS\n")
    #for t in mcp.tools_by_name:
    #    print(t)

    for t in relevant_tools:
        if t in mcp.tools_by_name:
            located.append(t)
        else:
            print(f"Tool not found in MCP: {t}")

    return located


def tool_cards(mcp: QC_MCP_Connection, valid_tools: List[str]) -> List[Dict[str, Any]]:
    """
    Compact tool metadata the agent can read to decide which tool + args to use next.
    """
    cards: List[Dict[str, Any]] = []
    for name in valid_tools:
        info = mcp.tools_by_name[name]
        cards.append({
            "name": info.name,
            "description": info.description,
            "input_schema": info.input_schema,
        })
    return cards

# ----------------------------
# OpenAI Agents SDK tools (MUST be Tool objects)
# ----------------------------

def make_agent_tools(mcp: QC_MCP_Connection, tools: List[str]):

    @function_tool
    async def qc_get_tools() -> str:
        """
        List the allowlisted QuantConnect MCP tools and their JSON input schemas.
        Use this first to determine the required fields for each call.
        """
        return json.dumps({"available_tools": tool_cards(mcp, tools)})

    @function_tool
    async def qc_call_tool(tool_name: str, arguments_json: str) -> str:
        """
        Execute a QuantConnect MCP tool by name.

        Parameters:
        - tool_name: str
            One of the allowed MCP tools (e.g., 'create_project', 'create_file',
            'create_compile', 'create_backtest', 'read_backtest').
        - arguments_json: str
            A JSON string (use double-quotes) with exactly the fields required by that tool's input schema.
            Obtain the schema via qc_get_tools().

        Returns:
            JSON string:
              { "ok": bool, "tool": str, "raw": str, "data": object|null, "error"?: str }
        """
        if tool_name not in tools:
            return json.dumps({
                "ok": False,
                "tool": tool_name,
                "error": f"Tool '{tool_name}' is not in allowlist.",
                "allowed_tools": tools,
            })

        # Parse JSON arguments expected by MCP
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments_json must decode to a JSON object (dict).")
        except Exception as e:
            return json.dumps({
                "ok": False,
                "tool": tool_name,
                "error": f"Invalid arguments_json: {e}",
                "arguments_json": arguments_json,
            })

        try:
            raw = await mcp.call_tool(tool_name, arguments)
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None

            return json.dumps({
                "ok": True,
                "tool": tool_name,
                "raw": raw,
                "data": parsed,
            })
        except Exception as e:
            return json.dumps({
                "ok": False,
                "tool": tool_name,
                "error": str(e),
            })

    return [qc_get_tools, qc_call_tool]

# ----------------------------
# Agent Instructions (Controller)
# ----------------------------

CONTROLLER_INSTRUCTIONS = f"""
You are a QuantConnect strategy controller.

Your job has TWO phases:

PHASE 1 (no tools, no code):
- First produce a VERBAL strategy specification (no code yet), structured as bullets:
  - Hypothesis / intuition
  - Instrument(s) + data requirements
  - Entry conditions
  - Exit conditions
  - Position sizing
  - Risk management rules
  - Schedule (times)
  - Constraints (min credit, delta selection, spread width, VIX filter, etc.)
  - Failure modes and mitigations

PHASE 2 (implementation + tools):
- Then implement the strategy as complete QuantConnect Python algorithm code (QCAlgorithm),
  and use the provided MCP tools (via qc_call_tool) to create/update the project files,
  compile, backtest, and read backtest results. Make sure that the complete final code appears and updates in QuantConnect project
- Do not wait for the instructions, execute the necessary tools
- If read_backtest shows 0 trades or errors, you MUST either adjust code or parameters and repeat the tool sequence (update_file_contents → create_compile → read_compile → create_backtest → read_backtest) until either trades occur or MAX_COMPILE_ATTEMPTS is exhausted.
- communicate the final assessment of the backtests

Tool access:
- qc_get_tools(): returns allowed tools and their JSON input schemas
- qc_call_tool(tool_name, arguments): executes an allowed MCP tool

Hard rules:
- Do NOT call MCP tools until you have produced the strategy spec (PHASE 1) AND written the compilable code.
- Use qc_get_tools() to learn required arguments before calling a tool.
- Maintain state: projectId, compileId, backtestId, etc.
- If compilation fails, revise code and try again, up to {MAX_COMPILE_ATTEMPTS} compile attempts.
- Stop once you have backtest results (or you cannot progress).

QuantConnect coding rules:
- Use: from AlgorithmImports import *
- class inherits QCAlgorithm
- Implement Initialize() and OnData()
- Use correct OptionChain access patterns for QuantConnect:
  - OptionChains accessed by the canonical option symbol returned by AddIndexOption/AddOption
- Scheduled entry must not rely on CurrentSlice:
- Cache the latest option chain in OnData and use that cached chain in scheduled handlers.
- Create a Python algorithm file with a 0DTE options strategy on SPX
- The strategy should enter positions daily, trade SPX options with 0DTE expiration
- Include proper risk management and exit logic
- Strictly adhere to QuantConnect API and 0DTE rules


Output requirements:
- Section A: Strategy Spec (structured bullets)
- Section B: Final QuantConnect code (single python code block)
- Section C: Tool execution summary (projectId, compileId, backtestId, and key outcomes)
"""

# ----------------------------
# Main
# ----------------------------

async def main():
    strategy_description = """
Create a 0DTE (zero days to expiration) trading strategy on QuantConnect.

Create a 0DTE iron condor strategy on SPX that enters daily at 10:00 AM ET by selling the 0.15 delta put and call with 2-strike-wide wings, collects a minimum $0.40 credit, exits at 50% profit or 2x loss or 3:45 PM ET, skips trading when VIX is above 30, and risks no more than 2% of portfolio per trade.
""".strip()

    async with QC_MCP_Connection() as mcp:
        print(f"Connected. MCP exposed {len(mcp.tools)} tools.")

        filtered_tools = build_tool_bank(mcp)

        print("\nFitered tools present in MCP server:")
        for t in filtered_tools:
            print("  -", t)

        agent_tools = make_agent_tools(mcp, filtered_tools)

        # Sanity check: tool objects must have .name
        for t in agent_tools:
            print("Agent tool:", getattr(t, "name", None), type(t))

        agent = Agent(
            name="QC_Controller",
            instructions=CONTROLLER_INSTRUCTIONS,
            model=OPENAI_MODEL,
            tools=agent_tools,
        )

        result = await Runner.run(
            agent,
            input=strategy_description,
            max_turns=MAX_AGENT_TURNS,
        )

        print("\n" + "=" * 80)
        print("FINAL OUTPUT")
        print("=" * 80)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
