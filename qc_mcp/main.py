"""
Multi-Agent Strategy Pipeline for 0DTE Options
===============================================

Orchestrates strategy development from verbal description, strategy implementation, compilation, and backtesting:
    Spec Agent → Code Agent → Exec Agent → (Revision Loop)

Usage:
    python main.py

Environment Variables:
    QUANTCONNECT_USER_ID    - QuantConnect API user ID
    QUANTCONNECT_API_TOKEN  - QuantConnect API token
    QC_PROJECT_NAME         - Project name (default: SPX_0DTE_Strategy)
    OPENAI_MODEL            - Model to use (default: gpt-5.2)
    STRATEGY_TASK           - Custom strategy description (optional)
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

from agents import Agent, Runner

try:
    from agents import function_tool
except ImportError:
    from agents.tool import function_tool

# Local imports
from config import (
    OPENAI_MODEL,
    MAX_AGENT_TURNS,
    MAX_COMPILE_ATTEMPTS,
    MAX_REVISION_ATTEMPTS,
    DEFAULT_PROJECT_NAME,
    DEFAULT_MAIN_FILE,
    AGENT_TIMEOUT,
    QC_TOOLS,
)
from strategy_agents import (
    SPEC_AGENT_INSTRUCTIONS,
    CODER_AGENT_INSTRUCTIONS,
    EXEC_AGENT_INSTRUCTIONS,
)
from utils import (
    QCMCPConnection,
    extract_python_code,
    build_code_prompt,
    build_compile_retry_prompt,
    build_zero_trades_prompt,
    build_exec_prompt,
)

load_dotenv()

# Default strategy task
DEFAULT_TASK = """Design a 0DTE SPX options strategy optimized for risk-adjusted returns.

Objectives:
- Target Sharpe ratio > 1.0
- Maximum drawdown < 15%
- Consistent daily execution

Constraints:
- Trade SPX index options expiring same day
- Entry window: 9:45 AM - 11:00 AM ET
- Must exit all positions by 3:50 PM ET
- Maximum risk: 2% of portfolio per trade
- Skip trading when VIX > 35

You choose:
- Strategy type (spreads, naked, directional, neutral)
- Strike selection method (delta, fixed offset, premium-based)
- Number of legs
- Entry triggers
- Exit conditions (profit target, stop loss, time-based)
- Position sizing approach

Justify your choices based on 0DTE characteristics: rapid theta decay, intraday volatility patterns, and gamma exposure."""


# =============================================================================
# Result Class
# =============================================================================

@dataclass
class ExecResult:
    # Structured result from execution agent.
    project_name: str = ""
    project_id: str = ""
    compile_ok: bool = False
    compile_id: str = ""
    compile_errors: list[str] = field(default_factory=list)
    backtest_ok: bool = False
    backtest_id: str = ""
    trades: int = 0
    notes: str = ""
    submitted: bool = False

    def to_dict(self) -> dict:
        return {
            "projectName": self.project_name,
            "projectId": self.project_id,
            "compileOk": self.compile_ok,
            "compileId": self.compile_id,
            "compileErrors": self.compile_errors,
            "backtestOk": self.backtest_ok,
            "backtestId": self.backtest_id,
            "trades": self.trades,
            "notes": self.notes,
        }

    def reset(self):
        # Reset for repeat execution attempts.
        self.project_name = ""
        self.project_id = ""
        self.compile_ok = False
        self.compile_id = ""
        self.compile_errors = []
        self.backtest_ok = False
        self.backtest_id = ""
        self.trades = 0
        self.notes = ""
        self.submitted = False


# =============================================================================
# Agent Tools Setup
# =============================================================================

def build_tool_bank(mcp: QCMCPConnection) -> list[str]:
    # Filter QC_TOOLS to only those available in MCP.
    return [t for t in QC_TOOLS if t in mcp.tools_by_name]


def tool_cards(mcp: QCMCPConnection, valid_tools: list[str]) -> list[dict]:
    # Generate tool documentation for agent.
    return [
        {
            "name": t,
            "description": mcp.tools_by_name[t].description,
            "input_schema": mcp.tools_by_name[t].input_schema
        }
        for t in valid_tools
    ]


def make_agent_tools(mcp: QCMCPConnection, tools: list[str], result_holder: ExecResult):
    # Create agent tools with shared result holder.
    
    @function_tool
    async def qc_get_tools() -> str:
        # ist available QuantConnect MCP tools and their JSON input schemas.
        return json.dumps({"available_tools": tool_cards(mcp, tools)})

    @function_tool
    async def qc_call_tool(tool_name: str, arguments_json: str) -> str:
        # Execute a QuantConnect MCP tool. Use qc_get_tools() first to get schemas.
        if tool_name not in tools:
            return json.dumps({
                "ok": False,
                "tool": tool_name,
                "error": f"Tool '{tool_name}' not in list",
                "available_tools": tools
            })
        
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
            if not isinstance(arguments, dict):
                raise ValueError("Must decode to dict")
        except Exception as e:
            return json.dumps({
                "ok": False,
                "tool": tool_name,
                "error": f"Invalid arguments_json: {e}"
            })
            
        try:
            raw = await mcp.call_tool(tool_name, arguments)
            data = json.loads(raw) if raw else {}
            
            # Auto-poll for backtest creation
            if tool_name == "create_backtest":
                print(f"create_backtest response: {data}")
                backtest_id = data.get("backtestId") or data.get("backtest", {}).get("backtestId")
                project_id = arguments.get("projectId")
                
                print(f"backtest_id: {backtest_id}")
                print(f"project_id: {project_id}")
                
                if not backtest_id:
                    print("No backtest_id found, returning early")
                    return json.dumps({"ok": False, "tool": tool_name, "error": "No backtestId", "data": data})
                
                print("Starting poll loop...")
                
                for i in range(30):
                    await asyncio.sleep(10)
                    print(f"Poll {i}...")
                    poll_result = await mcp.call_tool("read_backtest", {
                        "projectId": project_id,
                        "backtestId": backtest_id
                    })
                    poll_data = json.loads(poll_result) if poll_result else {}
                    status = poll_data.get("status", "")
                    print(f"Poll {i} status: {status}")
                    
                    if "Completed" in status or "Error" in status:
                        return json.dumps({"ok": True, "tool": tool_name, "data": poll_data})
                
                return json.dumps({"ok": False, "tool": tool_name, "error": "Backtest timeout"})
            
            return json.dumps({"ok": True, "tool": tool_name, "data": data})
        except Exception as e:
            print(f"Exception in qc_call_tool: {e}")
            return json.dumps({"ok": False, "tool": tool_name, "error": str(e)})


    @function_tool
    async def submit_exec_result(
        project_name: str,
        project_id: str,
        compile_ok: bool,
        compile_id: str,
        compile_errors: list[str],
        backtest_ok: bool,
        backtest_id: str,
        trades: int,
        notes: str
    ) -> str:
        #Submit final execution result. Call exactly once when finished.
        result_holder.project_name = project_name
        result_holder.project_id = project_id
        result_holder.compile_ok = compile_ok
        result_holder.compile_id = compile_id
        result_holder.compile_errors = compile_errors or []
        result_holder.backtest_ok = backtest_ok
        result_holder.backtest_id = backtest_id
        result_holder.trades = trades
        result_holder.notes = notes
        result_holder.submitted = True
        
        return json.dumps({
            "status": "Result recorded",
            "summary": result_holder.to_dict()
        })

    return [qc_get_tools, qc_call_tool, submit_exec_result]


# =============================================================================
# Agent Runner
# =============================================================================

async def run_agent(agent: Agent, input_text: str, max_turns: int, timeout: int = AGENT_TIMEOUT):
    #Run agent with a set timeout
    try:
        return await asyncio.wait_for(
            Runner.run(agent, input=input_text, max_turns=max_turns),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Agent {agent.name} timed out after {timeout}s")


# =============================================================================
# Main Orchestration
# =============================================================================

async def main():
    # Main pipeline: Spec → Code → Exec with revision loop
    
    # Configuration
    task = os.getenv("STRATEGY_TASK", DEFAULT_TASK)
    project_name = os.getenv("QC_PROJECT_NAME", DEFAULT_PROJECT_NAME)
    
    print("=" * 80)
    print("STRATEGY PIPELINE")
    print("=" * 80)
    print(f"Task: {task[:100]}...")
    print(f"Project: {project_name}")
    print(f"Model: {OPENAI_MODEL}")
    print()
    
    async with QCMCPConnection() as mcp:
        print(f"✓ MCP connected ({len(mcp.tools)} tools)")
        
        filtered_tools = build_tool_bank(mcp)
        print(f"✓ Tools: {', '.join(filtered_tools)}")
        
        # Shared result holder
        exec_result = ExecResult()
        agent_tools = make_agent_tools(mcp, filtered_tools, exec_result)
        
        # Initialize agents
        spec_agent = Agent(
            name="Spec_Agent",
            instructions=SPEC_AGENT_INSTRUCTIONS,
            model=OPENAI_MODEL,
            tools=[]
        )
        code_agent = Agent(
            name="Code_Agent",
            instructions=CODER_AGENT_INSTRUCTIONS,
            model=OPENAI_MODEL,
            tools=[]
        )
        exec_agent = Agent(
            name="Exec_Agent",
            instructions=EXEC_AGENT_INSTRUCTIONS,
            model=OPENAI_MODEL,
            tools=agent_tools
        )

        # =====================================================================
        # PHASE 1: SPEC WRITING
        # =====================================================================
        print(f"\n{'='*80}\nPHASE 1: SPECIFICATION\n{'='*80}")
        
        try:
            spec_result = await run_agent(spec_agent, task, max_turns=20)
            spec_text = spec_result.final_output
        except Exception as e:
            print(f"✗ Spec Agent failed: {e}")
            return None
        
        print(f"✓ Specification generated ({len(spec_text)} chars)")
        print(f"\n{spec_text}\n")

        # =====================================================================
        # PHASE 2: CODE GENERATION
        # =====================================================================
        print(f"\n{'='*80}\nPHASE 2: CODE GENERATION\n{'='*80}")
        
        code_prompt = build_code_prompt(spec_text)
        
        try:
            code_result = await run_agent(code_agent, code_prompt, max_turns=20)
        except Exception as e:
            print(f"✗ Code Agent failed: {e}")
            return None
        
        current_code = extract_python_code(code_result.final_output)
        if not current_code:
            print("✗ Code Agent did not produce valid Python code")
            return None
        
        print(f"✓ Code generated ({len(current_code)} chars)")
        print(f"GENERATED CODE: \n{current_code}")

        # =====================================================================
        # PHASE 3: EXECUTION & REVISION
        # =====================================================================
        print(f"\n{'='*80}\nPHASE 3: EXECUTION\n{'='*80}")
        
        revision_count = 0
        seen_error_hashes: set[str] = set()
        final_result = None

        project_id = None

        for attempt in range(1, MAX_COMPILE_ATTEMPTS + 1):
            print(f"\n--- Attempt {attempt}/{MAX_COMPILE_ATTEMPTS} "
                  f"(Revisions: {revision_count}/{MAX_REVISION_ATTEMPTS}) ---")
            
            exec_result.reset()
            exec_input = build_exec_prompt(
                project_name, 
                DEFAULT_MAIN_FILE, 
                current_code,
                project_id=project_id 
            )
            
            try:
                await run_agent(exec_agent, exec_input, max_turns=MAX_AGENT_TURNS)
            except Exception as e:
                print(f"✗ Exec Agent error: {e}")
                continue

            if not exec_result.submitted:
                print("⚠ Exec Agent did not submit results")
                continue
            
            if exec_result.project_id:
                project_id = exec_result.project_id

            # Log status
            print(f"  Project ID: {exec_result.project_id}")
            print(f"  Compile: {'✓' if exec_result.compile_ok else '✗'}")
            print(f"  Backtest: {'✓' if exec_result.backtest_ok else '✗'}")
            print(f"  Trades: {exec_result.trades}")

            # SUCCESS
            if exec_result.compile_ok and exec_result.backtest_ok and exec_result.trades > 0:
                print(f"\n{'='*80}")
                print(f"✓ SUCCESS: {exec_result.trades} trades executed")
                print(f"{'='*80}")
                final_result = exec_result.to_dict()
                break

            # Check revision budget
            if revision_count >= MAX_REVISION_ATTEMPTS:
                print(f"⚠ Max revisions ({MAX_REVISION_ATTEMPTS}) reached")
                final_result = exec_result.to_dict()
                break

            # COMPILE FAILURE
            if not exec_result.compile_ok:
                errors = exec_result.compile_errors or ["Unknown compile error"]
                error_hash = hash(tuple(sorted(errors)))
                
                if error_hash in seen_error_hashes:
                    print("⚠ Same errors repeated, stopping")
                    final_result = exec_result.to_dict()
                    break
                
                seen_error_hashes.add(error_hash)
                print(f"  Errors: {errors[:3]}{'...' if len(errors) > 3 else ''}")
                
                # Request fix
                retry_prompt = build_compile_retry_prompt(current_code, errors)
                
                try:
                    retry_result = await run_agent(code_agent, retry_prompt, max_turns=15)
                    new_code = extract_python_code(retry_result.final_output)
                    
                    if new_code and new_code != current_code:
                        current_code = new_code
                        revision_count += 1
                        print(f"  ✓ Code revised (revision #{revision_count})")
                        continue
                except Exception as e:
                    print(f"  ✗ Revision failed: {e}")
                
                continue

            # ZERO TRADES
            if exec_result.trades == 0:
                print("  ⚠ Backtest succeeded but no trades")
                
                revision_prompt = build_zero_trades_prompt(current_code, exec_result.to_dict())
                
                try:
                    revision_result = await run_agent(code_agent, revision_prompt, max_turns=20)
                    new_code = extract_python_code(revision_result.final_output)
                    
                    if new_code and new_code != current_code:
                        current_code = new_code
                        revision_count += 1
                        seen_error_hashes.clear()
                        print(f"  ✓ Logic revised (revision #{revision_count})")
                        continue
                except Exception as e:
                    print(f"  ✗ Revision failed: {e}")
                
                continue

        # Final output
        if final_result:
            print(f"\n{'='*80}\nFINAL RESULT\n{'='*80}")
            print(json.dumps(final_result, indent=2))
        else:
            print("\n✗ Pipeline failed to produce results")
        
        return final_result


if __name__ == "__main__":
    asyncio.run(main())