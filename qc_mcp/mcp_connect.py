"""QuantConnect MCP Integration."""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).parent / ".env")

# Constants
MAX_ITERATIONS = 20  # Prevent infinite loops
MODEL = "claude-sonnet-4-20250514"

def get_server_params() -> StdioServerParameters:
    """Create MCP server parameters (single source of truth)."""
    user_id = os.getenv("QUANTCONNECT_USER_ID")
    api_token = os.getenv("QUANTCONNECT_API_TOKEN")
    
    if not user_id or not api_token:
        raise ValueError("QUANTCONNECT_USER_ID and QUANTCONNECT_API_TOKEN must be set")
    
    return StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "--platform", "linux/arm64",
            "-e", f"QUANTCONNECT_USER_ID={user_id}",
            "-e", f"QUANTCONNECT_API_TOKEN={api_token}",
            "quantconnect/mcp-server"
        ]
    )

async def run_agentic_loop(user_message: str, verbose: bool = True) -> str:
    """Run an agentic loop with Claude using QuantConnect MCP tools."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    async with stdio_client(get_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get tools
            mcp_tools = await session.list_tools()
            tools = [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in mcp_tools.tools
            ]
            
            if verbose:
                print(f"Loaded {len(tools)} tools")
            
            messages = [{"role": "user", "content": user_message}]
            
            for i in range(MAX_ITERATIONS):
                if verbose:
                    print(f"\n--- Iteration {i+1} ---")
                
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=8000,
                    tools=tools,
                    messages=messages
                )
                
                # Done?
                if response.stop_reason == "end_turn":
                    return "".join(b.text for b in response.content if hasattr(b, "text"))
                
                # Process response
                tool_results = []
                for block in response.content:
                    if hasattr(block, "text") and verbose:
                        print(f"Agent: {block.text[:200]}...")
                    
                    if block.type == "tool_use":
                        if verbose:
                            print(f"Calling: {block.name}")
                        
                        try:
                            result = await session.call_tool(block.name, arguments=block.input)
                            content = result.content[0].text if result.content else "Empty"
                        except Exception as e:
                            content = f"Error: {e}"
                            if verbose:
                                print(f"Tool error: {e}")
                        
                        if verbose:
                            print(f"Result: {content[:300]}...")
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content
                        })
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            
            return "Max iterations reached"

async def list_tools(filter_name: Optional[str] = None):
    """List available tools from QuantConnect MCP server."""
    async with stdio_client(get_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            
            for t in tools.tools:
                if filter_name and filter_name not in t.name:
                    continue
                print(f"\n{t.name}: {t.description}")
                if filter_name:
                    print(json.dumps(t.inputSchema, indent=2))

async def main():
    user_message = """Create a 0DTE (zero days to expiration) trading strategy on QuantConnect. 
    
    Create a 0DTE iron condor strategy on SPX that enters daily at 10:00 AM ET by selling the 0.15 delta put and call with 2-strike-wide wings, collects a minimum $0.40 credit, exits at 50% profit or 2x loss or 3:45 PM ET, skips trading when VIX is above 30, and risks no more than 2% of portfolio per trade.

    Use the QuantConnect MCP tools to:
    1. Create a new project
    2. Create a Python algorithm file with a 0DTE options strategy on SPX
    3. The strategy should enter positions daily, trade SPX options with 0DTE expiration
    4. Include proper risk management and exit logic
    
    Use the create_project, create_file, and other necessary tools to build this strategy."""
    result = await run_agentic_loop(user_message)
    print(f"\nFinal: {result}")

if __name__ == "__main__":
    asyncio.run(main())

