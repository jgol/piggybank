"""QuantConnect MCP connection handler."""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolInfo:
    """MCP tool metadata."""
    name: str
    description: str
    input_schema: dict[str, Any]


class QCMCPConnection:
    """Manages connection to QuantConnect MCP server via Docker."""
    
    def __init__(self, init_timeout: float = 30.0, tool_timeout: float = 120.0):
        self.session: ClientSession | None = None
        self._stdio_context = None
        self._session_context = None
        self.tools: list[ToolInfo] = []
        self.tools_by_name: dict[str, ToolInfo] = {}
        self.init_timeout = init_timeout
        self.tool_timeout = tool_timeout

    async def __aenter__(self):
        user_id = os.getenv("QUANTCONNECT_USER_ID")
        api_token = os.getenv("QUANTCONNECT_API_TOKEN")
        
        if not user_id or not api_token:
            raise RuntimeError(
                "Missing credentials. Set QUANTCONNECT_USER_ID and "
                "QUANTCONNECT_API_TOKEN environment variables."
            )

        # Build Docker command
        args = ["run", "-i", "--rm"]
        if platform := os.getenv("DOCKER_PLATFORM"):
            args += ["--platform", platform]
        args += [
            "-e", f"QUANTCONNECT_USER_ID={user_id}",
            "-e", f"QUANTCONNECT_API_TOKEN={api_token}",
            "quantconnect/mcp-server"
        ]

        # Start Docker container
        try:
            self._stdio_context = stdio_client(
                StdioServerParameters(command="docker", args=args)
            )
            read, write = await self._stdio_context.__aenter__()
        except FileNotFoundError:
            raise RuntimeError("Docker not found. Ensure Docker is installed and running.")
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP container: {e}")

        # Initialize MCP session
        try:
            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()
            await asyncio.wait_for(self.session.initialize(), timeout=self.init_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP session initialization timed out ({self.init_timeout}s). "
                "Check network and container status."
            )

        # Load available tools
        self.tools = [
            ToolInfo(t.name, t.description or "", t.inputSchema or {})
            for t in (await self.session.list_tools()).tools
        ]
        self.tools_by_name = {t.name: t for t in self.tools}
        
        return self

    async def __aexit__(self, *exc_info):
        for ctx in (self._session_context, self._stdio_context):
            if ctx:
                try:
                    await ctx.__aexit__(*exc_info)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute an MCP tool with timeout protection."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(name, arguments=arguments),
                timeout=self.tool_timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Tool '{name}' timed out after {self.tool_timeout}s")
        
        if not result.content:
            return ""
        
        # MCP response may contain various types of content
        chunks = []
        for block in result.content:
            if text := getattr(block, "text", None):
                chunks.append(text)
            elif (data := getattr(block, "data", None)) is not None:
                chunks.append(json.dumps(data) if isinstance(data, (dict, list)) else str(data))
            elif hasattr(block, "model_dump_json"):
                chunks.append(block.model_dump_json())
            else:
                chunks.append(str(block))
        
        return "\n".join(chunks)
    
    async def health_check(self) -> bool:
        """Verify MCP connection is responsive."""
        if not self.session:
            return False
        try:
            await asyncio.wait_for(self.session.list_tools(), timeout=5.0)
            return True
        except (asyncio.TimeoutError, Exception):
            return False