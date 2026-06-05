"""L3 subagent runner: agentic tool-use loop on Anthropic SDK.

Implements the task() delegation mechanism from the Deep Agents pattern.
Each subagent call creates a fresh messages.create() with role-specific
tools and system prompt — no shared conversation history (context isolation).

The agentic loop:
1. Send system prompt + user message + tools to Claude
2. If response contains tool_use blocks: execute each tool, append results
3. Re-call with tool results until model returns a final-answer tool
   (submit_assessment, submit_mutation, report_result) or text, or max_turns
4. Extract structured output from the final-answer tool call

Built from scratch using anthropic.messages.create() — NOT extending
llm_client.py which only supports plain text in/out.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from layer3.config import SUPERVISOR_MODEL

import anthropic

# Audit log for all LLM calls (append-only JSONL)
AUDIT_LOG_DIR = Path(__file__).parent / ".llm_audit"

# Final-answer tool names — when the model calls one of these, the loop ends
FINAL_ANSWER_TOOLS = frozenset({
    "submit_assessment",
    "submit_mutation",
    "report_result",
    "submit_portfolio_assessment",
})


def _get_client() -> anthropic.Anthropic:
    """Get or create Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _audit_log(record: dict) -> None:
    """Append one record to the JSONL audit log."""
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_LOG_DIR / "agent_calls.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _compute_cache_key(
    system: str, tools: List[dict], messages: List[dict], model: str,
) -> str:
    """SHA-256 cache key from the full request."""
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(system.encode())
    h.update(json.dumps(tools, sort_keys=True).encode())
    h.update(json.dumps(messages, sort_keys=True, default=str).encode())
    return h.hexdigest()


async def task(
    role: str,
    user_message: str,
    system_prompt: str,
    tools: List[dict],
    tool_handlers: Dict[str, Callable],
    model: str = SUPERVISOR_MODEL,
    max_turns: int = 10,
    temperature: float = 0.0,
) -> Tuple[Optional[dict], dict]:
    """Delegate a task to a subagent.

    Creates a fresh conversation with the specified system prompt and tools.
    Runs an agentic loop: if the model returns tool_use blocks, execute
    each tool via tool_handlers, append tool_result, re-call. Loop ends
    when:
      (a) Model calls a final-answer tool (submit_assessment, etc.)
      (b) Model returns a text-only response (no tool calls)
      (c) max_turns is reached

    Args:
        role: Agent role name (for logging).
        user_message: The task description assembled by the supervisor.
        system_prompt: Role-specific system prompt from prompts.py.
        tools: Anthropic tool definitions (list of dicts with name,
            description, input_schema).
        tool_handlers: Map of tool_name → callable(input_dict) → output_str.
            Must include all tools in the tools list.
        model: Claude model ID.
        max_turns: Maximum conversation turns (each turn = one API call).
        temperature: Sampling temperature (0.0 for reproducibility).

    Returns:
        (structured_output, metadata) where:
        - structured_output: dict from final-answer tool input, or None
          if the agent returned text instead
        - metadata: dict with role, model, turns, tokens, wall_time_s,
          cache_key, final_text (if text response)
    """
    client = _get_client()
    messages = [{"role": "user", "content": user_message}]

    total_input_tokens = 0
    total_output_tokens = 0
    t_start = time.time()
    structured_output = None
    final_text = None

    for turn in range(max_turns):
        # Make API call
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            tools=tools if tools else anthropic.NOT_GIVEN,
            messages=messages,
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Model finished without tool calls — extract text
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text
                    break
            break

        # Process content blocks
        tool_use_blocks = []
        for block in response.content:
            if block.type == "tool_use":
                tool_use_blocks.append(block)
            elif hasattr(block, "text") and block.text:
                final_text = block.text

        if not tool_use_blocks:
            # No tool calls — done
            break

        # Check for final-answer tool
        for tb in tool_use_blocks:
            if tb.name in FINAL_ANSWER_TOOLS:
                structured_output = tb.input
                # Don't execute the final-answer tool — its input IS the output
                break

        if structured_output is not None:
            break

        # Execute non-final tools and build tool results
        # First, add the assistant message with ALL content blocks
        messages.append({
            "role": "assistant",
            "content": [
                {"type": b.type, "id": b.id, "name": b.name, "input": b.input}
                if b.type == "tool_use"
                else {"type": "text", "text": b.text}
                for b in response.content
            ],
        })

        # Then add tool results for each tool_use block
        tool_results = []
        for tb in tool_use_blocks:
            handler = tool_handlers.get(tb.name)
            if handler is None:
                result_text = json.dumps({"error": f"Unknown tool: {tb.name}"})
            else:
                try:
                    result = handler(tb.input)
                    result_text = result if isinstance(result, str) else json.dumps(result, default=str)
                except Exception as e:
                    result_text = json.dumps({"error": f"{type(e).__name__}: {e}"})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})

    wall_time = time.time() - t_start

    # Build metadata
    cache_key = _compute_cache_key(
        system_prompt, tools,
        [{"role": "user", "content": user_message}],
        model,
    )
    metadata = {
        "role": role,
        "model": model,
        "turns": turn + 1,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "wall_time_s": round(wall_time, 2),
        "cache_key": cache_key,
        "final_text": final_text,
        "has_structured_output": structured_output is not None,
    }

    # Audit log
    _audit_log({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **metadata,
        "structured_output_keys": list(structured_output.keys()) if structured_output else None,
    })

    return structured_output, metadata


def task_sync(
    role: str,
    user_message: str,
    system_prompt: str,
    tools: List[dict],
    tool_handlers: Dict[str, Callable],
    model: str = SUPERVISOR_MODEL,
    max_turns: int = 10,
    temperature: float = 0.0,
) -> Tuple[Optional[dict], dict]:
    """Synchronous wrapper for task(). Use when not in an async context."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Already in an async context — can't use asyncio.run
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                task(role, user_message, system_prompt, tools,
                     tool_handlers, model, max_turns, temperature),
            )
            return future.result()
    except RuntimeError:
        # No running loop — use asyncio.run directly
        return asyncio.run(
            task(role, user_message, system_prompt, tools,
                 tool_handlers, model, max_turns, temperature),
        )
