"""Dynamic prompt builders to easily iterate through agent interactions."""

import json
from strategy_agents.templates import QC_API_REFERENCE


def build_code_prompt(spec_text: str, max_spec_length: int = 8000) -> str:
    
    return f"""Implement this strategy specification:
    {spec_text}
    Produce complete, compilable QuantConnect Python code following the template and patterns in your instructions."""

# To fix compilation errors
def build_compile_retry_prompt(code: str, errors: list[str]) -> str:
    
    error_text = "\n".join(f"  - {e}" for e in errors[:10])  # Limit to 10 errors
    
    return f"""Your previous code failed to compile. Fix the errors while preserving strategy logic.

    **Compilation Errors:**
    {error_text}

    **Your Code:**
    ```python
    {code}
    ```

    **API Reference (follow exactly):**
    {QC_API_REFERENCE}

    **Common Fixes:**
    - Option chain: Use `data.OptionChains.get(symbol)` not `data.OptionChains[symbol]`
    - Greeks: Always check `if c.Greeks and c.Greeks.Delta is not None`
    - Imports: Use only `from AlgorithmImports import *`
    - Time: Use `self.Time` not `datetime.now()`

    Return the complete fixed code in a single ```python``` block. Do not explain the changes."""

# When there are no trades...
def build_zero_trades_prompt(code: str, backtest_info: dict | None = None) -> str:
    
    info_text = json.dumps(backtest_info, indent=2) if backtest_info else "No additional info"
    
    return f"""The code compiles and backtests successfully, but generated 0 trades.

    **Possible Causes:**
    1. Entry conditions too restrictive (time window, VIX filter, delta targets)
    2. Option chain filtering too narrow (no contracts match criteria)
    3. Scheduled function not triggering properly
    4. Logic errors preventing order execution

    **Backtest Info:**
    {info_text}

    **Current Code:**
    ```python
    {code}
    ```

    **Instructions:**
    1. Add debug logging to trace execution:
    - Log when scheduled functions are called
    - Log available contract count after filtering
    - Log why entry conditions might fail
    2. Consider relaxing filters:
    - Widen strike range in SetFilter (e.g., -50 to +50)
    - Relax delta tolerance or premium minimums
    - Extend time windows
    3. Verify option chain access is correct
    4. Ensure orders use correct symbols (contract.Symbol, not strings)

    Return the complete revised code in a single ```python``` block."""


def build_exec_prompt(project_name: str, file_name: str, code: str, project_id: str | None = None) -> str:
    if project_id:
        project_info = f'**Project ID:** {project_id} (use this for all operations)'
    else:
        project_info = f'**Project Name:** "{project_name}" (create if not exists, then reuse projectId)'
    
    return f"""Deploy and test this code on QuantConnect.

    {project_info}
    **File:** "{file_name}"

    **Code:**
    ```python
    {code}
    ```

    Follow your workflow: verify/create project → upload code → compile → backtest → report results.
    Call `submit_exec_result()` exactly once when finished."""