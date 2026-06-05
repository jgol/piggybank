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

def build_zero_trades_prompt(code: str, backtest_info: dict | None = None) -> str:
    info_text = json.dumps(backtest_info, indent=2) if backtest_info else "No additional info"
    
    return f"""The code compiles but generated 0 trades.

    **Most Likely Causes:**
    1. Greeks are None - code returns early without fallback
    2. No contracts match criteria - filters too restrictive
    3. Chain empty in scheduled function - not using cached contracts

    **Backtest Info:**
    {info_text}

    **Current Code:**
    ```python
    {code}
    ```

    **Required Fixes:**

    1. Add strike-based fallback when Greeks unavailable:
    ```python
    # Instead of only this:
    if not contracts_with_greeks:
        return

    # Add fallback:
    if not contracts_with_greeks:
        self.Debug(f"{{self.Time}}: No Greeks, using strike-based selection")
        underlying = self.Securities[self.spx.Symbol].Price
        # Select ~5% OTM by strike distance
        otm_puts = [c for c in puts if c.Strike < underlying * 0.95]
        otm_calls = [c for c in calls if c.Strike > underlying * 1.05]
    ```

    2. Add debug logging at every decision point:
    ```python
    self.Debug(f"{{self.Time}}: {{len(contracts)}} contracts, {{len(puts)}} puts, {{len(calls)}} calls")
    self.Debug(f"{{self.Time}}: {{len(with_greeks)}} have Greeks")
    ```

    3. Verify using cached contracts (not CurrentSlice) in TryEntry

    4. Ensure orders use contract.Symbol (not strings)

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