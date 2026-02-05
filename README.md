# QuantConnect MCP Client

Simple setup using a `.env` file.

## Quick Setup

1. Create a `.env` file in the `qc_mcp` directory:
   ```bash
   cd qc_mcp
   ```

2. Add your credentials to `.env`:
   ```
   QUANTCONNECT_USER_ID=your_user_id_here
   QUANTCONNECT_API_TOKEN=your_api_token_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

3. Run the pipeline:
   ```bash
   python main.py
   ```

That's it! The script automatically loads credentials from `.env`.

## Alternative: Pass credentials directly

You can also pass credentials programmatically:

```python
client = QuantConnectMCPClient(
    user_id="your_user_id",
    api_token="your_api_token"
)
```

Parameters passed directly override `.env` file values.

## Project structure

```
piggybank/
├── LICENSE
├── README.md
└── qc_mcp/
    ├── main.py                      # Orchestration logic (Spec → Code → Exec → revision loop)
    ├── config.py                    # Configuration constants
    ├── env.example                  # Environment variable template
    ├── strategy_agents/
    │   ├── __init__.py              # Package exports
    │   ├── instructions.py          # Agent instructions (easy to edit)
    │   └── templates.py             # QC templates / reference code skeleton
    ├── utils/
    │   ├── __init__.py              # Package exports
    │   ├── mcp_connection.py        # MCP connection handler
    │   ├── parsing.py               # Code extraction, error parsing
    │   └── prompts.py               # Dynamic prompt builders
    └── (venv files)                 # `bin/`, `lib/`, `include/`, `pyvenv.cfg` (checked in here)
```