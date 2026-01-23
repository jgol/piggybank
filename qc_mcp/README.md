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

3. Run the script:
   ```bash
   python mcp_connect.py
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
