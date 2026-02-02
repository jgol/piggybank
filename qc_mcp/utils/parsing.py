"""Parsing utilities for code extraction and error handling."""

import json
import re

def extract_python_code(text: str) -> str | None:
    """
    Extract the first Python code block from markdown text.
    
    Args:
        text: Text potentially containing markdown code blocks
        
    Returns:
        Extracted Python code or None if not found
    """
    # Try explicit python fence first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic fence
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Verify it looks like Python
        if any(kw in code for kw in ["def ", "class ", "import ", "from "]):
            return code
    
    # Fallback: try to find code without fences
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith(('from ', 'import ', 'class ', 'def ')):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        result = '\n'.join(code_lines).strip()
        if "QCAlgorithm" in result or "AlgorithmImports" in result:
            return result
    
    return None

# Optional
def extract_compile_errors(raw_response: str) -> list[str]:
    """
    Extract compile error messages from MCP response.
    
    Args:
        raw_response: Raw response string from MCP tool
        
    Returns:
        List of error message strings
    """
    errors = []
    
    try:
        data = json.loads(raw_response)
        
        if isinstance(data, dict):
            # Direct errors field
            if "errors" in data:
                err_list = data["errors"]
                if isinstance(err_list, list):
                    errors.extend(str(e) for e in err_list)
                elif isinstance(err_list, str):
                    errors.append(err_list)
            
            # Nested in compile result
            if "compile" in data and isinstance(data["compile"], dict):
                if "logs" in data["compile"]:
                    errors.extend(data["compile"]["logs"])
            
            # Error/message fields
            if "error" in data:
                errors.append(str(data["error"]))
            if "message" in data and "error" in str(data.get("state", "")).lower():
                errors.append(str(data["message"]))
                
    except json.JSONDecodeError:
        # Extract error patterns from raw text
        error_patterns = [
            r"error[:\s]+(.+?)(?:\n|$)",
            r"Error[:\s]+(.+?)(?:\n|$)",
            r"CS\d+[:\s]+(.+?)(?:\n|$)",
            r"line \d+[:\s]+(.+?)(?:\n|$)",
        ]
        for pattern in error_patterns:
            matches = re.findall(pattern, raw_response, re.IGNORECASE)
            errors.extend(matches)
    
    # Deduplicate while preserving order
    seen = set()
    unique_errors = []
    for e in errors:
        e_clean = e.strip()
        if e_clean and e_clean not in seen:
            seen.add(e_clean)
            unique_errors.append(e_clean)
    
    return unique_errors if unique_errors else ["Compilation failed - no specific error extracted"]


# def prepare_code_for_json(code: str) -> str:
#     """
#     Escape code for embedding in JSON string.
    
#     Args:
#         code: Raw Python code
        
#     Returns:
#         JSON-safe escaped string (without outer quotes)
#     """
#     escaped = json.dumps(code)
#     return escaped[1:-1]  # Remove surrounding quotes


# def truncate_text(text: str, limit: int = 20000) -> str:
#     """
#     Truncate text to a character limit, keeping beginning and end.
    
#     Args:
#         text: Text to truncate
#         limit: Maximum character count
        
#     Returns:
#         Truncated text with middle ellipsis if needed
#     """
#     if len(text) <= limit:
#         return text
#     half = limit // 2
#     return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"