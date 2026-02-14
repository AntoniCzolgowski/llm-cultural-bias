"""
Response parser - extracts integer 1-10 from model output.
"""
import re

def parse_response(text: str) -> int | None:
    """Parse LLM response to extract integer 1-10."""
    if not text:
        return None
    text = re.sub(r'<\|.*?\|>', '', text).strip()
    match = re.search(r'\b(10|[1-9])\b', text)
    return int(match.group(1)) if match else None
