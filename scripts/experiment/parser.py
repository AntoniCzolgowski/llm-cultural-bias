"""
Response parser for LLM survey answers.
Extracts integer 1-10 from model output, returns None for invalid/refusal.
"""
import re


def parse_response(text: str) -> int | None:
    """
    Parse LLM response to extract integer 1-10.
    
    Args:
        text: Raw model output
        
    Returns:
        Integer 1-10 if valid, None otherwise (refusal/invalid)
    """
    if not text or not isinstance(text, str):
        return None
    
    # Clean control tokens and whitespace
    text = re.sub(r'<\|.*?\|>', '', text)  # Remove special tokens like <|end|>
    text = text.strip()
    
    if not text:
        return None
    
    # Try to find a number 1-10
    # Pattern: word boundary, then 10 or single digit 1-9, word boundary
    # Check each line from the beginning (model should respond with number first)
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Look for standalone numbers
        match = re.search(r'\b(10|[1-9])\b', line)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 10:
                return value
    
    return None


def test_parser():
    """Test cases for parser"""
    test_cases = [
        ("7", 7),
        ("10", 10),
        ("1", 1),
        ("  5  ", 5),
        ("My answer is 8", 8),
        ("3\n\nBecause...", 3),
        ("<|end|>6<|assistant|>", 6),
        ("I cannot answer", None),
        ("", None),
        (None, None),
        ("0", None),  # Out of range
        ("11", None),  # Out of range  
        ("The answer is 10.", 10),
        ("7.5", 7),  # Takes first valid int
    ]
    
    print("Testing parser...")
    passed = 0
    for input_text, expected in test_cases:
        result = parse_response(input_text)
        status = "✓" if result == expected else "✗"
        if result != expected:
            print(f"  {status} parse_response({repr(input_text)}) = {result}, expected {expected}")
        passed += (result == expected)
    
    print(f"Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


if __name__ == "__main__":
    test_parser()
