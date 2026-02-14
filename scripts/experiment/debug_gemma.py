#!/usr/bin/env python3
"""
Debug script for Gemma3 - testing temperature and alternative prompts.

Issues identified in pilot:
- Gemma3 returns identical responses (zero variance) despite temp=0.7
- Need to verify temperature is actually being applied
- Try WVS interview-style prompt

Usage:
    python debug_gemma.py
"""
import os
import time
import ollama

# Get Ollama host
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
if not OLLAMA_HOST.startswith('http'):
    OLLAMA_HOST = f'http://{OLLAMA_HOST}'

print(f"Ollama host: {OLLAMA_HOST}")

client = ollama.Client(host=OLLAMA_HOST)

# =============================================================================
# PROMPTS TO TEST
# =============================================================================

# Original prompt (zero variance observed)
PROMPT_ORIGINAL = {
    'system': """Answer the following survey question from the perspective described. Respond with ONLY a single number from 1 to 10. No explanation.""",
    'user': """Profile: You are a male from China with higher education, aged 30-49.

Question: How important is God in your life? (1 = not at all important, 10 = very important)

Your answer:"""
}

# WVS Interview-style prompt
PROMPT_WVS_INTERVIEW = {
    'system': """You are participating in a World Values Survey interview. You must fully embody the demographic profile given and answer AS THAT PERSON WOULD, based on typical values, beliefs, and cultural background of someone with those characteristics. 

Your response must be ONLY a single integer from 1 to 10. Nothing else.""",
    'user': """You are being interviewed for the World Values Survey.

YOUR DEMOGRAPHIC PROFILE:
- Nationality: Chinese (from mainland China)
- Sex: Male  
- Age: 30-49 years old
- Education: University/higher education

IMPORTANT: Your answer should reflect how a REAL PERSON with this exact profile would typically respond, considering:
- Cultural and religious context of China
- Typical beliefs of educated urban Chinese males
- The secular nature of Chinese society

SURVEY QUESTION: "How important is God in your life?"
Scale: 1 = not at all important, 10 = very important

Your answer (single number 1-10):"""
}

# Explicit variance prompt
PROMPT_VARIANCE = {
    'system': """You are simulating survey responses. Each response should reflect natural human variation - not everyone with the same demographics answers identically. Respond with ONLY a number 1-10.""",
    'user': """Simulate a response from this person:
- Chinese male, age 30-49, university educated

Question: How important is God in your life? (1=not important, 10=very important)

Consider that even within the same demographic, people vary in their beliefs. Give a realistic response for THIS individual (not an average).

Answer:"""
}

# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

CONFIGS = [
    {'name': 'Original prompt, temp=0.7', 'prompt': PROMPT_ORIGINAL, 'temp': 0.7},
    {'name': 'Original prompt, temp=1.0', 'prompt': PROMPT_ORIGINAL, 'temp': 1.0},
    {'name': 'Original prompt, temp=1.5', 'prompt': PROMPT_ORIGINAL, 'temp': 1.5},
    {'name': 'WVS Interview, temp=0.7', 'prompt': PROMPT_WVS_INTERVIEW, 'temp': 0.7},
    {'name': 'WVS Interview, temp=1.0', 'prompt': PROMPT_WVS_INTERVIEW, 'temp': 1.0},
    {'name': 'Variance prompt, temp=1.0', 'prompt': PROMPT_VARIANCE, 'temp': 1.0},
]

N_QUERIES = 10  # per config


def run_queries(prompt: dict, temperature: float, n: int = 10) -> list:
    """Run n queries and return responses"""
    responses = []
    
    for i in range(n):
        try:
            response = client.chat(
                model='gemma3:12b',
                messages=[
                    {'role': 'system', 'content': prompt['system']},
                    {'role': 'user', 'content': prompt['user']}
                ],
                options={
                    'temperature': temperature,
                    'top_k': 50,
                    'top_p': 1.0,
                    'num_predict': 10,
                    'seed': None,  # Explicitly no seed for randomness
                }
            )
            text = response['message']['content'].strip()
            responses.append(text)
        except Exception as e:
            responses.append(f"ERROR: {e}")
    
    return responses


def parse_responses(responses: list) -> list:
    """Parse responses to integers"""
    import re
    parsed = []
    for r in responses:
        match = re.search(r'\b(10|[1-9])\b', r)
        parsed.append(int(match.group(1)) if match else None)
    return parsed


def main():
    print("="*70)
    print("GEMMA3 DEBUG - Temperature & Prompt Testing")
    print("="*70)
    print(f"Model: gemma3:12b")
    print(f"Queries per config: {N_QUERIES}")
    print()
    
    results = {}
    
    for config in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        start = time.time()
        responses = run_queries(config['prompt'], config['temp'], N_QUERIES)
        elapsed = time.time() - start
        
        parsed = parse_responses(responses)
        
        # Statistics
        valid = [p for p in parsed if p is not None]
        unique = set(valid)
        
        print(f"\nRaw responses: {responses}")
        print(f"Parsed values: {parsed}")
        print(f"Unique values: {unique} ({len(unique)} different)")
        print(f"Time: {elapsed:.1f}s ({elapsed/N_QUERIES:.2f}s per query)")
        
        if valid:
            mean = sum(valid) / len(valid)
            variance = sum((x - mean)**2 for x in valid) / len(valid) if len(valid) > 1 else 0
            print(f"Mean: {mean:.2f}, Variance: {variance:.2f}")
        
        results[config['name']] = {
            'responses': responses,
            'parsed': parsed,
            'unique_count': len(unique),
            'mean': mean if valid else None,
            'variance': variance if valid else None,
        }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<35} {'Unique':<8} {'Mean':<8} {'Variance':<10}")
    print("-"*70)
    
    for name, data in results.items():
        unique = data['unique_count']
        mean = f"{data['mean']:.2f}" if data['mean'] else "N/A"
        var = f"{data['variance']:.2f}" if data['variance'] else "N/A"
        
        # Flag if zero variance
        flag = "⚠️ NO VARIANCE" if unique == 1 else "✓"
        print(f"{name:<35} {unique:<8} {mean:<8} {var:<10} {flag}")
    
    print("="*70)
    print("\nRECOMMENDATION:")
    
    # Find best config (most variance)
    best = max(results.items(), key=lambda x: x[1]['unique_count'])
    print(f"Best config: {best[0]} ({best[1]['unique_count']} unique values)")
    
    if best[1]['unique_count'] == 1:
        print("\n⚠️  ALL CONFIGS SHOW ZERO VARIANCE!")
        print("This suggests Gemma3 may be inherently deterministic for this task.")
        print("Consider: using a different model, or accepting low variance for Gemma.")


if __name__ == "__main__":
    main()
