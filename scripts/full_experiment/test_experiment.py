#!/usr/bin/env python3
"""
LLM Cultural Bias Survey - Comprehensive Test
==============================================
Tests everything before full experiment:
  1. Config & data loading
  2. Parser edge cases
  3. All 3 models respond correctly
  4. Timing per model (extrapolate full experiment)
  5. Response quality (compare with human means)
  6. Checkpoint save/load
  7. Full mini-experiment (3 personas × 10 queries × 3 models)

Estimated runtime: ~8-10 minutes
"""

import os
import sys
import re
import gc
import time
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG (inline — no dependency on existing config.py)
# =============================================================================
BASE_DIR = '/projects/ancz7294/llm-cultural-bias'
MODELS_DIR = '/projects/ancz7294/models'

PATHS = {
    'human_distributions': f'{BASE_DIR}/data/raw/human_distributions.csv',
    'test_results': f'{BASE_DIR}/results/test',
    'test_checkpoints': f'{BASE_DIR}/checkpoints/test',
    'bielik': f'{MODELS_DIR}/bielik-11b-v3',
    'qwen': f'{MODELS_DIR}/qwen3-4b-instruct-2507',
}

MODEL_PARAMS = {
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 1.0,
    'max_new_tokens': 10,
}

SYSTEM_PROMPT = """Answer the following survey question from the perspective described. Respond with ONLY a single number from 1 to 10. No explanation."""

USER_PROMPT_TEMPLATE = """Profile: You are a {sex} from {country} with {education} education, aged {age_group}.

Question: How important is God in your life? (1 = not at all important, 10 = very important)

Your answer:"""

COUNTRY_MAP = {'CHN': 'China', 'SVK': 'Slovakia', 'USA': 'the United States'}
EDUCATION_MAP = {'Lower': 'lower', 'Medium': 'medium', 'Higher': 'higher'}
AGE_MAP = {'18-29': '18-29', '30-49': '30-49', '50-64': '50-64', '65+': '65 or older'}
SEX_MAP = {'Male': 'male', 'Female': 'female'}

# Test personas: 1 per country, diverse demographics
TEST_PERSONAS = [
    'CHN_Male_30-49_Higher',
    'SVK_Female_50-64_Medium', 
    'USA_Female_18-29_Higher',
]
QUERIES_PER_TEST = 10

# =============================================================================
# PARSER
# =============================================================================
def parse_response(text: str) -> int | None:
    if not text:
        return None
    text = re.sub(r'<\|.*?\|>', '', text).strip()
    match = re.search(r'\b(10|[1-9])\b', text)
    return int(match.group(1)) if match else None

# =============================================================================
# HELPERS
# =============================================================================
def get_ollama_host():
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    return f'http://{host}' if not host.startswith('http') else host

def format_prompt(persona_row):
    return USER_PROMPT_TEMPLATE.format(
        sex=SEX_MAP[persona_row['sex']],
        country=COUNTRY_MAP[persona_row['country']],
        education=EDUCATION_MAP[persona_row['education']],
        age_group=AGE_MAP[persona_row['age_group']],
    )

def get_human_mean(persona_row):
    """Calculate human mean from PMF columns."""
    pmf_cols = [f'pmf_{i}' for i in range(1, 11)]
    pmf = [persona_row[c] for c in pmf_cols]
    return sum((i+1) * p for i, p in enumerate(pmf))

def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def pass_fail(ok):
    return "PASS" if ok else "FAIL"

# =============================================================================
# TEST 1: Config & Data
# =============================================================================
def test_config_and_data():
    separator("TEST 1: Config & Data Loading")
    
    errors = []
    
    # Check paths exist
    csv_path = PATHS['human_distributions']
    if os.path.exists(csv_path):
        print(f"  [OK] Human distributions CSV exists")
    else:
        print(f"  [FAIL] CSV not found: {csv_path}")
        errors.append("CSV missing")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"  [OK] Loaded {len(df)} personas")
    
    # Check countries
    countries = sorted(df['country'].unique())
    expected = ['CHN', 'SVK', 'USA']
    ok = countries == expected
    print(f"  [{pass_fail(ok)}] Countries: {countries} (expected {expected})")
    if not ok: errors.append("Countries mismatch")
    
    # Check test personas exist
    for pid in TEST_PERSONAS:
        exists = pid in df['persona_id'].values
        print(f"  [{pass_fail(exists)}] Test persona '{pid}' exists")
        if not exists: errors.append(f"Missing persona {pid}")
    
    # Check PMF columns
    pmf_cols = [f'pmf_{i}' for i in range(1, 11)]
    has_pmf = all(c in df.columns for c in pmf_cols)
    print(f"  [{pass_fail(has_pmf)}] PMF columns (pmf_1 to pmf_10) present")
    if not has_pmf: errors.append("PMF columns missing")
    
    # Validate PMFs sum to ~1
    if has_pmf:
        pmf_sums = df[pmf_cols].sum(axis=1)
        all_close = all(abs(s - 1.0) < 0.01 for s in pmf_sums)
        print(f"  [{pass_fail(all_close)}] All PMFs sum to ~1.0 (range: {pmf_sums.min():.4f} - {pmf_sums.max():.4f})")
        if not all_close: errors.append("PMF sums not ~1.0")
    
    # Check demographic breakdowns
    print(f"\n  Demographics breakdown:")
    for col in ['country', 'sex', 'age_group', 'education']:
        vals = sorted(df[col].unique())
        print(f"    {col}: {vals}")
    
    # Model paths
    for name, path in [('bielik', PATHS['bielik']), ('qwen', PATHS['qwen'])]:
        exists = os.path.exists(path)
        print(f"  [{pass_fail(exists)}] Model path '{name}': {path}")
        if not exists: errors.append(f"Model path missing: {name}")
    
    # Ollama
    host = get_ollama_host()
    print(f"  [OK] Ollama host: {host}")
    
    return df, errors

# =============================================================================
# TEST 2: Parser
# =============================================================================
def test_parser():
    separator("TEST 2: Parser Edge Cases")
    
    test_cases = [
        ("7", 7),
        ("3\n", 3),
        ("10", 10),
        ("1", 1),
        ("  5  ", 5),
        ("I would say 8", 8),
        ("My answer is 6.", 6),
        ("Rating: 4/10", 4),
        ("<|end|>3<|eot_id|>", 3),
        ("", None),
        (None, None),
        ("I cannot answer", None),
        ("ten", None),
        ("The importance of God is 9 in my life", 9),
    ]
    
    errors = []
    for input_text, expected in test_cases:
        result = parse_response(input_text)
        ok = result == expected
        display = repr(input_text) if input_text else str(input_text)
        print(f"  [{pass_fail(ok)}] parse({display:45s}) = {result} (expected {expected})")
        if not ok: errors.append(f"Parser: {input_text} -> {result} != {expected}")
    
    return errors

# =============================================================================
# TEST 3: Model Queries
# =============================================================================
def test_model_gemma(persona_row):
    """Test gemma3 via Ollama."""
    import ollama
    client = ollama.Client(host=get_ollama_host())
    
    # Warmup
    client.chat(model='gemma3:12b', messages=[{'role': 'user', 'content': 'OK'}], options={'num_predict': 5})
    
    prompt = format_prompt(persona_row)
    
    t0 = time.time()
    r = client.chat(
        model='gemma3:12b',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
        options={
            'temperature': MODEL_PARAMS['temperature'],
            'top_k': MODEL_PARAMS['top_k'],
            'num_predict': MODEL_PARAMS['max_new_tokens'],
        }
    )
    elapsed = time.time() - t0
    raw = r['message']['content'].strip()
    parsed = parse_response(raw)
    
    return raw, parsed, elapsed

def test_model_transformers(path, persona_row):
    """Test a Transformers model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map='auto')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = format_prompt(persona_row)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors='pt', add_generation_prompt=True, tokenize=True
    ).to(model.device)
    
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=MODEL_PARAMS['max_new_tokens'],
            temperature=MODEL_PARAMS['temperature'],
            top_k=MODEL_PARAMS['top_k'],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    raw = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    parsed = parse_response(raw)
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    return raw, parsed, elapsed

def test_all_models(df):
    separator("TEST 3: Model Connectivity & Response Quality")
    
    persona_row = df[df['persona_id'] == TEST_PERSONAS[0]].iloc[0]
    human_mean = get_human_mean(persona_row)
    prompt_preview = format_prompt(persona_row)
    
    print(f"  Test persona: {TEST_PERSONAS[0]}")
    print(f"  Human mean: {human_mean:.2f}")
    print(f"  Prompt preview:\n    {prompt_preview[:120]}...")
    print()
    
    model_results = {}
    errors = []
    
    # Gemma3
    print(f"  Testing gemma3 (Ollama)...")
    try:
        raw, parsed, elapsed = test_model_gemma(persona_row)
        ok = parsed is not None and 1 <= parsed <= 10
        print(f"    [{pass_fail(ok)}] Raw: '{raw}' | Parsed: {parsed} | Time: {elapsed:.2f}s")
        model_results['gemma3'] = {'raw': raw, 'parsed': parsed, 'time': elapsed}
        if not ok: errors.append("gemma3 invalid response")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        errors.append(f"gemma3 error: {e}")
    
    # Bielik
    print(f"  Testing bielik (Transformers 4-bit)...")
    try:
        raw, parsed, elapsed = test_model_transformers(PATHS['bielik'], persona_row)
        ok = parsed is not None and 1 <= parsed <= 10
        print(f"    [{pass_fail(ok)}] Raw: '{raw}' | Parsed: {parsed} | Time: {elapsed:.2f}s (excl. load)")
        model_results['bielik'] = {'raw': raw, 'parsed': parsed, 'time': elapsed}
        if not ok: errors.append("bielik invalid response")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        errors.append(f"bielik error: {e}")
    
    # Qwen
    print(f"  Testing qwen (Transformers 4-bit)...")
    try:
        raw, parsed, elapsed = test_model_transformers(PATHS['qwen'], persona_row)
        ok = parsed is not None and 1 <= parsed <= 10
        print(f"    [{pass_fail(ok)}] Raw: '{raw}' | Parsed: {parsed} | Time: {elapsed:.2f}s (excl. load)")
        model_results['qwen'] = {'raw': raw, 'parsed': parsed, 'time': elapsed}
        if not ok: errors.append("qwen invalid response")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        errors.append(f"qwen error: {e}")
    
    return model_results, errors

# =============================================================================
# TEST 4: Mini Experiment (3 personas × 10 queries × 3 models)
# =============================================================================
def run_mini_experiment(df):
    separator("TEST 4: Mini Experiment (3 personas × 10 queries × 3 models = 90 queries)")
    
    all_results = []
    timing = {}
    errors = []
    
    # ---------- GEMMA3 ----------
    print(f"\n  --- gemma3 (Ollama) ---")
    import ollama
    client = ollama.Client(host=get_ollama_host())
    client.chat(model='gemma3:12b', messages=[{'role': 'user', 'content': 'OK'}], options={'num_predict': 5})
    
    gemma_times = []
    for pid in TEST_PERSONAS:
        persona_row = df[df['persona_id'] == pid].iloc[0]
        prompt = format_prompt(persona_row)
        human_mean = get_human_mean(persona_row)
        
        responses = []
        for j in range(QUERIES_PER_TEST):
            t0 = time.time()
            r = client.chat(
                model='gemma3:12b',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': MODEL_PARAMS['temperature'],
                    'top_k': MODEL_PARAMS['top_k'],
                    'num_predict': MODEL_PARAMS['max_new_tokens'],
                }
            )
            elapsed = time.time() - t0
            raw = r['message']['content'].strip()
            parsed = parse_response(raw)
            responses.append(parsed)
            gemma_times.append(elapsed)
            
            all_results.append({
                'persona_id': pid, 'model': 'gemma3', 'query_idx': j,
                'raw_response': raw, 'parsed_value': parsed,
                'is_valid': parsed is not None, 'response_time_sec': round(elapsed, 3),
            })
        
        valid = [r for r in responses if r is not None]
        model_mean = np.mean(valid) if valid else float('nan')
        unique = len(set(valid))
        print(f"    {pid}: responses={responses} | mean={model_mean:.1f} | human={human_mean:.1f} | unique={unique} | valid={len(valid)}/{QUERIES_PER_TEST}")
    
    timing['gemma3'] = np.mean(gemma_times)
    
    # ---------- BIELIK ----------
    print(f"\n  --- bielik (Transformers) ---")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"    Loading model...")
    t_load = time.time()
    bielik_tok = AutoTokenizer.from_pretrained(PATHS['bielik'])
    bielik_model = AutoModelForCausalLM.from_pretrained(PATHS['bielik'], quantization_config=bnb, device_map='auto')
    if bielik_tok.pad_token is None:
        bielik_tok.pad_token = bielik_tok.eos_token
    print(f"    Loaded in {time.time()-t_load:.1f}s")
    
    bielik_times = []
    for pid in TEST_PERSONAS:
        persona_row = df[df['persona_id'] == pid].iloc[0]
        prompt = format_prompt(persona_row)
        human_mean = get_human_mean(persona_row)
        
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
        inputs = bielik_tok.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True, tokenize=True).to(bielik_model.device)
        
        responses = []
        for j in range(QUERIES_PER_TEST):
            t0 = time.time()
            with torch.no_grad():
                out = bielik_model.generate(inputs, max_new_tokens=MODEL_PARAMS['max_new_tokens'],
                    temperature=MODEL_PARAMS['temperature'], top_k=MODEL_PARAMS['top_k'],
                    do_sample=True, pad_token_id=bielik_tok.pad_token_id)
            elapsed = time.time() - t0
            raw = bielik_tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            parsed = parse_response(raw)
            responses.append(parsed)
            bielik_times.append(elapsed)
            
            all_results.append({
                'persona_id': pid, 'model': 'bielik', 'query_idx': j,
                'raw_response': raw, 'parsed_value': parsed,
                'is_valid': parsed is not None, 'response_time_sec': round(elapsed, 3),
            })
        
        valid = [r for r in responses if r is not None]
        model_mean = np.mean(valid) if valid else float('nan')
        unique = len(set(valid))
        print(f"    {pid}: responses={responses} | mean={model_mean:.1f} | human={human_mean:.1f} | unique={unique} | valid={len(valid)}/{QUERIES_PER_TEST}")
    
    timing['bielik'] = np.mean(bielik_times)
    
    # Cleanup bielik
    del bielik_model, bielik_tok
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    
    # ---------- QWEN ----------
    print(f"\n  --- qwen (Transformers) ---")
    print(f"    Loading model...")
    t_load = time.time()
    qwen_tok = AutoTokenizer.from_pretrained(PATHS['qwen'])
    qwen_model = AutoModelForCausalLM.from_pretrained(PATHS['qwen'], quantization_config=bnb, device_map='auto')
    if qwen_tok.pad_token is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    print(f"    Loaded in {time.time()-t_load:.1f}s")
    
    qwen_times = []
    for pid in TEST_PERSONAS:
        persona_row = df[df['persona_id'] == pid].iloc[0]
        prompt = format_prompt(persona_row)
        human_mean = get_human_mean(persona_row)
        
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
        inputs = qwen_tok.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True, tokenize=True).to(qwen_model.device)
        
        responses = []
        for j in range(QUERIES_PER_TEST):
            t0 = time.time()
            with torch.no_grad():
                out = qwen_model.generate(inputs, max_new_tokens=MODEL_PARAMS['max_new_tokens'],
                    temperature=MODEL_PARAMS['temperature'], top_k=MODEL_PARAMS['top_k'],
                    do_sample=True, pad_token_id=qwen_tok.pad_token_id)
            elapsed = time.time() - t0
            raw = qwen_tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            parsed = parse_response(raw)
            responses.append(parsed)
            qwen_times.append(elapsed)
            
            all_results.append({
                'persona_id': pid, 'model': 'qwen', 'query_idx': j,
                'raw_response': raw, 'parsed_value': parsed,
                'is_valid': parsed is not None, 'response_time_sec': round(elapsed, 3),
            })
        
        valid = [r for r in responses if r is not None]
        model_mean = np.mean(valid) if valid else float('nan')
        unique = len(set(valid))
        print(f"    {pid}: responses={responses} | mean={model_mean:.1f} | human={human_mean:.1f} | unique={unique} | valid={len(valid)}/{QUERIES_PER_TEST}")
    
    timing['qwen'] = np.mean(qwen_times)
    
    # Cleanup qwen
    del qwen_model, qwen_tok
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_results, timing, errors

# =============================================================================
# TEST 5: Checkpoint Save/Load
# =============================================================================
def test_checkpoint(all_results):
    separator("TEST 5: Checkpoint Save/Load")
    
    errors = []
    cp_dir = Path(PATHS['test_checkpoints'])
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_path = cp_dir / 'test_checkpoint.csv'
    
    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(cp_path, index=False)
    print(f"  [OK] Saved {len(df)} rows to {cp_path}")
    
    # Load back
    df_loaded = pd.read_csv(cp_path)
    ok = len(df_loaded) == len(df)
    print(f"  [{pass_fail(ok)}] Loaded back {len(df_loaded)} rows")
    if not ok: errors.append("Checkpoint row count mismatch")
    
    # Check columns
    expected_cols = ['persona_id', 'model', 'query_idx', 'raw_response', 'parsed_value', 'is_valid', 'response_time_sec']
    has_cols = all(c in df_loaded.columns for c in expected_cols)
    print(f"  [{pass_fail(has_cols)}] All expected columns present")
    if not has_cols: errors.append("Checkpoint columns missing")
    
    # Check completed persona tracking
    completed = set(df_loaded['persona_id'].unique())
    print(f"  [OK] Can track completed personas: {completed}")
    
    # Simulate append (checkpoint resume)
    extra = [{'persona_id': 'TEST_APPEND', 'model': 'test', 'query_idx': 0, 
              'raw_response': '5', 'parsed_value': 5, 'is_valid': True, 'response_time_sec': 0.1}]
    pd.DataFrame(extra).to_csv(cp_path, mode='a', header=False, index=False)
    df_appended = pd.read_csv(cp_path)
    ok = len(df_appended) == len(df) + 1
    print(f"  [{pass_fail(ok)}] Append mode works ({len(df_appended)} rows after append)")
    if not ok: errors.append("Checkpoint append failed")
    
    # Cleanup test checkpoint
    cp_path.unlink()
    print(f"  [OK] Cleaned up test checkpoint")
    
    return errors

# =============================================================================
# SUMMARY & TIME ESTIMATES
# =============================================================================
def print_summary(timing, all_results, all_errors):
    separator("SUMMARY & TIME ESTIMATES")
    
    # Timing
    print(f"\n  Measured query times:")
    total_est = 0
    for model, avg_time in timing.items():
        queries = 63 * 100
        est_minutes = queries * avg_time / 60
        total_est += est_minutes
        print(f"    {model:8s}: {avg_time:.3f}s/query → {est_minutes:.0f} min for full experiment (6300 queries)")
    
    print(f"\n  Total estimated experiment time: {total_est:.0f} minutes ({total_est/60:.1f} hours)")
    print(f"  Recommended session time: {max(4, int(total_est/60) + 2)} hours")
    
    # Results quality
    results_df = pd.DataFrame(all_results)
    print(f"\n  Results quality:")
    for model in ['gemma3', 'bielik', 'qwen']:
        mdf = results_df[results_df['model'] == model]
        valid = mdf['is_valid'].sum()
        total = len(mdf)
        refusal = total - valid
        unique_vals = mdf['parsed_value'].dropna().nunique()
        print(f"    {model:8s}: {valid}/{total} valid ({refusal} refusals) | {unique_vals} unique values")
    
    # Errors
    print(f"\n  Total errors: {len(all_errors)}")
    if all_errors:
        for e in all_errors:
            print(f"    !! {e}")
    else:
        print(f"    None — all tests passed!")
    
    # Verdict
    print(f"\n{'='*70}")
    if not all_errors:
        print(f"  ✅ ALL TESTS PASSED — READY FOR FULL EXPERIMENT")
    else:
        print(f"  ❌ {len(all_errors)} ERRORS — FIX BEFORE RUNNING")
    print(f"{'='*70}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("  LLM CULTURAL BIAS SURVEY — COMPREHENSIVE TEST")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    all_errors = []
    
    # Test 1: Config & Data
    df, errors = test_config_and_data()
    all_errors.extend(errors)
    if errors:
        print("\n  CRITICAL: Config/data errors. Cannot continue.")
        sys.exit(1)
    
    # Test 2: Parser
    errors = test_parser()
    all_errors.extend(errors)
    
    # Test 3: Single model queries
    model_results, errors = test_all_models(df)
    all_errors.extend(errors)
    if errors:
        print("\n  WARNING: Model errors detected. Mini experiment may fail.")
    
    # Test 4: Mini experiment
    all_results, timing, errors = run_mini_experiment(df)
    all_errors.extend(errors)
    
    # Test 5: Checkpoint
    errors = test_checkpoint(all_results)
    all_errors.extend(errors)
    
    # Summary
    print_summary(timing, all_results, all_errors)
    
    # Save test results
    test_dir = Path(PATHS['test_results'])
    test_dir.mkdir(parents=True, exist_ok=True)
    test_path = test_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(all_results).to_csv(test_path, index=False)
    print(f"\n  Test results saved: {test_path}")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
