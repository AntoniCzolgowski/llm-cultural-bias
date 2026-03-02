#!/usr/bin/env python3
"""
LLM Cultural Bias Survey - Full Baseline Experiment
====================================================
63 personas × 100 queries × 3 models = 18,900 queries
Estimated time: ~80 minutes (based on test: gemma 0.29s, bielik 0.31s, qwen 0.15s)

Checkpoints after each persona (100 queries). Resume-safe.

Usage:
    python run_baseline.py                  # All 3 models
    python run_baseline.py gemma3           # Single model
    python run_baseline.py bielik qwen      # Two models
    python run_baseline.py --no-resume      # Start fresh (clears checkpoints)
"""

import os
import sys
import re
import gc
import time
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = '/projects/ancz7294/llm-cultural-bias'
MODELS_DIR = '/projects/ancz7294/models'

RESPONSES_PER_PERSONA = 100

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

MODELS = {
    'gemma3': {'type': 'ollama', 'name': 'gemma3:12b', 'origin': 'USA'},
    'bielik': {'type': 'transformers', 'path': f'{MODELS_DIR}/bielik-11b-v3', 'origin': 'Poland'},
    'qwen':   {'type': 'transformers', 'path': f'{MODELS_DIR}/qwen3-4b-instruct-2507', 'origin': 'China'},
}

PATHS = {
    'human_distributions': f'{BASE_DIR}/data/raw/human_distributions.csv',
    'results':      f'{BASE_DIR}/results/baseline',
    'checkpoints':  f'{BASE_DIR}/checkpoints',
}

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

def setup_dirs():
    Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
    Path(PATHS['checkpoints']).mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL RUNNERS
# =============================================================================
class OllamaRunner:
    def __init__(self, model_name):
        import ollama
        self.client = ollama.Client(host=get_ollama_host())
        self.model_name = model_name
        # Warmup query to load model into GPU
        self.client.chat(model=self.model_name, 
                         messages=[{'role': 'user', 'content': 'OK'}], 
                         options={'num_predict': 5})
        print(f"    Model ready")

    def query(self, prompt: str) -> str:
        r = self.client.chat(
            model=self.model_name,
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
        return r['message']['content'].strip()

    def cleanup(self):
        pass


class TransformersRunner:
    def __init__(self, path):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb, device_map='auto'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"    Model ready")

    def query(self, prompt: str) -> str:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors='pt', add_generation_prompt=True, tokenize=True
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=MODEL_PARAMS['max_new_tokens'],
                temperature=MODEL_PARAMS['temperature'],
                top_k=MODEL_PARAMS['top_k'],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    def cleanup(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================
def checkpoint_path(model_key: str) -> Path:
    return Path(PATHS['checkpoints']) / f"baseline_{model_key}.csv"

def load_completed(model_key: str) -> set:
    cp = checkpoint_path(model_key)
    if cp.exists():
        df = pd.read_csv(cp)
        completed = set(df['persona_id'].unique())
        return completed
    return set()

def save_persona_checkpoint(model_key: str, results: list):
    """Append one persona's results to checkpoint file."""
    cp = checkpoint_path(model_key)
    df = pd.DataFrame(results)
    if cp.exists():
        df.to_csv(cp, mode='a', header=False, index=False)
    else:
        df.to_csv(cp, index=False)

def clear_checkpoint(model_key: str):
    cp = checkpoint_path(model_key)
    if cp.exists():
        cp.unlink()
        print(f"    Cleared checkpoint: {cp}")

# =============================================================================
# RUN ONE MODEL
# =============================================================================
def run_model(model_key: str, personas_df: pd.DataFrame, resume: bool):
    config = MODELS[model_key]
    
    print(f"\n{'#'*70}")
    print(f"#  MODEL: {model_key.upper()} ({config['origin']})")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # Handle resume
    if not resume:
        clear_checkpoint(model_key)
    completed = load_completed(model_key) if resume else set()

    all_personas = personas_df['persona_id'].tolist()
    remaining = [p for p in all_personas if p not in completed]

    print(f"    Completed: {len(completed)} | Remaining: {len(remaining)} | Queries: {len(remaining) * RESPONSES_PER_PERSONA}")

    if not remaining:
        print(f"    Nothing to do — all personas already completed!")
        return

    # Load model
    print(f"    Loading {model_key}...")
    if config['type'] == 'ollama':
        runner = OllamaRunner(config['name'])
    else:
        runner = TransformersRunner(config['path'])

    # Run queries
    model_start = time.time()
    total_queries = 0
    total_valid = 0

    for i, persona_id in enumerate(remaining):
        persona_row = personas_df[personas_df['persona_id'] == persona_id].iloc[0]
        prompt = format_prompt(persona_row)

        persona_results = []
        persona_valid = 0

        for j in range(RESPONSES_PER_PERSONA):
            t0 = time.time()
            try:
                raw = runner.query(prompt)
            except Exception as e:
                raw = f"ERROR: {e}"
            elapsed = time.time() - t0

            parsed = parse_response(raw)
            if parsed is not None:
                persona_valid += 1

            persona_results.append({
                'persona_id': persona_id,
                'model': model_key,
                'model_origin': config['origin'],
                'country': persona_row['country'],
                'sex': persona_row['sex'],
                'age_group': persona_row['age_group'],
                'education': persona_row['education'],
                'query_idx': j,
                'raw_response': raw,
                'parsed_value': parsed,
                'is_valid': parsed is not None,
                'response_time_sec': round(elapsed, 3),
                'timestamp': datetime.now().isoformat(),
            })

        # Checkpoint after each persona
        save_persona_checkpoint(model_key, persona_results)
        total_queries += RESPONSES_PER_PERSONA
        total_valid += persona_valid

        # Progress
        elapsed_total = time.time() - model_start
        rate = total_queries / elapsed_total
        eta = (len(remaining) - i - 1) * RESPONSES_PER_PERSONA / rate / 60

        # Get unique values for this persona
        vals = [r['parsed_value'] for r in persona_results if r['parsed_value'] is not None]
        unique = len(set(vals))
        mean_val = np.mean(vals) if vals else float('nan')

        print(f"    [{i+1:3d}/{len(remaining)}] {persona_id:30s} | "
              f"valid={persona_valid:3d}/{RESPONSES_PER_PERSONA} | "
              f"mean={mean_val:4.1f} | unique={unique} | "
              f"{rate:.1f} q/s | ETA: {eta:.0f}min")

    # Summary
    elapsed_total = time.time() - model_start
    print(f"\n    Completed {model_key} in {elapsed_total/60:.1f} min | "
          f"{total_queries} queries | {total_valid}/{total_queries} valid "
          f"({total_queries - total_valid} refusals)")

    # Cleanup
    runner.cleanup()

# =============================================================================
# SAVE FINAL RESULTS
# =============================================================================
def save_final_results(model_key: str):
    """Copy checkpoint to timestamped results file + generate summary."""
    cp = checkpoint_path(model_key)
    if not cp.exists():
        print(f"    No checkpoint for {model_key}, skipping save.")
        return

    df = pd.read_csv(cp)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(PATHS['results'])

    # Full results
    results_path = results_dir / f"results_{model_key}_{ts}.csv"
    df.to_csv(results_path, index=False)
    print(f"    Saved results: {results_path} ({len(df)} rows)")

    # Per-persona summary
    valid_df = df[df['is_valid'] == True]
    summary_rows = []
    for pid, group in valid_df.groupby('persona_id'):
        vals = group['parsed_value'].values
        summary_rows.append({
            'persona_id': pid,
            'model': model_key,
            'n_valid': len(vals),
            'n_invalid': RESPONSES_PER_PERSONA - len(vals),
            'mean': round(np.mean(vals), 3),
            'std': round(np.std(vals, ddof=1), 3) if len(vals) > 1 else 0,
            'median': np.median(vals),
            'min': int(np.min(vals)),
            'max': int(np.max(vals)),
            'unique_values': len(set(vals)),
            'avg_response_time': round(group['response_time_sec'].mean(), 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / f"summary_{model_key}_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"    Saved summary: {summary_path} ({len(summary_df)} personas)")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='LLM Cultural Bias — Full Baseline Experiment')
    parser.add_argument('models', nargs='*', default=['gemma3', 'bielik', 'qwen'],
                        help='Models to run (default: all)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoints (default)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start fresh, clear checkpoints')
    args = parser.parse_args()

    valid_models = ['gemma3', 'bielik', 'qwen']
    models = [m for m in args.models if m in valid_models]
    if not models:
        print(f"No valid models. Choose from: {valid_models}")
        sys.exit(1)

    print("=" * 70)
    print("  LLM CULTURAL BIAS SURVEY — FULL BASELINE EXPERIMENT")
    print("=" * 70)
    print(f"  Start:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:   {models}")
    print(f"  Resume:   {args.resume}")
    print(f"  Personas: 63 | Queries/persona: {RESPONSES_PER_PERSONA} | Total: {63 * RESPONSES_PER_PERSONA * len(models)}")
    print("=" * 70)

    setup_dirs()
    personas_df = pd.read_csv(PATHS['human_distributions'])
    print(f"  Loaded {len(personas_df)} personas from CSV")

    experiment_start = time.time()

    for model_key in models:
        try:
            run_model(model_key, personas_df, resume=args.resume)
            save_final_results(model_key)
        except Exception as e:
            print(f"\n  !! ERROR in {model_key}: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with next model...\n")
            continue

    total_time = time.time() - experiment_start

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print(f"  End:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration:  {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  Results:   {PATHS['results']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
