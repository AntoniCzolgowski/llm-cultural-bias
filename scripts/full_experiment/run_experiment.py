#!/usr/bin/env python3
"""
LLM Cultural Bias Survey - Full Baseline Experiment

63 personas × 100 queries × 3 models = 18,900 queries
Estimated time: ~3 hours

Usage:
    python run_experiment.py                    # Run all models
    python run_experiment.py gemma3             # Run only gemma3
    python run_experiment.py bielik qwen        # Run bielik and qwen
    python run_experiment.py --resume           # Resume from checkpoints
"""
import os
import sys
import re
import gc
import time
import argparse
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

from config import (
    PATHS, MODELS, MODEL_PARAMS, RESPONSES_PER_PERSONA,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    COUNTRY_MAP, EDUCATION_MAP, AGE_MAP, SEX_MAP, get_ollama_host
)
from parser import parse_response

# =============================================================================
# SETUP
# =============================================================================
def setup_dirs():
    Path(PATHS['results']).mkdir(parents=True, exist_ok=True)
    Path(PATHS['checkpoints']).mkdir(parents=True, exist_ok=True)

def load_personas():
    df = pd.read_csv(PATHS['human_distributions'])
    print(f"Loaded {len(df)} personas")
    return df

def format_prompt(persona: pd.Series) -> str:
    return USER_PROMPT_TEMPLATE.format(
        sex=SEX_MAP[persona['sex']],
        country=COUNTRY_MAP[persona['country']],
        education=EDUCATION_MAP[persona['education']],
        age_group=AGE_MAP[persona['age_group']],
    )

# =============================================================================
# MODEL RUNNERS
# =============================================================================
class GemmaRunner:
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=get_ollama_host())
        # Warmup
        self.client.chat(model='gemma3:12b', messages=[{'role': 'user', 'content': 'OK'}], options={'num_predict': 5})
        print("  ✓ Gemma ready")
    
    def query(self, prompt: str) -> str:
        r = self.client.chat(
            model='gemma3:12b',
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}],
            options={'temperature': MODEL_PARAMS['temperature'], 'top_k': MODEL_PARAMS['top_k'], 'num_predict': MODEL_PARAMS['max_new_tokens']}
        )
        return r['message']['content'].strip()
    
    def cleanup(self):
        pass  # Ollama manages memory


class TransformersRunner:
    def __init__(self, path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map='auto')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"  ✓ Model ready")
    
    def query(self, prompt: str) -> str:
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
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
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)


# =============================================================================
# EXPERIMENT
# =============================================================================
def get_checkpoint_path(model_key: str) -> Path:
    return Path(PATHS['checkpoints']) / f"full_{model_key}.csv"

def load_completed_personas(model_key: str) -> set:
    cp = get_checkpoint_path(model_key)
    if cp.exists():
        df = pd.read_csv(cp)
        completed = set(df['persona_id'].unique())
        print(f"  Checkpoint: {len(completed)} personas done")
        return completed
    return set()

def save_checkpoint(model_key: str, results: list):
    cp = get_checkpoint_path(model_key)
    df = pd.DataFrame(results)
    if cp.exists():
        df.to_csv(cp, mode='a', header=False, index=False)
    else:
        df.to_csv(cp, index=False)

def run_model(model_key: str, personas_df: pd.DataFrame, resume: bool = True):
    print(f"\n{'#'*70}")
    print(f"# {model_key.upper()}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load model
    print(f"Loading {model_key}...")
    config = MODELS[model_key]
    
    if config['type'] == 'ollama':
        runner = GemmaRunner()
    else:
        runner = TransformersRunner(config['path'])
    
    # Get personas to process
    all_personas = personas_df['persona_id'].tolist()
    completed = load_completed_personas(model_key) if resume else set()
    remaining = [p for p in all_personas if p not in completed]
    
    print(f"  To process: {len(remaining)} personas ({len(completed)} skipped)")
    print(f"  Queries: {len(remaining) * RESPONSES_PER_PERSONA}")
    
    if not remaining:
        print("  Nothing to do!")
        runner.cleanup()
        return
    
    # Run queries
    start_time = time.time()
    
    for i, persona_id in enumerate(remaining):
        persona = personas_df[personas_df['persona_id'] == persona_id].iloc[0]
        prompt = format_prompt(persona)
        
        results = []
        valid = 0
        
        for j in range(RESPONSES_PER_PERSONA):
            t0 = time.time()
            raw = runner.query(prompt)
            parsed = parse_response(raw)
            
            results.append({
                'persona_id': persona_id,
                'model': model_key,
                'query_idx': j,
                'raw_response': raw,
                'parsed_value': parsed,
                'is_valid': parsed is not None,
                'response_time_sec': round(time.time() - t0, 3),
                'timestamp': datetime.now().isoformat(),
            })
            
            if parsed:
                valid += 1
        
        # Checkpoint after each persona
        save_checkpoint(model_key, results)
        
        elapsed = time.time() - start_time
        rate = (i + 1) * RESPONSES_PER_PERSONA / elapsed
        eta = (len(remaining) - i - 1) * RESPONSES_PER_PERSONA / rate / 60
        
        print(f"  [{i+1}/{len(remaining)}] {persona_id}: {valid}/{RESPONSES_PER_PERSONA} valid | {rate:.1f} q/s | ETA: {eta:.0f}min")
    
    total_time = time.time() - start_time
    print(f"\n  ✓ Completed in {total_time/60:.1f} minutes")
    
    # Cleanup
    runner.cleanup()

def save_final_results(model_key: str):
    cp = get_checkpoint_path(model_key)
    if not cp.exists():
        return
    
    df = pd.read_csv(cp)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Full results
    out_path = Path(PATHS['results']) / f"results_{model_key}_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    
    # Summary
    summary = df.groupby('persona_id').agg({
        'parsed_value': ['count', 'mean', 'std', lambda x: x.isna().sum()],
        'response_time_sec': 'mean',
    }).round(3)
    summary.columns = ['n', 'mean', 'std', 'n_invalid', 'avg_time']
    summary_path = Path(PATHS['results']) / f"summary_{model_key}_{timestamp}.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='LLM Cultural Bias Full Experiment')
    parser.add_argument('models', nargs='*', default=['gemma3', 'bielik', 'qwen'])
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    args = parser.parse_args()
    
    valid = ['gemma3', 'bielik', 'qwen']
    models = [m for m in args.models if m in valid]
    
    if not models:
        print(f"Invalid models. Choose from: {valid}")
        sys.exit(1)
    
    print("="*70)
    print("LLM CULTURAL BIAS SURVEY - FULL EXPERIMENT")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {models}")
    print(f"Resume: {args.resume}")
    print(f"Personas: 63 | Queries/persona: {RESPONSES_PER_PERSONA}")
    print("="*70)
    
    setup_dirs()
    personas_df = load_personas()
    
    for model_key in models:
        try:
            run_model(model_key, personas_df, resume=args.resume)
            save_final_results(model_key)
        except Exception as e:
            print(f"\n❌ ERROR in {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results in: {PATHS['results']}")
    print("="*70)


if __name__ == "__main__":
    main()
