#!/usr/bin/env python3
"""
Pilot Test for LLM Cultural Bias Survey

Runs: 3 personas × 10 queries × 3 models = 90 queries
Expected time: ~10-15 minutes

Usage:
    python run_pilot.py              # Run all models
    python run_pilot.py gemma3       # Run only gemma3
    python run_pilot.py bielik qwen  # Run bielik and qwen
"""
import sys
import time
from datetime import datetime

from config import PILOT_PERSONAS, PATHS, PILOT_RESPONSES_PER_PERSONA
from experiment import Experiment


def run_pilot(model_keys: list = None):
    """
    Run pilot test for specified models.
    
    Args:
        model_keys: List of model keys to test. Default: all three.
    """
    if model_keys is None:
        model_keys = ['gemma3', 'bielik', 'qwen']
    
    print("="*70)
    print("LLM CULTURAL BIAS SURVEY - PILOT TEST")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Personas: {PILOT_PERSONAS}")
    print(f"Queries per persona: {PILOT_RESPONSES_PER_PERSONA}")
    print(f"Models: {model_keys}")
    print(f"Total queries: {len(PILOT_PERSONAS) * PILOT_RESPONSES_PER_PERSONA * len(model_keys)}")
    print("="*70)
    
    results_summary = {}
    total_start = time.time()
    
    for model_key in model_keys:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_key.upper()}")
        print(f"{'#'*70}")
        
        model_start = time.time()
        
        try:
            exp = Experiment(
                model_key=model_key,
                responses_per_persona=PILOT_RESPONSES_PER_PERSONA,
                output_dir=PATHS['results_pilot'],
                checkpoint_dir=PATHS['checkpoints'],
            )
            exp.run(persona_ids=PILOT_PERSONAS, resume=False)
            
            model_time = time.time() - model_start
            n_queries = len(PILOT_PERSONAS) * PILOT_RESPONSES_PER_PERSONA
            
            results_summary[model_key] = {
                'status': 'SUCCESS',
                'time_sec': model_time,
                'queries': n_queries,
                'sec_per_query': model_time / n_queries,
            }
            
        except Exception as e:
            results_summary[model_key] = {
                'status': f'FAILED: {e}',
                'time_sec': time.time() - model_start,
            }
            print(f"\n❌ ERROR in {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("PILOT TEST SUMMARY")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    
    print(f"{'Model':<15} {'Status':<15} {'Time (s)':<12} {'Queries':<10} {'Sec/Query':<10}")
    print("-"*62)
    
    for model_key, stats in results_summary.items():
        if stats['status'] == 'SUCCESS':
            print(f"{model_key:<15} {'✓ SUCCESS':<15} {stats['time_sec']:<12.1f} {stats['queries']:<10} {stats['sec_per_query']:<10.2f}")
        else:
            print(f"{model_key:<15} {'✗ FAILED':<15} {stats['time_sec']:<12.1f}")
    
    print()
    print(f"Results saved to: {PATHS['results_pilot']}")
    print(f"Checkpoints in: {PATHS['checkpoints']}")
    print("="*70)
    
    # Cost estimation for full experiment
    if any(s['status'] == 'SUCCESS' for s in results_summary.values()):
        print("\n📊 COST ESTIMATION FOR FULL EXPERIMENT:")
        print("-"*50)
        
        full_personas = 63
        full_queries_per_persona = 100
        
        for model_key, stats in results_summary.items():
            if stats['status'] == 'SUCCESS':
                full_time = stats['sec_per_query'] * full_personas * full_queries_per_persona
                gpu_hours = full_time / 3600
                sus = gpu_hours * 108  # 108 SU/hour for aa100
                print(f"{model_key}:")
                print(f"  Time: {full_time/3600:.1f} hours ({full_time/60:.0f} min)")
                print(f"  SU cost: ~{sus:.0f} SU")
        print("-"*50)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        models = sys.argv[1:]
        valid_models = ['gemma3', 'bielik', 'qwen']
        models = [m for m in models if m in valid_models]
        if not models:
            print(f"Invalid models. Choose from: {valid_models}")
            sys.exit(1)
        run_pilot(models)
    else:
        run_pilot()
