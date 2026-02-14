#!/usr/bin/env python3
"""
Full Baseline Experiment for LLM Cultural Bias Survey

Runs: 63 personas × 100 queries × 3 models = 18,900 queries
Expected time: ~4-6 hours total

Usage:
    python run_baseline.py              # Run all models sequentially
    python run_baseline.py gemma3       # Run only gemma3
    python run_baseline.py --resume     # Resume from checkpoints
"""
import sys
import time
import argparse
from datetime import datetime

from config import PATHS, RESPONSES_PER_PERSONA
from experiment import Experiment


def run_baseline(model_keys: list = None, resume: bool = True):
    """
    Run full baseline experiment.
    
    Args:
        model_keys: List of model keys to run. Default: all three.
        resume: If True, resume from checkpoints.
    """
    if model_keys is None:
        model_keys = ['gemma3', 'bielik', 'qwen']
    
    print("="*70)
    print("LLM CULTURAL BIAS SURVEY - FULL BASELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {model_keys}")
    print(f"Resume from checkpoint: {resume}")
    print("="*70)
    
    for model_key in model_keys:
        print(f"\n{'#'*70}")
        print(f"# STARTING: {model_key.upper()}")
        print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")
        
        try:
            exp = Experiment(
                model_key=model_key,
                responses_per_persona=RESPONSES_PER_PERSONA,
                output_dir=PATHS['results_baseline'],
                checkpoint_dir=PATHS['checkpoints'],
            )
            exp.run(resume=resume)
            
        except Exception as e:
            print(f"\n❌ ERROR in {model_key}: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing with next model...")
            continue
    
    print("\n" + "="*70)
    print("BASELINE EXPERIMENT COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results in: {PATHS['results_baseline']}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLM Cultural Bias baseline experiment')
    parser.add_argument('models', nargs='*', default=['gemma3', 'bielik', 'qwen'],
                        help='Models to run (default: all)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoints (default: True)')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Start fresh, ignore checkpoints')
    
    args = parser.parse_args()
    
    valid_models = ['gemma3', 'bielik', 'qwen']
    models = [m for m in args.models if m in valid_models]
    
    if not models:
        print(f"No valid models specified. Choose from: {valid_models}")
        sys.exit(1)
    
    run_baseline(models, resume=args.resume)
