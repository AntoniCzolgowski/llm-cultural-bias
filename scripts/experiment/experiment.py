"""
Main experiment class for LLM Cultural Bias Survey.
Handles persona loading, query execution, and checkpointing.
"""
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import (
    PATHS, RESPONSES_PER_PERSONA, COUNTRY_MAP, EDUCATION_MAP, 
    AGE_MAP, SEX_MAP, USER_PROMPT_TEMPLATE, RANDOM_SEED
)
from models import load_model, BaseModel
from parser import parse_response


class Experiment:
    """
    Runs LLM survey experiment with checkpointing.
    """
    
    def __init__(
        self,
        model_key: str,
        responses_per_persona: int = RESPONSES_PER_PERSONA,
        output_dir: str = None,
        checkpoint_dir: str = None,
    ):
        """
        Initialize experiment.
        
        Args:
            model_key: One of 'gemma3', 'bielik', 'qwen'
            responses_per_persona: Number of queries per persona
            output_dir: Directory for final results
            checkpoint_dir: Directory for checkpoints
        """
        self.model_key = model_key
        self.responses_per_persona = responses_per_persona
        self.output_dir = Path(output_dir or PATHS['results_baseline'])
        self.checkpoint_dir = Path(checkpoint_dir or PATHS['checkpoints'])
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load human distributions
        self.personas_df = pd.read_csv(PATHS['human_distributions'])
        print(f"Loaded {len(self.personas_df)} personas from human_distributions.csv")
        
        # Initialize model (will be loaded on first run)
        self.model: BaseModel = None
        
        # Results storage
        self.results = []
        
    def _format_prompt(self, persona: pd.Series) -> str:
        """Format user prompt for a persona"""
        return USER_PROMPT_TEMPLATE.format(
            sex=SEX_MAP[persona['sex']],
            country=COUNTRY_MAP[persona['country']],
            education=EDUCATION_MAP[persona['education']],
            age_group=AGE_MAP[persona['age_group']],
        )
    
    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path for current model"""
        return self.checkpoint_dir / f"checkpoint_{self.model_key}.csv"
    
    def _load_checkpoint(self) -> set:
        """Load completed persona IDs from checkpoint"""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            df = pd.read_csv(checkpoint_path)
            completed = set(df['persona_id'].unique())
            print(f"Loaded checkpoint: {len(completed)} personas completed")
            return completed
        return set()
    
    def _save_checkpoint(self, persona_results: list):
        """Append results for one persona to checkpoint file"""
        checkpoint_path = self._get_checkpoint_path()
        df = pd.DataFrame(persona_results)
        
        # Append mode
        if checkpoint_path.exists():
            df.to_csv(checkpoint_path, mode='a', header=False, index=False)
        else:
            df.to_csv(checkpoint_path, index=False)
    
    def run_persona(self, persona_id: str) -> list:
        """
        Run all queries for a single persona.
        
        Returns:
            List of result dicts
        """
        persona = self.personas_df[self.personas_df['persona_id'] == persona_id].iloc[0]
        prompt = self._format_prompt(persona)
        
        results = []
        valid_count = 0
        
        for i in range(self.responses_per_persona):
            start_time = time.time()
            
            # Generate response
            raw_response = self.model.generate(prompt)
            
            # Parse to int
            parsed_value = parse_response(raw_response)
            
            elapsed = time.time() - start_time
            
            results.append({
                'persona_id': persona_id,
                'model': self.model_key,
                'query_idx': i,
                'raw_response': raw_response,
                'parsed_value': parsed_value,
                'is_valid': parsed_value is not None,
                'response_time_sec': round(elapsed, 3),
                'timestamp': datetime.now().isoformat(),
            })
            
            if parsed_value is not None:
                valid_count += 1
        
        # Log progress
        refusal_rate = (self.responses_per_persona - valid_count) / self.responses_per_persona
        print(f"    {persona_id}: {valid_count}/{self.responses_per_persona} valid ({refusal_rate:.1%} refusal)")
        
        return results
    
    def run(self, persona_ids: list = None, resume: bool = True):
        """
        Run experiment for specified personas (or all).
        
        Args:
            persona_ids: List of persona IDs to run (None = all)
            resume: If True, skip already completed personas from checkpoint
        """
        # Load model
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.model_key}")
        print(f"{'='*60}")
        
        self.model = load_model(self.model_key)
        
        # Get personas to process
        if persona_ids is None:
            persona_ids = self.personas_df['persona_id'].tolist()
        
        # Check for completed personas
        completed = self._load_checkpoint() if resume else set()
        remaining = [p for p in persona_ids if p not in completed]
        
        print(f"\nPersonas to process: {len(remaining)} (skipping {len(completed)} completed)")
        print(f"Queries per persona: {self.responses_per_persona}")
        print(f"Total queries: {len(remaining) * self.responses_per_persona}")
        print()
        
        # Run queries
        total_start = time.time()
        
        for i, persona_id in enumerate(remaining):
            print(f"  [{i+1}/{len(remaining)}] Processing {persona_id}...")
            
            persona_results = self.run_persona(persona_id)
            self.results.extend(persona_results)
            
            # Checkpoint after each persona
            self._save_checkpoint(persona_results)
        
        total_time = time.time() - total_start
        
        # Summary
        print(f"\n{'='*60}")
        print(f"COMPLETED: {self.model_key}")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Avg time per query: {total_time/(len(remaining)*self.responses_per_persona):.2f} sec")
        
        # Save final results
        self._save_final_results()
        
    def _save_final_results(self):
        """Save final consolidated results"""
        if not self.results:
            print("No results to save")
            return
            
        output_path = self.output_dir / f"results_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Also save summary statistics
        self._save_summary()
    
    def _save_summary(self):
        """Save summary statistics"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        summary = df.groupby('persona_id').agg({
            'parsed_value': ['count', 'mean', 'std', lambda x: x.isna().sum()],
            'response_time_sec': 'mean',
        }).round(3)
        
        summary.columns = ['n_responses', 'mean_response', 'std_response', 'n_invalid', 'avg_time_sec']
        summary['refusal_rate'] = (summary['n_invalid'] / summary['n_responses']).round(3)
        
        summary_path = self.output_dir / f"summary_{self.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")


def run_single_model(model_key: str, persona_ids: list = None, responses_per_persona: int = RESPONSES_PER_PERSONA):
    """Convenience function to run experiment for a single model"""
    exp = Experiment(
        model_key=model_key,
        responses_per_persona=responses_per_persona,
    )
    exp.run(persona_ids=persona_ids)


if __name__ == "__main__":
    # Test with minimal run
    print("Running test with 1 persona, 3 queries...")
    exp = Experiment(
        model_key='gemma3',
        responses_per_persona=3,
        output_dir=PATHS['results_pilot'],
    )
    exp.run(persona_ids=['CHN_Male_30-49_Higher'], resume=False)
