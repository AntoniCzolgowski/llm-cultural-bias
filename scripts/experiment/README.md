# LLM Cultural Bias Experiment

## Quick Start

```bash
# 1. Setup environment (na początku sesji)
module load anaconda
module load ollama
conda activate llm_bias_survey

# 2. Go to experiment folder
cd /projects/ancz7294/llm-cultural-bias/scripts/experiment

# 3. Quick test (verify everything works)
python quick_test.py

# 4. Run pilot (3 personas × 10 queries × 3 models = 90 queries)
python run_pilot.py

# 5. Run full baseline (63 personas × 100 queries × 3 models)
python run_baseline.py
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Parameters, paths, mappings |
| `parser.py` | Response parser (extracts 1-10) |
| `models.py` | Unified wrapper for Ollama + Transformers |
| `experiment.py` | Main experiment class with checkpointing |
| `quick_test.py` | Verify setup before running |
| `run_pilot.py` | Pilot test (~10-15 min) |
| `run_baseline.py` | Full experiment (~4-6 hours) |

## Models

| Model | Type | Origin | Path |
|-------|------|--------|------|
| gemma3:12b | Ollama | USA (Google) | CURC-provided |
| bielik | Transformers | Poland | /projects/ancz7294/models/bielik-11b-v3 |
| qwen | Transformers | China (Alibaba) | /projects/ancz7294/models/qwen3-4b-instruct-2507 |

## Output

Results are saved to:
- Pilot: `/projects/ancz7294/llm-cultural-bias/results/pilot/`
- Baseline: `/projects/ancz7294/llm-cultural-bias/results/baseline/`
- Checkpoints: `/projects/ancz7294/llm-cultural-bias/checkpoints/`

Checkpoint happens after each persona → safe to interrupt and resume.

## Running Individual Models

```bash
# Run only gemma3
python run_pilot.py gemma3

# Run bielik and qwen
python run_pilot.py bielik qwen

# Full baseline with only one model
python run_baseline.py gemma3
```
