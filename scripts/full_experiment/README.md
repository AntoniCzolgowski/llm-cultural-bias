# Full Experiment - LLM Cultural Bias Survey

## Overview

- **63 personas** × **100 queries** × **3 models** = **18,900 queries**
- **Estimated time: ~3 hours**
- **Estimated SU cost: ~350 SU** (at 108 SU/hour)

## Time Breakdown (estimated)

| Model | Time/query | Total time | Notes |
|-------|-----------|------------|-------|
| gemma3 | ~0.3s | ~35 min | Ollama, deterministic |
| bielik | ~0.7s | ~75 min | Transformers, some variance |
| qwen | ~0.25s | ~30 min | Transformers, mostly deterministic |
| **Total** | | **~2.5-3 hours** | + model loading |

## Before Running

### 1. Start GPU Session (aa100 partition for full experiment)

On Open OnDemand → VS Code-Server:
```
Configuration: Custom
Account: ucb757_asc1
Partition: aa100
QoS: normal
Time: 6 (hours) - buffer for safety
Cores: 8
gres: gpu:1
```

### 2. Setup Environment

```bash
module load anaconda
module load ollama
conda activate llm_bias_survey

# Verify
echo $OLLAMA_HOST
nvidia-smi
```

### 3. Navigate to folder

```bash
cd /projects/ancz7294/llm-cultural-bias/scripts/full_experiment
```

## Running the Experiment

### Full run (all 3 models)
```bash
python run_experiment.py
```

### Single model
```bash
python run_experiment.py gemma3
python run_experiment.py bielik
python run_experiment.py qwen
```

### Resume after interruption
```bash
python run_experiment.py --resume   # default, resumes from checkpoint
python run_experiment.py --no-resume  # start fresh
```

## Output

### Results
```
/projects/ancz7294/llm-cultural-bias/results/baseline/
├── results_gemma3_YYYYMMDD_HHMMSS.csv   # Full results
├── results_bielik_YYYYMMDD_HHMMSS.csv
├── results_qwen_YYYYMMDD_HHMMSS.csv
├── summary_gemma3_YYYYMMDD_HHMMSS.csv   # Per-persona summary
├── summary_bielik_YYYYMMDD_HHMMSS.csv
└── summary_qwen_YYYYMMDD_HHMMSS.csv
```

### Checkpoints (auto-saved after each persona)
```
/projects/ancz7294/llm-cultural-bias/checkpoints/
├── full_gemma3.csv
├── full_bielik.csv
└── full_qwen.csv
```

## Troubleshooting

### GPU memory error
Models are run sequentially, but if error occurs:
```bash
# Check GPU memory
nvidia-smi

# Restart session and try single model
python run_experiment.py qwen
```

### Ollama connection error
```bash
# Verify Ollama is running
module load ollama
echo $OLLAMA_HOST
ollama list
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Parameters, paths, prompt |
| `parser.py` | Response parsing |
| `run_experiment.py` | Main experiment script |
| `README.md` | This file |
