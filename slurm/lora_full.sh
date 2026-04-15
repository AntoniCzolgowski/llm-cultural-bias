#!/bin/bash
#SBATCH --job-name=lora-full
#SBATCH --account=${SLURM_ACCOUNT:-YOUR_ACCOUNT}
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=85G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "============================================"
echo "LoRA Full Training - $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Setup
module load anaconda
conda activate llm_bias_survey

cd "${LLM_BIAS_DIR}/scripts"

# Train Qwen (4B, ~15-30 min)
echo ""
echo ">>> [1/2] Full LoRA training: Qwen..."
python 05_lora_finetune.py --model qwen --mode full

# Train Bielik (11B, ~30-60 min)
echo ""
echo ">>> [2/2] Full LoRA training: Bielik..."
python 05_lora_finetune.py --model bielik --mode full

echo ""
echo "============================================"
echo "Full training complete - $(date)"
echo "============================================"
echo ""
echo "Adapter locations:"
echo "  Qwen:   ${LLM_BIAS_MODELS_DIR}/qwen3-4b-lora/"
echo "  Bielik: ${LLM_BIAS_MODELS_DIR}/bielik-11b-v3-lora/"
echo ""
echo "Next: Run post-evaluation baseline with:"
echo "  python 06_run_posteval.py"
