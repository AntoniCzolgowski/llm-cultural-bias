#!/bin/bash
#SBATCH --job-name=lora-test
#SBATCH --account=ucb757_asc1
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "============================================"
echo "LoRA Test Run - $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# Setup
module load anaconda
conda activate llm_bias_survey

cd /projects/ancz7294/llm-cultural-bias/scripts/experiment

# Step 1: Check dependencies
echo ""
echo ">>> Checking dependencies..."
python -c "
import peft; print(f'peft: {peft.__version__}')
" 2>/dev/null || {
    echo '>>> Installing peft...'
    pip install peft --quiet --break-system-packages 2>/dev/null || pip install peft --quiet
}

python -c "
import torch, transformers, peft, pandas
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Step 2: Test with Bielik (smaller test first if Qwen is 4B)
echo ""
echo ">>> Testing LoRA on Qwen (smaller model, faster test)..."
python lora_finetune.py --model qwen --mode test

echo ""
echo ">>> Testing LoRA on Bielik..."
python lora_finetune.py --model bielik --mode test

echo ""
echo "============================================"
echo "Test complete - $(date)"
echo "============================================"
