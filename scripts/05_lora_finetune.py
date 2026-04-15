"""
LoRA Fine-Tuning for Cross-Cultural Bias Mitigation
====================================================
Trains model on WVS opinion questions for worst-case personas
to improve cultural calibration on held-out Q164 (Importance of God).

Usage:
  python lora_finetune.py --model bielik --mode test   # Quick test (~5 min)
  python lora_finetune.py --model bielik --mode full   # Full training
  python lora_finetune.py --model qwen --mode full
"""

import argparse
import json
import os
import time
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = os.environ.get(
    'LLM_BIAS_DIR',
    str(Path(__file__).resolve().parent.parent)
)
MODELS_DIR = os.environ.get(
    'LLM_BIAS_MODELS_DIR',
    os.path.join(BASE_DIR, 'models')
)

MODEL_CONFIGS = {
    "bielik": {
        "path": os.path.join(MODELS_DIR, "bielik-11b-v3"),
        "training_data": os.path.join(BASE_DIR, "data", "training", "bielik_training_full.csv"),
        "output_dir": os.path.join(MODELS_DIR, "bielik-11b-v3-lora"),
        "target_modules": ["q_proj", "v_proj"],
    },
    "qwen": {
        "path": os.path.join(MODELS_DIR, "qwen3-4b-instruct-2507"),
        "training_data": os.path.join(BASE_DIR, "data", "training", "qwen_training_full.csv"),
        "output_dir": os.path.join(MODELS_DIR, "qwen3-4b-lora"),
        "target_modules": ["q_proj", "v_proj"],
    },
}

LORA_PARAMS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

COUNTRY_NAMES = {"CHN": "China", "SVK": "Slovakia", "USA": "United States"}

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================
class SurveyDataset(Dataset):
    """Converts WVS training CSV into tokenized prompt-response pairs."""

    def __init__(self, csv_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        df = pd.read_csv(csv_path)
        log.info(f"Loaded {len(df)} training pairs from {csv_path}")

        for _, row in df.iterrows():
            prompt = self._build_prompt(row)
            response = str(round(row["mean_response"]))
            self.samples.append((prompt, response))

    def _build_prompt(self, row):
        country_name = COUNTRY_NAMES.get(row["country"], row["country"])
        return (
            f"Answer the following survey question from the perspective described. "
            f"Respond with ONLY a single number. No explanation.\n\n"
            f"Profile: You are a {row['sex'].lower()} from {country_name} "
            f"with {row['education'].lower()} education, aged {row['age_group']}.\n\n"
            f"Question: {row['question_text']}\n"
            f"Scale: {row['scale_description']}\n\n"
            f"Answer:"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        # Tokenize full sequence (prompt + response)
        full_text = prompt + " " + response
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Create labels: mask prompt tokens with -100, only train on response
        prompt_encoding = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length
        )
        prompt_len = len(prompt_encoding["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Don't compute loss on prompt
        labels[attention_mask == 0] = -100  # Don't compute loss on padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bielik", "qwen"])
    parser.add_argument("--mode", default="test", choices=["test", "full"])
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    is_test = args.mode == "test"

    log.info(f"{'='*60}")
    log.info(f"LoRA Fine-Tuning: {args.model} ({args.mode} mode)")
    log.info(f"{'='*60}")

    # ----- Check GPU -----
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found! Run this on a GPU node.")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ----- Load tokenizer -----
    log.info(f"Loading tokenizer from {config['path']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("Set pad_token = eos_token")

    # ----- Load dataset -----
    dataset = SurveyDataset(config["training_data"], tokenizer)

    if is_test:
        # Use only first 20 samples for test
        dataset.samples = dataset.samples[:20]
        log.info(f"TEST MODE: Using {len(dataset)} samples")
    else:
        log.info(f"FULL MODE: Using {len(dataset)} samples")

    # ----- Load model -----
    log.info(f"Loading model from {config['path']}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        config["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    load_time = time.time() - t0
    log.info(f"Model loaded in {load_time:.1f}s")

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params / 1e9:.2f}B")

    # ----- Apply LoRA -----
    lora_config = LoraConfig(
        **LORA_PARAMS,
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ----- Training arguments -----
    output_dir = config["output_dir"]
    if is_test:
        output_dir = output_dir + "-test"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if is_test else 3,
        per_device_train_batch_size=2 if args.model == "qwen" else 1,
        gradient_accumulation_steps=4 if args.model == "qwen" else 8,
        learning_rate=2e-4,
        warmup_steps=10 if is_test else 50,
        weight_decay=0.01,
        logging_steps=5 if is_test else 10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # ----- Train -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    log.info("Starting training...")
    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    log.info(f"Training complete in {train_time:.1f}s ({train_time/60:.1f} min)")
    log.info(f"Training loss: {result.training_loss:.4f}")

    # ----- Save LoRA adapter -----
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"LoRA adapter saved to: {output_dir}")

    # Check adapter size
    adapter_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    ) / 1e6
    log.info(f"Adapter size: {adapter_size:.1f} MB")

    # ----- Summary -----
    summary = {
        "model": args.model,
        "mode": args.mode,
        "training_samples": len(dataset),
        "epochs": training_args.num_train_epochs,
        "training_loss": result.training_loss,
        "training_time_seconds": round(train_time, 1),
        "training_time_minutes": round(train_time / 60, 1),
        "adapter_size_mb": round(adapter_size, 1),
        "gpu": gpu_name,
        "lora_r": LORA_PARAMS["r"],
        "lora_alpha": LORA_PARAMS["lora_alpha"],
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to: {summary_path}")

    log.info(f"\n{'='*60}")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")
    log.info(f"{'='*60}")

    # ----- Quick inference test -----
    log.info("\nQuick inference test with LoRA adapter...")
    test_prompt = (
        "Answer the following survey question from the perspective described. "
        "Respond with ONLY a single number. No explanation.\n\n"
        f"Profile: You are a male from {'China' if args.model == 'qwen' else 'United States'} "
        f"with medium education, aged 30-49.\n\n"
        f"Question: All things considered, how satisfied are you with your life as a whole these days?\n"
        f"Scale: 1=Completely dissatisfied to 10=Completely satisfied\n\n"
        f"Answer:"
    )
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=5, temperature=0.7, do_sample=True
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.info(f"Test response: '{response.strip()}'")

    log.info("\nDONE!")


if __name__ == "__main__":
    main()
