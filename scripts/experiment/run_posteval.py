#!/usr/bin/env python3
"""
Post-Evaluation: Re-run baseline with LoRA fine-tuned models
=============================================================
Runs Bielik+LoRA and Qwen+LoRA on ALL 63 personas × 100 responses.
Results saved separately from baseline for comparison.

Usage:
    python run_posteval.py                    # Both models
    python run_posteval.py --model bielik     # Only Bielik+LoRA
    python run_posteval.py --model qwen       # Only Qwen+LoRA
"""

import argparse
import csv
import os
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# =============================================================================
# CONFIG (matching baseline exactly)
# =============================================================================
BASE_DIR = "/projects/ancz7294/llm-cultural-bias"
MODELS_DIR = "/projects/ancz7294/models"

HUMAN_DIST_PATH = f"{BASE_DIR}/data/raw/human_distributions.csv"
RESULTS_DIR = f"{BASE_DIR}/results/posteval"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/posteval"

MODEL_CONFIGS = {
    "bielik": {
        "base_path": f"{MODELS_DIR}/bielik-11b-v3",
        "lora_path": f"{MODELS_DIR}/bielik-11b-v3-lora",
        "save_name": "bielik_lora",
    },
    "qwen": {
        "base_path": f"{MODELS_DIR}/qwen3-4b-instruct-2507",
        "lora_path": f"{MODELS_DIR}/qwen3-4b-lora",
        "save_name": "qwen_lora",
    },
}

# Must match baseline exactly
RESPONSES_PER_PERSONA = 100
MODEL_PARAMS = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1.0,
    "max_new_tokens": 10,
}

SYSTEM_PROMPT = "Answer the following survey question from the perspective described. Respond with ONLY a single number from 1 to 10. No explanation."

COUNTRY_NAMES = {
    "CHN": "China",
    "SVK": "Slovakia",
    "USA": "the United States",
}

AGE_MAP = {
    "18-29": "18-29",
    "30-49": "30-49",
    "50-64": "50-64",
    "65+": "65 or older",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# MODEL LOADING (base + LoRA)
# =============================================================================
def load_model_with_lora(model_key):
    """Load base model with LoRA adapter for inference."""
    config = MODEL_CONFIGS[model_key]

    log.info(f"Loading {model_key} base model from {config['base_path']}...")

    tokenizer = AutoTokenizer.from_pretrained(config["base_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 4-bit quantization (same as baseline)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter
    log.info(f"Loading LoRA adapter from {config['lora_path']}...")
    model = PeftModel.from_pretrained(model, config["lora_path"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  ✓ {model_key}+LoRA loaded ({total_params/1e9:.2f}B params, {trainable} trainable)")

    return model, tokenizer


# =============================================================================
# INFERENCE
# =============================================================================
def generate_response(model, tokenizer, user_prompt):
    """Generate single response (matching baseline exactly)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MODEL_PARAMS["max_new_tokens"],
            temperature=MODEL_PARAMS["temperature"],
            top_k=MODEL_PARAMS["top_k"],
            top_p=MODEL_PARAMS["top_p"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def parse_response(raw_response):
    """Extract integer 1-10 from model response."""
    import re

    # Try to find a number 1-10
    numbers = re.findall(r"\b(10|[1-9])\b", raw_response)
    if numbers:
        return int(numbers[0])
    return None


# =============================================================================
# CHECKPOINT
# =============================================================================
def load_checkpoint(model_key):
    """Load checkpoint: returns set of completed persona_ids."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_key}_done.txt")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            done = set(line.strip() for line in f if line.strip())
        log.info(f"Checkpoint: {len(done)} personas already completed")
        return done
    return set()


def save_checkpoint(model_key, persona_id):
    """Append completed persona to checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_key}_done.txt")
    with open(ckpt_path, "a") as f:
        f.write(persona_id + "\n")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_posteval(model_key):
    """Run post-evaluation for one model."""
    config = MODEL_CONFIGS[model_key]
    save_name = config["save_name"]

    # Create dirs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load personas from human_distributions.csv
    human_df = pd.read_csv(HUMAN_DIST_PATH)
    personas = human_df[["persona_id", "country", "sex", "age_group", "education"]].drop_duplicates()
    log.info(f"Loaded {len(personas)} personas")

    # Load checkpoint
    done = load_checkpoint(save_name)
    remaining = [p for _, p in personas.iterrows() if p["persona_id"] not in done]
    log.info(f"Remaining: {len(remaining)} personas ({len(done)} done)")

    if not remaining:
        log.info("All personas completed!")
        return

    # Load model
    model, tokenizer = load_model_with_lora(model_key)

    # Output CSV
    results_path = os.path.join(RESULTS_DIR, f"{save_name}_responses.csv")
    file_exists = os.path.exists(results_path)

    total_queries = len(remaining) * RESPONSES_PER_PERSONA
    queries_done = 0
    start_time = time.time()

    log.info(f"Starting: {len(remaining)} personas × {RESPONSES_PER_PERSONA} = {total_queries} queries")

    with open(results_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["model", "persona_id", "country", "sex", "age_group",
                             "education", "response_idx", "raw_response", "parsed_value"])

        for persona_idx, persona in enumerate(remaining):
            pid = persona["persona_id"]
            country_name = COUNTRY_NAMES.get(persona["country"], persona["country"])

            user_prompt = (
                f"Profile: You are a {persona['sex'].lower()} from {country_name} "
                f"with {persona['education'].lower()} education, aged {AGE_MAP[persona['age_group']]}.\n\n"
                f"Question: How important is God in your life? "
                f"(1 = not at all important, 10 = very important)\n\n"
                f"Your answer:"
            )

            valid_count = 0
            for resp_idx in range(RESPONSES_PER_PERSONA):
                raw = generate_response(model, tokenizer, user_prompt)
                parsed = parse_response(raw)
                if parsed is not None:
                    valid_count += 1

                writer.writerow([
                    save_name, pid, persona["country"], persona["sex"],
                    persona["age_group"], persona["education"],
                    resp_idx, raw, parsed
                ])
                queries_done += 1

            # Flush after each persona
            csvfile.flush()
            save_checkpoint(save_name, pid)

            # Progress
            elapsed = time.time() - start_time
            rate = queries_done / elapsed if elapsed > 0 else 0
            eta = (total_queries - queries_done) / rate if rate > 0 else 0

            log.info(
                f"  [{persona_idx+1}/{len(remaining)}] {pid}: "
                f"{valid_count}/{RESPONSES_PER_PERSONA} valid | "
                f"{rate:.1f} q/s | ETA {eta/60:.0f} min"
            )

    elapsed_total = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"  {save_name} COMPLETE")
    log.info(f"  Total queries: {queries_done}")
    log.info(f"  Total time: {elapsed_total/60:.1f} min")
    log.info(f"  Results: {results_path}")
    log.info(f"{'='*60}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bielik", "qwen"], default=None,
                        help="Run single model. Default: both.")
    parser.add_argument("--mode", choices=["full", "test"], default="full",
                        help="test=5 queries per model, full=all 63 personas × 100")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"POST-EVALUATION: LoRA Fine-Tuned Models ({args.mode} mode)")
    log.info(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    models_to_run = [args.model] if args.model else ["qwen", "bielik"]

    if args.mode == "test":
        # Quick test: 5 responses per model, no checkpoint/save
        for m in models_to_run:
            log.info(f"\n=== Testing {m}+LoRA ===")
            model, tok = load_model_with_lora(m)
            country = "China" if m == "qwen" else "the United States"
            prompt = (
                f"Profile: You are a male from {country} "
                f"with medium education, aged 30-49.\n\n"
                f"Question: How important is God in your life? "
                f"(1 = not at all important, 10 = very important)\n\n"
                f"Your answer:"
            )
            for i in range(5):
                raw = generate_response(model, tok, prompt)
                parsed = parse_response(raw)
                log.info(f"  Response {i+1}: raw={repr(raw)}, parsed={parsed}")
            del model, tok
            torch.cuda.empty_cache()
        log.info("\n=== TEST PASSED ===")
    else:
        for m in models_to_run:
            run_posteval(m)

    log.info(f"\nAll done! Results in: {RESULTS_DIR}")
