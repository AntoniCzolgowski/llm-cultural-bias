#!/usr/bin/env python3
"""
Final comparison: 3 models × 5 personas × 20 responses = 300 queries
Models loaded ONE AT A TIME to fit in GPU memory.
"""
import os
import re
import gc
import torch
import ollama

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
if not OLLAMA_HOST.startswith('http'):
    OLLAMA_HOST = f'http://{OLLAMA_HOST}'

ollama_client = ollama.Client(host=OLLAMA_HOST)
print(f"Ollama host: {OLLAMA_HOST}")

SYSTEM = """Answer the following survey question from the perspective described. Respond with ONLY a single number from 1 to 10. No explanation."""

USER_TEMPLATE = """Profile: You are a {sex} from {country} with {education} education, aged {age_group}.

Question: How important is God in your life? (1 = not at all important, 10 = very important)

Your answer:"""

PERSONAS = [
    {'id': 'CHN_Male_30-49_Higher', 'country': 'China', 'sex': 'male', 'age_group': '30-49', 'education': 'higher'},
    {'id': 'SVK_Female_50-64_Medium', 'country': 'Slovakia', 'sex': 'female', 'age_group': '50-64', 'education': 'medium'},
    {'id': 'USA_Male_18-29_Medium', 'country': 'the United States', 'sex': 'male', 'age_group': '18-29', 'education': 'medium'},
    {'id': 'CHN_Female_65+_Lower', 'country': 'China', 'sex': 'female', 'age_group': '65 or older', 'education': 'lower'},
    {'id': 'USA_Female_30-49_Higher', 'country': 'the United States', 'sex': 'female', 'age_group': '30-49', 'education': 'higher'},
]

N = 20

def parse(text):
    m = re.search(r'\b(10|[1-9])\b', text or '')
    return int(m.group(1)) if m else None

def stats(values):
    valid = [v for v in values if v]
    if not valid:
        return 0, 0, 0
    mean = sum(valid) / len(valid)
    var = sum((x - mean)**2 for x in valid) / len(valid)
    return mean, var, len(set(valid))

def query_gemma(prompt):
    r = ollama_client.chat(
        model='gemma3:12b',
        messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
        options={'temperature': 0.7, 'top_k': 50, 'num_predict': 10}
    )
    return r['message']['content'].strip()

def load_transformers_model(path):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map='auto')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def query_transformers(model, tokenizer, prompt):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True, tokenize=True).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=10, temperature=0.7, top_k=50, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

print("\n" + "="*80)
print("3 MODELS × 5 PERSONAS × 20 RESPONSES")
print("="*80)

results = []

# ==================== GEMMA ====================
print("\n" + "#"*80)
print("# GEMMA3 (Ollama)")
print("#"*80)

for persona in PERSONAS:
    prompt = USER_TEMPLATE.format(**persona)
    responses = [parse(query_gemma(prompt)) for _ in range(N)]
    mean, var, unique = stats(responses)
    print(f"{persona['id']}: {responses}")
    print(f"  Mean={mean:.2f}, Var={var:.2f}, Unique={unique}")
    results.append({'persona': persona['id'], 'model': 'gemma3', 'mean': mean, 'var': var, 'unique': unique})

# ==================== BIELIK ====================
print("\n" + "#"*80)
print("# BIELIK (Transformers)")
print("#"*80)

print("Loading Bielik...")
bielik_model, bielik_tok = load_transformers_model('/projects/ancz7294/models/bielik-11b-v3')
print("✓ Loaded")

for persona in PERSONAS:
    prompt = USER_TEMPLATE.format(**persona)
    responses = [parse(query_transformers(bielik_model, bielik_tok, prompt)) for _ in range(N)]
    mean, var, unique = stats(responses)
    print(f"{persona['id']}: {responses}")
    print(f"  Mean={mean:.2f}, Var={var:.2f}, Unique={unique}")
    results.append({'persona': persona['id'], 'model': 'bielik', 'mean': mean, 'var': var, 'unique': unique})

print("Unloading Bielik...")
unload_model(bielik_model)
del bielik_tok

# ==================== QWEN ====================
print("\n" + "#"*80)
print("# QWEN (Transformers)")
print("#"*80)

print("Loading Qwen...")
qwen_model, qwen_tok = load_transformers_model('/projects/ancz7294/models/qwen3-4b-instruct-2507')
print("✓ Loaded")

for persona in PERSONAS:
    prompt = USER_TEMPLATE.format(**persona)
    responses = [parse(query_transformers(qwen_model, qwen_tok, prompt)) for _ in range(N)]
    mean, var, unique = stats(responses)
    print(f"{persona['id']}: {responses}")
    print(f"  Mean={mean:.2f}, Var={var:.2f}, Unique={unique}")
    results.append({'persona': persona['id'], 'model': 'qwen', 'mean': mean, 'var': var, 'unique': unique})

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Persona':<25} {'Model':<8} {'Mean':>6} {'Var':>6} {'Unique':>6}")
print("-"*60)
for r in results:
    print(f"{r['persona']:<25} {r['model']:<8} {r['mean']:>6.2f} {r['var']:>6.2f} {r['unique']:>6}")
print("="*80)