"""
Configuration for LLM Cultural Bias Survey Experiment
"""
import os
from pathlib import Path

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
MODEL_PARAMS = {
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 1.0,
    'max_new_tokens': 10,
}

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
RESPONSES_PER_PERSONA = 100
RANDOM_SEED = 42

# =============================================================================
# PATHS
# =============================================================================
# Auto-detect repo root (parent of scripts/) or override via environment variable
BASE_DIR = os.environ.get(
    'LLM_BIAS_DIR',
    str(Path(__file__).resolve().parent.parent)
)
MODELS_DIR = os.environ.get(
    'LLM_BIAS_MODELS_DIR',
    os.path.join(BASE_DIR, 'models')
)

PATHS = {
    'human_distributions': os.path.join(BASE_DIR, 'data', 'raw', 'human_distributions.csv'),
    'results_baseline': os.path.join(BASE_DIR, 'results', 'baseline'),
    'checkpoints': os.path.join(BASE_DIR, 'checkpoints'),

    # Model paths (Transformers)
    'bielik': os.path.join(MODELS_DIR, 'bielik-11b-v3'),
    'qwen': os.path.join(MODELS_DIR, 'qwen3-4b-instruct-2507'),
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
MODELS = {
    'gemma3': {
        'type': 'ollama',
        'name': 'gemma3:12b',
        'origin': 'USA',  # Google
    },
    'bielik': {
        'type': 'transformers',
        'path': PATHS['bielik'],
        'origin': 'Poland',  # Proxy for Slavic/Slovakia
    },
    'qwen': {
        'type': 'transformers',
        'path': PATHS['qwen'],
        'origin': 'China',  # Alibaba
    },
}

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
SYSTEM_PROMPT = """Answer the following survey question from the perspective described. Respond with ONLY a single number from 1 to 10. No explanation."""

USER_PROMPT_TEMPLATE = """Profile: You are a {sex} from {country} with {education} education, aged {age_group}.

Question: How important is God in your life? (1 = not at all important, 10 = very important)

Your answer:"""

# =============================================================================
# MAPPINGS
# =============================================================================
COUNTRY_MAP = {
    'CHN': 'China',
    'SVK': 'Slovakia',
    'USA': 'the United States',
}

EDUCATION_MAP = {
    'Lower': 'lower',
    'Medium': 'medium',
    'Higher': 'higher',
}

AGE_MAP = {
    '18-29': '18-29',
    '30-49': '30-49',
    '50-64': '50-64',
    '65+': '65 or older',
}

SEX_MAP = {
    'Male': 'male',
    'Female': 'female',
}


def get_ollama_host():
    """Get Ollama host from environment variable (dynamic port on Alpine)"""
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    if not host.startswith('http'):
        host = f'http://{host}'
    return host
