"""
Configuration for LLM Cultural Bias Survey - Full Experiment
"""
import os

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
RESPONSES_PER_PERSONA = 100
RANDOM_SEED = 42

MODEL_PARAMS = {
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 1.0,
    'max_new_tokens': 10,
}

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = '/projects/ancz7294/llm-cultural-bias'
MODELS_DIR = '/projects/ancz7294/models'

PATHS = {
    'human_distributions': f'{BASE_DIR}/data/raw/human_distributions.csv',
    'results': f'{BASE_DIR}/results/baseline',
    'checkpoints': f'{BASE_DIR}/checkpoints',
    'bielik': f'{MODELS_DIR}/bielik-11b-v3',
    'qwen': f'{MODELS_DIR}/qwen3-4b-instruct-2507',
}

# =============================================================================
# MODELS
# =============================================================================
MODELS = {
    'gemma3': {'type': 'ollama', 'name': 'gemma3:12b', 'origin': 'USA'},
    'bielik': {'type': 'transformers', 'path': PATHS['bielik'], 'origin': 'Poland'},
    'qwen': {'type': 'transformers', 'path': PATHS['qwen'], 'origin': 'China'},
}

# =============================================================================
# PROMPT
# =============================================================================
SYSTEM_PROMPT = """Answer the following survey question from the perspective described. Respond with ONLY a single number from 1 to 10. No explanation."""

USER_PROMPT_TEMPLATE = """Profile: You are a {sex} from {country} with {education} education, aged {age_group}.

Question: How important is God in your life? (1 = not at all important, 10 = very important)

Your answer:"""

# =============================================================================
# MAPPINGS
# =============================================================================
COUNTRY_MAP = {'CHN': 'China', 'SVK': 'Slovakia', 'USA': 'the United States'}
EDUCATION_MAP = {'Lower': 'lower', 'Medium': 'medium', 'Higher': 'higher'}
AGE_MAP = {'18-29': '18-29', '30-49': '30-49', '50-64': '50-64', '65+': '65 or older'}
SEX_MAP = {'Male': 'male', 'Female': 'female'}

def get_ollama_host():
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    return f'http://{host}' if not host.startswith('http') else host
