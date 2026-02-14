"""
Model wrappers for LLM Cultural Bias Survey.
Provides unified interface for Ollama (gemma3) and Transformers (Bielik, Qwen) models.
"""
import os
import torch
from abc import ABC, abstractmethod

from config import MODEL_PARAMS, SYSTEM_PROMPT, get_ollama_host


class BaseModel(ABC):
    """Abstract base class for model wrappers"""
    
    @abstractmethod
    def generate(self, user_prompt: str) -> str:
        """Generate response for given prompt"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name for logging"""
        pass


class OllamaModel(BaseModel):
    """Wrapper for Ollama models (gemma3:12b)"""
    
    def __init__(self, model_name: str):
        import ollama
        self.model_name = model_name
        self.client = ollama.Client(host=get_ollama_host())
        
        # Warm up - first call loads model to GPU
        print(f"Loading {model_name} via Ollama...")
        try:
            self.client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Say OK'}],
                options={'num_predict': 5}
            )
            print(f"  ✓ {model_name} loaded")
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
            raise
    
    def generate(self, user_prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': MODEL_PARAMS['temperature'],
                    'top_k': MODEL_PARAMS['top_k'],
                    'top_p': MODEL_PARAMS['top_p'],
                    'num_predict': MODEL_PARAMS['max_new_tokens'],
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"  Error in Ollama generate: {e}")
            return ""
    
    def get_name(self) -> str:
        return self.model_name.replace(':', '_')


class TransformersModel(BaseModel):
    """Wrapper for HuggingFace Transformers models (Bielik, Qwen)"""
    
    def __init__(self, model_path: str, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model_name = model_name
        self.model_path = model_path
        
        print(f"Loading {model_name} from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization for memory efficiency
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        print(f"  ✓ {model_name} loaded")
    
    def generate(self, user_prompt: str) -> str:
        """Generate response using Transformers"""
        try:
            # Format as chat
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ]
            
            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MODEL_PARAMS['max_new_tokens'],
                    temperature=MODEL_PARAMS['temperature'],
                    top_k=MODEL_PARAMS['top_k'],
                    top_p=MODEL_PARAMS['top_p'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode only new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"  Error in Transformers generate: {e}")
            return ""
    
    def get_name(self) -> str:
        return self.model_name


def load_model(model_key: str) -> BaseModel:
    """
    Factory function to load model by key.
    
    Args:
        model_key: One of 'gemma3', 'bielik', 'qwen'
        
    Returns:
        Initialized model wrapper
    """
    from config import MODELS
    
    config = MODELS[model_key]
    
    if config['type'] == 'ollama':
        return OllamaModel(config['name'])
    elif config['type'] == 'transformers':
        return TransformersModel(config['path'], model_key)
    else:
        raise ValueError(f"Unknown model type: {config['type']}")


def test_models():
    """Quick test of all models"""
    test_prompt = "Profile: You are a male from China with higher education, aged 30-49.\n\nQuestion: How important is God in your life? (1 = not at all important, 10 = very important)\n\nYour answer:"
    
    for model_key in ['gemma3', 'bielik', 'qwen']:
        print(f"\n{'='*50}")
        print(f"Testing {model_key}...")
        print('='*50)
        
        try:
            model = load_model(model_key)
            response = model.generate(test_prompt)
            print(f"Response: {repr(response)}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_models()
