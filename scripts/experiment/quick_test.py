#!/usr/bin/env python3
"""
Quick Test - Verify experiment setup before running pilot.

Tests:
1. Config loads correctly
2. Human distributions CSV loads
3. Parser works
4. Each model can generate 1 response

Usage:
    python quick_test.py           # Test all models
    python quick_test.py gemma3    # Test only gemma3
"""
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    """Test config imports"""
    print("\n1. Testing config...")
    try:
        from config import MODELS, PATHS, MODEL_PARAMS, PILOT_PERSONAS, get_ollama_host
        print(f"   ✓ Models: {list(MODELS.keys())}")
        print(f"   ✓ Ollama host: {get_ollama_host()}")
        print(f"   ✓ Pilot personas: {PILOT_PERSONAS}")
        return True
    except Exception as e:
        print(f"   ✗ Config error: {e}")
        return False


def test_data():
    """Test data loading"""
    print("\n2. Testing data loading...")
    try:
        import pandas as pd
        from config import PATHS
        
        df = pd.read_csv(PATHS['human_distributions'])
        print(f"   ✓ Loaded {len(df)} personas")
        print(f"   ✓ Countries: {df['country'].unique().tolist()}")
        print(f"   ✓ Columns: {list(df.columns)[:5]}...")
        return True
    except Exception as e:
        print(f"   ✗ Data error: {e}")
        return False


def test_parser():
    """Test response parser"""
    print("\n3. Testing parser...")
    try:
        from parser import parse_response, test_parser
        
        # Quick tests
        assert parse_response("7") == 7
        assert parse_response("10") == 10
        assert parse_response("") is None
        assert parse_response("I cannot answer") is None
        
        print("   ✓ Parser works correctly")
        return True
    except Exception as e:
        print(f"   ✗ Parser error: {e}")
        return False


def test_model(model_key: str):
    """Test single model"""
    print(f"\n4. Testing {model_key}...")
    try:
        from models import load_model
        from parser import parse_response
        
        model = load_model(model_key)
        
        prompt = """Profile: You are a male from China with higher education, aged 30-49.

Question: How important is God in your life? (1 = not at all important, 10 = very important)

Your answer:"""
        
        response = model.generate(prompt)
        parsed = parse_response(response)
        
        print(f"   Raw response: {repr(response[:100])}")
        print(f"   Parsed value: {parsed}")
        print(f"   ✓ {model_key} works!")
        return True
    except Exception as e:
        print(f"   ✗ {model_key} error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(models_to_test: list = None):
    """Run all tests"""
    print("="*60)
    print("QUICK TEST - LLM Cultural Bias Experiment")
    print("="*60)
    
    results = {}
    
    # Basic tests
    results['config'] = test_config()
    results['data'] = test_data()
    results['parser'] = test_parser()
    
    # Model tests
    if models_to_test is None:
        models_to_test = ['gemma3', 'bielik', 'qwen']
    
    for model_key in models_to_test:
        results[model_key] = test_model(model_key)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<15} {status}")
        all_passed = all_passed and passed
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed! Ready to run pilot.")
        print("   Command: python run_pilot.py")
    else:
        print("\n❌ Some tests failed. Fix issues before running pilot.")
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        models = sys.argv[1:]
        valid = ['gemma3', 'bielik', 'qwen']
        models = [m for m in models if m in valid]
        main(models if models else None)
    else:
        main()
