import ollama
import json

try:
    models = ollama.list()
    print("Raw Ollama response:")
    print(json.dumps(models, indent=2))
    
    if 'models' in models:
        print("\nFirst model details:")
        first_model = models['models'][0]
        print(json.dumps(first_model, indent=2))
        
        print("\nModel names:")
        for model in models['models']:
            name = model.get('model', model.get('name', 'Unknown'))
            print(f"- {name}")
    else:
        print("No models found in response")
except Exception as e:
    print(f"Error: {e}")
