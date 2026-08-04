import json
import sys

from inference import extract_fields
from model_loader import get_model

if len(sys.argv) < 2:
    print("Usage: python test_inference.py <image_path>")
    sys.exit(1)

processor, model = get_model()
result = extract_fields(processor, model, sys.argv[1])
print(json.dumps(result, indent=2))
