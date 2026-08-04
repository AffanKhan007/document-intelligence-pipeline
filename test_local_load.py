import time

import torch

from model_loader import get_model

start = time.time()
processor, model = get_model()
elapsed = time.time() - start

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Load time: {elapsed:.2f}s")
print(f"Num labels: {model.config.num_labels}")

for i in range(5):
    label = model.config.id2label.get(i, "N/A")
    print(f"  id2label[{i}] = {label}")
