import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "Affankhan007/receipt-extractor-layoutlmv3")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
