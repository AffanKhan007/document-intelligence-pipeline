import torch
from transformers import AutoProcessor, AutoModelForTokenClassification

from config import MODEL_ID

_processor = None
_model = None


def get_model():
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model

    _processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)

    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            _model = AutoModelForTokenClassification.from_pretrained(
                MODEL_ID,
                load_in_8bit=True,
                device_map="auto",
            )
        except Exception:
            _model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    else:
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

    _model.eval()
    return _processor, _model
