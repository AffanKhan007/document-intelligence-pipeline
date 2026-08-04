import re
from collections import OrderedDict

import torch
from PIL import Image

_easyocr_reader = None


def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def parse_money(s):
    if s is None or s == "":
        return None
    s = s.replace("O", "0").replace("o", "0").replace("Q", "0")
    s = s.replace("l", "1").replace("I", "1")
    s = s.replace("S", "5")
    s = s.replace("B", "8")
    candidates = re.findall(r"\d{1,3}(?:[.,]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", s)
    if not candidates:
        return None
    last = candidates[-1]
    last = last.replace(",", "")
    return float(last)


def extract_fields(processor, model, image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    reader = _get_reader()
    ocr_results = reader.readtext(image_path)

    if not ocr_results:
        return {
            "items": [],
            "subtotal": None,
            "tax": None,
            "total": None,
            "overall_confidence": 0.0,
            "raw": {},
        }

    words = []
    boxes = []
    for bbox, text, _conf in ocr_results:
        words.append(text)
        quad = {
            "x1": float(bbox[0][0]),
            "y1": float(bbox[0][1]),
            "x2": float(bbox[1][0]),
            "y2": float(bbox[1][1]),
            "x3": float(bbox[2][0]),
            "y3": float(bbox[2][1]),
            "x4": float(bbox[3][0]),
            "y4": float(bbox[3][1]),
        }
        xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
        ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
        x0 = max(0.0, min(xs))
        x1 = min(float(w), max(xs))
        y0 = max(0.0, min(ys))
        y1 = min(float(h), max(ys))
        norm_box = [
            int(x0 / w * 1000),
            int(y0 / h * 1000),
            int(x1 / w * 1000),
            int(y1 / h * 1000),
        ]
        boxes.append(norm_box)

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)
        token_labels = torch.argmax(logits, dim=-1)

    seq_len = token_labels.size(1)
    token_conf = probs[0, torch.arange(seq_len), token_labels[0]]

    word_ids = encoding.word_ids(batch_index=0)

    word_label_map = {}
    word_conf_map = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_label_map:
            label_id = token_labels[0, token_idx].item()
            label = model.config.id2label[label_id]
            word_label_map[word_idx] = label
            word_conf_map[word_idx] = token_conf[token_idx].item()

    label_words = OrderedDict()
    label_confs = OrderedDict()
    for word_idx, label in word_label_map.items():
        if label == "O":
            continue
        text = words[word_idx]
        conf = word_conf_map[word_idx]
        if label not in label_words:
            label_words[label] = []
            label_confs[label] = []
        label_words[label].append(text)
        label_confs[label].append(conf)

    raw = {}
    field_confidences = {}
    for label in label_words:
        joined = " ".join(label_words[label])
        raw[label] = joined
        field_confidences[label] = sum(label_confs[label]) / len(label_confs[label])

    menu_names = label_words.get("menu.nm", [])
    menu_prices = label_words.get("menu.price", [])
    items = []
    for name, price in zip(menu_names, menu_prices):
        items.append({"name": name, "price": parse_money(price)})

    total = parse_money(raw.get("total.total_price", ""))
    subtotal = parse_money(raw.get("sub_total.subtotal_price", ""))
    tax = parse_money(raw.get("sub_total.tax_price", ""))

    if field_confidences:
        overall_confidence = sum(field_confidences.values()) / len(field_confidences)
    else:
        overall_confidence = 0.0

    return {
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "overall_confidence": overall_confidence,
        "raw": raw,
    }
