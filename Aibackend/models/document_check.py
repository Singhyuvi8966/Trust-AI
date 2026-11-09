"""
DOCUMENT CHECK MODULE - tamper detection

REAL MODE: if the file  models/document_tamper.pt  exists and PyTorch is installed
(see requirements-doc.txt), a ResNet-18 classifier (trained with ml/train.py) looks
at the ID image and says which of 5 things it sees:
    genuine / name edited / dob edited / expiry edited / id_number edited
"Tampered" = probability of "genuine" below 0.5; the most likely edited field is
returned as `flagged_field`.

MOCK MODE: no model file (or no PyTorch) -> random fake results, so the rest of
the team can still build against the API. `analysis_mode` tells you which one ran.

IMPORTANT: the model that ml/train.py produces is trained on SYNTHETIC cards only.
It will not be reliable on real IDs until you fine-tune it on real examples.

CONTRACT (matches schemas.DocumentCheckResult):
    Input:  raw image bytes (the ID document)
    Output: {"tampered": bool, "confidence": float, "flagged_field": str | None,
             "notes": str | None, "analysis_mode": "model" | "mock"}
"""

import logging
import os
import random
import threading
import time
from io import BytesIO
from pathlib import Path

logger = logging.getLogger("document-check")

MODEL_PATH = Path(os.getenv("DOC_MODEL_PATH", Path(__file__).with_name("document_tamper.pt")))
TAMPER_THRESHOLD = 0.5   # tampered when P(genuine) is below this

FIELD_NAMES = {
    "name": "full_name",
    "dob": "date_of_birth",
    "expiry": "expiry_date",
    "id_number": "id_number",
}

try:
    import torch
    import torchvision
    from PIL import Image, ImageOps
    from torchvision import transforms

    TORCH_OK = True
except Exception as _e:  # not installed, or failed to load
    TORCH_OK = False
    _IMPORT_ERROR = _e

_lock = threading.Lock()
_state = {"loaded": False, "bundle": None}


def _load_model():
    """Loads the model once. Returns None when it isn't available (-> mock mode)."""
    with _lock:
        if _state["loaded"]:
            return _state["bundle"]
        _state["loaded"] = True
        if not MODEL_PATH.exists():
            logger.warning("No document model at %s - document check running in MOCK mode.", MODEL_PATH)
            return None
        if not TORCH_OK:
            logger.warning("Found %s but PyTorch isn't available (%s) - MOCK mode. "
                           "Install requirements-doc.txt.", MODEL_PATH.name, _IMPORT_ERROR)
            return None
        try:
            try:
                ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            except TypeError:                      # very old torch
                ckpt = torch.load(MODEL_PATH, map_location="cpu")
            classes = list(ckpt["classes"])
            height, width = ckpt["img_size"]
            net = torchvision.models.resnet18(weights=None)
            net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            tf = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(ckpt["mean"], ckpt["std"]),
            ])
            _state["bundle"] = {"net": net, "classes": classes, "tf": tf}
            logger.info("Document tamper model loaded (%s).", MODEL_PATH.name)
        except Exception:
            logger.exception("Could not load the document model - MOCK mode.")
            _state["bundle"] = None
        return _state["bundle"]


def _interpret(probs: dict) -> dict:
    """Turns {class_name: probability} into the API result. Pure function (easy to test)."""
    p_genuine = probs.get("genuine", 0.0)
    if p_genuine < TAMPER_THRESHOLD:
        edited = {k: v for k, v in probs.items() if k != "genuine"}
        cls = max(edited, key=edited.get)
        field = FIELD_NAMES.get(cls, cls)
        return {
            "tampered": True,
            "confidence": round(1.0 - p_genuine, 2),
            "flagged_field": field,
            "notes": f"The model sees signs of editing around '{field}' "
                     f"(field guess {round(edited[cls] / max(1e-9, 1.0 - p_genuine) * 100)}% sure).",
            "analysis_mode": "model",
        }
    return {
        "tampered": False,
        "confidence": round(p_genuine, 2),
        "flagged_field": None,
        "notes": "No signs of editing found in the name, date of birth, expiry date or ID number.",
        "analysis_mode": "model",
    }


def _model_check(bundle, image_bytes: bytes) -> dict:
    try:
        img = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    except Exception:
        # fail safe: an unreadable document must not pass as "fine"
        return {"tampered": True, "confidence": 0.0, "flagged_field": None,
                "notes": "The document image could not be read.", "analysis_mode": "model"}
    with torch.no_grad():
        out = bundle["net"](bundle["tf"](img).unsqueeze(0)).softmax(1)[0].tolist()
    return _interpret(dict(zip(bundle["classes"], out)))


def _mock_check(image_bytes: bytes) -> dict:
    """MOCK: random results (about 20% flagged), same shape as the real thing."""
    time.sleep(0.5)
    if random.random() < 0.2:
        field = random.choice(["date_of_birth", "expiry_date", "id_number"])
        return {
            "tampered": True,
            "confidence": round(random.uniform(0.75, 0.98), 2),
            "flagged_field": field,
            "notes": f"Inconsistent font/spacing detected around '{field}' field.",
            "analysis_mode": "mock",
        }
    return {
        "tampered": False,
        "confidence": round(random.uniform(0.85, 0.99), 2),
        "flagged_field": None,
        "notes": "Document layout and fields appear consistent.",
        "analysis_mode": "mock",
    }


def model_available() -> bool:
    """True when the trained model is loaded (so results are not simulated)."""
    return _load_model() is not None


def run_document_check(image_bytes: bytes) -> dict:
    bundle = _load_model()
    if bundle is None:
        return _mock_check(image_bytes)
    return _model_check(bundle, image_bytes)
