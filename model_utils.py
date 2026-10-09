"""Helper functions for AI Image Forensics Studio."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "cifake_mobilenetv2.keras"
MODEL_PATH = BASE_DIR / MODEL_FILENAME
IMG_SIZE = 224

# CIFAKE folders are normally sorted as FAKE=0, REAL=1.
# The UI displays FAKE as AI-generated.
CLASS_INDICES = {"AI": 0, "REAL": 1}
INDEX_TO_CLASS = {0: "AI", 1: "REAL"}

# Replace this with the exact held-out test accuracy from model.evaluate().
# It is deliberately not guessed here.
RECORDED_TEST_ACCURACY = 0.0


def format_file_size(size_bytes: int) -> str:
    """Format a file size for display."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size_bytes} B"


def validate_image(uploaded_file: Any) -> tuple[bool, str]:
    """Validate uploaded file type, size, and image readability."""
    if uploaded_file is None:
        return False, "Please upload an image."

    allowed_types = {"image/jpeg", "image/png"}
    file_type = getattr(uploaded_file, "type", "")
    file_name = getattr(uploaded_file, "name", "")
    if file_type and file_type not in allowed_types:
        return False, "Unsupported file type. Please upload a JPG, JPEG, or PNG image."
    if file_name and Path(file_name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return False, "Unsupported file extension. Please upload a JPG, JPEG, or PNG image."

    try:
        uploaded_file.seek(0)
        raw = uploaded_file.getvalue()
        if not raw:
            return False, "The uploaded file is empty."
        if len(raw) > 20 * 1024 * 1024:
            return False, "Image is too large. Please upload an image smaller than 20 MB."
        with Image.open(uploaded_file) as img:
            img.verify()
        uploaded_file.seek(0)
        return True, "Image is valid."
    except (UnidentifiedImageError, OSError, ValueError):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return False, "This file could not be read as an image."


def load_image_rgb(uploaded_file: Any) -> Image.Image:
    """Read the uploaded image and return an RGB PIL image."""
    uploaded_file.seek(0)
    with Image.open(uploaded_file) as image:
        rgb_image = image.convert("RGB")
    uploaded_file.seek(0)
    return rgb_image


def get_image_meta(uploaded_file: Any, image: Image.Image) -> dict[str, Any]:
    """Return metadata used by the Streamlit interface."""
    uploaded_file.seek(0)
    size_bytes = len(uploaded_file.getvalue())
    uploaded_file.seek(0)
    return {
        "filename": getattr(uploaded_file, "name", "uploaded_image"),
        "width": int(image.width),
        "height": int(image.height),
        "size_bytes": int(size_bytes),
        "format": image.format or "Unknown",
        "mode": image.mode,
    }


@st.cache_resource(show_spinner="Loading trained image model...")
def load_model():
    """Load the saved Keras model once and cache it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place cifake_mobilenetv2.keras in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def model_status() -> tuple[bool, str]:
    """Report whether the saved model file is available."""
    if not MODEL_PATH.is_file():
        return False, f"Model file missing: {MODEL_FILENAME}"
    try:
        return True, f"Model ready · {MODEL_FILENAME}"
    except Exception as exc:
        return False, f"Model unavailable: {exc}"


def predict(model, image: Image.Image) -> dict[str, Any]:
    """Run binary classification. Sigmoid output is assumed to mean REAL (index 1)."""
    rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(rgb, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    raw_output = model.predict(arr, verbose=0)
    p_real = float(np.asarray(raw_output).reshape(-1)[0])
    p_real = max(0.0, min(1.0, p_real))
    p_ai = 1.0 - p_real

    predicted_index = 1 if p_real >= 0.5 else 0
    predicted_class = INDEX_TO_CLASS[predicted_index]
    confidence = p_real if predicted_index == 1 else p_ai

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence_pct": round(confidence * 100.0, 2),
        "raw_sigmoid": p_real,
        "prob_real": p_real,
        "prob_ai": p_ai,
        "status": "OK",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
