"""
Export the trained scikit-learn Random Forest model to ONNX format
so it can run in the browser via onnxruntime-web.

Usage:
    pip install skl2onnx onnxruntime
    python export_model_onnx.py

Output:
    web/model.onnx   – the converted model
    web/labels.json  – label mapping {index: letter}
"""

import os
import json
import pickle

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = "./model.p"
OUTPUT_DIR = "./web"
ONNX_PATH = os.path.join(OUTPUT_DIR, "model.onnx")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.json")

# ── Label mapping (must match recognize_live.py) ───────────────────
labels_dict = {0: "A", 1: "B", 2: "L", 3: "C", 4: "D", 5: "E", 6: "F"}

# ── Load the trained model ─────────────────────────────────────────
print("Loading model from", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    model_dict = pickle.load(f)
model = model_dict["model"]

# ── Determine input shape ──────────────────────────────────────────
# Each hand has 21 landmarks × 2 coordinates (x, y) = 42 features
n_features = model.n_features_in_
print(f"Model expects {n_features} input features")

# ── Convert to ONNX ────────────────────────────────────────────────
initial_type = [("float_input", FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={type(model): {"zipmap": False}},  # simpler output tensor
    target_opset=12,
)

# ── Save outputs ────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ONNX_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Saved ONNX model  -> {ONNX_PATH}")

with open(LABELS_PATH, "w") as f:
    json.dump(labels_dict, f, indent=2)
print(f"Saved labels      -> {LABELS_PATH}")

# ── Quick verification ──────────────────────────────────────────────
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_PATH)
    dummy = np.zeros((1, n_features), dtype=np.float32)
    result = sess.run(None, {"float_input": dummy})
    print(f"Verification OK   -> predicted class: {result[0][0]}")
except ImportError:
    print("  (onnxruntime not installed – skipping verification)")

print("\nDone! You can now serve the web/ folder.")
