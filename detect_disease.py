"""
================================================================
  Case Study 2 — Leaf Disease Inference Script
  Detects disease from a leaf image using DenseNet121
================================================================
"""
import os, json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(r"c:\Users\prana\OneDrive\Desktop\SUB\SEM 6\NNDL\NNDK")
MODEL_PATH  = BASE_DIR / "Case study 2" / "outputs" / "densenet" / "models" / "tomato3class_densenet_final.keras"
LBL_MAP_PATH = BASE_DIR / "Case study 2" / "outputs" / "densenet" / "label_mapping.json"
IMG_SIZE    = (224, 224)

# ── Load Model and Labels ─────────────────────────────────────
print(f"Loading trained model from: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

with open(LBL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)

def predict_leaf(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img).astype("float32")
    
    # DenseNet built-in preprocessing
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    
    disease_name = label_mapping[str(class_idx)].replace("Tomato_", "")
    return disease_name, confidence

# ── Demonstrate on a few test images ──────────────────────────
# Find some sample images from the dataset folders
DATA_DIR = BASE_DIR / "archive_1" / "PlantVillage"
CLASSES  = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight"]

print("\n" + "="*40)
print("     REAL-TIME LEAF DISEASE DETECTION     ")
print("="*40)

for cls in CLASSES:
    cls_folder = DATA_DIR / cls
    if cls_folder.exists():
        samples = [f for f in cls_folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        if samples:
            test_img = samples[0]
            name, conf = predict_leaf(test_img)
            print(f"Leaf Image  : {test_img.name}")
            print(f"Actual      : {cls.replace('Tomato_', '')}")
            print(f"Detected    : {name} (Confidence: {conf*100:.1f}%)")
            print("-" * 40)
