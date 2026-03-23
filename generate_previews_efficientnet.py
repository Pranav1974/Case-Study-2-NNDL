"""
================================================================
  Case Study 2 — EfficientNet Detection Visualization (Expanded)
  Generates 6 image previews (2 per class) with labels.
================================================================
"""
import os, json, random
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(r"c:\Users\prana\OneDrive\Desktop\SUB\SEM 6\NNDL\NNDK")
OUTPUT_DIR  = BASE_DIR / "Case study 2" / "outputs" / "efficientnet"
MODEL_PATH  = OUTPUT_DIR / "models" / "tomato3class_efficientnet_final.keras"
LBL_MAP_PATH = OUTPUT_DIR / "label_mapping.json"
PREVIEW_DIR = OUTPUT_DIR / "detection_previews"
IMG_SIZE    = (224, 224)

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Model and Labels ─────────────────────────────────────
print(f"Loading trained model...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

with open(LBL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)

# ── Process Samples ───────────────────────────────────────────
DATA_DIR = BASE_DIR / "archive_1" / "PlantVillage"
CLASSES  = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight"]

for cls in CLASSES:
    cls_folder = DATA_DIR / cls
    samples = [f for f in cls_folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    
    # Pick 2 samples per class to reach 6 total
    for i in range(2):
        test_img_path = random.choice(samples)
        img_pil = Image.open(test_img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img_pil).astype("float32")
        
        proc_array = img_array.astype("float32") # EffNet internal normalization
        proc_array = np.expand_dims(proc_array, axis=0)
        
        preds = model.predict(proc_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        pred_name = label_mapping[str(class_idx)].replace("Tomato_", "")
        true_name = cls.replace("Tomato_", "")

        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0D1117")
        ax.imshow(img_array.astype("uint8"))
        ax.set_axis_off()
        
        status = "✅ CORRECT" if pred_name == true_name else "❌ MISMATCH"
        color = "#4CAF50" if pred_name == true_name else "#F44336"
        
        title_text = f"EFFICIENTNET DETECTION: {status}\nTrue: {true_name}\nPred: {pred_name} ({confidence*100:.1f}%)"
        ax.set_title(title_text, color=color, fontweight="bold", fontsize=10, pad=10)
        
        out_name = f"detection_eff_{true_name}_{i+1}.png"
        plt.savefig(PREVIEW_DIR / out_name, bbox_inches="tight", facecolor="#0D1117", dpi=120)
        plt.close()
        print(f"   Saved: {out_name}")

print(f"\n✔ Done! All images saved to: outputs/efficientnet/detection_previews/")
