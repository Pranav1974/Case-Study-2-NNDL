"""
Fix EfficientNet Confusion Matrix (standalone — no retraining needed)
Loads the saved model, re-evaluates on the test set, saves a corrected CM.
"""
import os, json, random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"c:\Users\prana\OneDrive\Desktop\SUB\SEM 6\NNDL\NNDK")
DATA_DIR   = BASE_DIR / "archive_1" / "PlantVillage"
OUTPUT_DIR = BASE_DIR / "Case study 2" / "outputs" / "efficientnet"
MODEL_PATH = OUTPUT_DIR / "models" / "tomato3class_efficientnet_final.keras"

TARGET_CLASSES    = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight"]
SAMPLES_PER_CLASS = 1000
IMG_H = IMG_W     = 224
BATCH_SIZE        = 16
SEED              = 42
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Rebuild exact same test split ─────────────────────────────────────────────
valid_exts  = {".jpg", ".jpeg", ".png", ".bmp"}
classes     = sorted(TARGET_CLASSES)
num_classes = len(classes)
name_to_idx = {n: i for i, n in enumerate(classes)}
idx_to_name = {i: n for n, i in name_to_idx.items()}

image_paths, labels = [], []
for cls_name in classes:
    cls_dir = DATA_DIR / cls_name
    cls_files = [str(f) for f in cls_dir.iterdir() if f.suffix.lower() in valid_exts]
    if len(cls_files) > SAMPLES_PER_CLASS:
        cls_files = random.sample(cls_files, SAMPLES_PER_CLASS)
    image_paths.extend(cls_files)
    labels.extend([name_to_idx[cls_name]] * len(cls_files))

image_paths = np.array(image_paths)
labels      = np.array(labels, dtype=np.int32)

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=(VAL_SPLIT + TEST_SPLIT),
    random_state=SEED, stratify=labels)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    random_state=SEED, stratify=y_temp)

# ── Build test dataset ────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
def parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    return tf.cast(img, tf.float32), label

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
           .map(parse_image, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print("Model loaded. Running predictions ...")

y_pred_probs = model.predict(test_ds, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred,
      target_names=[idx_to_name[i] for i in range(num_classes)]))

# ── Confusion Matrix (FIXED) ──────────────────────────────────────────────────
cm        = confusion_matrix(y_test, y_pred)
short_names = [n.replace("Tomato_", "") for n in [idx_to_name[i] for i in range(num_classes)]]

cm_sum  = cm.sum(axis=1, keepdims=True)
cm_pct  = cm.astype(float) / cm_sum.clip(min=1) * 100
annot_labels = np.array(
    [[f"{cnt}\n({pct:.1f}%)" for cnt, pct in zip(row_c, row_p)]
     for row_c, row_p in zip(cm, cm_pct)]
)

fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0D1117")
ax.set_facecolor("#161B22")
hm = sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=short_names,
    yticklabels=short_names,
    ax=ax,
    linewidths=1.0,
    linecolor="#333",
    vmin=0
)

vmax = cm.max() if cm.max() > 0 else 1
threshold = vmax * 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        text_color = "white" if val > threshold else "#0D1117"
        ax.text(j + 0.5, i + 0.5, annot_labels[i, j],
                ha="center", va="center", color=text_color,
                fontsize=11, fontweight="bold")

ax.set_title("Confusion Matrix — EfficientNetB0 Test Set (3-Class Tomato)",
             color="white", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", color="white", fontsize=11)
ax.set_ylabel("True Label",      color="white", fontsize=11)
ax.tick_params(colors="white", labelsize=9)
plt.xticks(rotation=30, ha="right", color="white")
plt.yticks(rotation=0, color="white")
ax.collections[0].colorbar.ax.yaxis.set_tick_params(color="white")
plt.setp(ax.collections[0].colorbar.ax.yaxis.get_ticklabels(), color="white")
plt.tight_layout()

out_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print(f"\n✔ Saved fixed confusion matrix → {out_path}")
