"""
================================================================
  Case Study 2 — Tomato Disease Detection (3-Class)
  DenseNet121 Transfer Learning + Fine-Tuning
================================================================
  Target Classes:
    • Tomato_Bacterial_spot
    • Tomato_Early_blight
    • Tomato_Late_blight

  Dataset  : PlantVillage (archive_1/PlantVillage)
  Split    : 70% Train | 15% Validation | 15% Test
  Model    : DenseNet121  (ImageNet weights)

  Outputs (saved to Case study 2/outputs/densenet/):
    • training_curves.png
    • confusion_matrix.png
    • classification_report.txt
    • models/            ← best checkpoints + final model
    • label_mapping.json
================================================================
"""

# ── stdlib / third-party ────────────────────────────────────────
import os, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHTED FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════════
class WeightedFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss with per-class alpha weights.
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, class_weights, gamma=2.0, name="weighted_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights_list = class_weights
        self.gamma = gamma
        self._cw = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        batch   = tf.shape(y_true)[0]
        idx     = tf.stack([tf.range(batch), y_true], axis=1)
        p_t     = tf.gather_nd(y_pred, idx)

        alpha_t = tf.gather(self._cw, y_true)
        focal   = -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"class_weights": self.class_weights_list, "gamma": self.gamma})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR    = Path(r"c:\Users\prana\OneDrive\Desktop\SUB\SEM 6\NNDL\NNDK")
DATA_DIR    = BASE_DIR / "archive_1" / "PlantVillage"
CS2_DIR     = BASE_DIR / "Case study 2"
OUTPUT_DIR  = CS2_DIR / "outputs" / "densenet"
MODEL_DIR   = OUTPUT_DIR / "models"
LOG_DIR     = OUTPUT_DIR / "logs"

# ── Only these 3 classes ──────────────────────────────────────────────────────
TARGET_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
]

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_H            = 224
IMG_W            = 224
BATCH_SIZE       = 16
EPOCHS_P1        = 10       # Phase-1 — head only (frozen base)
EPOCHS_P2        = 10       # Phase-2 — fine-tune last block of layers
LR_P1            = 1e-3
LR_P2            = 5e-5
SEED             = 42
TRAIN_SPLIT      = 0.70
VAL_SPLIT        = 0.15
TEST_SPLIT       = 0.15
SAMPLES_PER_CLASS = 1000    # balance: randomly cap each class at this many images

# ── Create output folders ─────────────────────────────────────────────────────
for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✔ GPU : {gpus[0].name}")
else:
    print("⚠ No GPU — training on CPU")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATASET (3 classes only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 1 — Loading 3-Class Tomato Dataset")
print("="*60)

valid_exts  = {".jpg", ".jpeg", ".png", ".bmp"}
classes     = sorted(TARGET_CLASSES)
num_classes = len(classes)
name_to_idx = {name: idx for idx, name in enumerate(classes)}
idx_to_name = {idx: name for name, idx in name_to_idx.items()}

print(f"\n  Target classes ({num_classes}):")
for idx, name in idx_to_name.items():
    print(f"   [{idx}]  {name}")

image_paths, labels = [], []
for cls_name in classes:
    cls_dir = DATA_DIR / cls_name
    if not cls_dir.is_dir():
        raise FileNotFoundError(f"Class folder not found: {cls_dir}")
    lbl = name_to_idx[cls_name]
    cls_files = [str(f) for f in cls_dir.iterdir() if f.suffix.lower() in valid_exts]
    if len(cls_files) > SAMPLES_PER_CLASS:
        cls_files = random.sample(cls_files, SAMPLES_PER_CLASS)
    image_paths.extend(cls_files)
    labels.extend([lbl] * len(cls_files))
    status = "(capped)" if len(cls_files) == SAMPLES_PER_CLASS else "(all)"
    print(f"   {cls_name:<35} → {len(cls_files):>4} images  {status}")

image_paths = np.array(image_paths)
labels      = np.array(labels, dtype=np.int32)
total       = len(labels)
print(f"\n  Total images : {total}")

with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
    json.dump({int(k): v for k, v in idx_to_name.items()}, f, indent=2)

# ── 70 / 15 / 15 stratified split ────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels,
    test_size=(VAL_SPLIT + TEST_SPLIT),
    random_state=SEED, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    random_state=SEED, stratify=y_temp
)

print(f"\n  Split (70 / 15 / 15):")
print(f"    Train  : {len(y_train):>6}  images")
print(f"    Val    : {len(y_val):>6}  images")
print(f"    Test   : {len(y_test):>6}  images")

cw_arr  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
focal_loss = WeightedFocalLoss(class_weights=cw_arr.tolist(), gamma=2.0)
print(f"\n  Weighted Focal Loss ready  (gamma=2.0, alpha={[round(w,3) for w in cw_arr.tolist()]})")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — tf.data PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 2 — Building tf.data pipelines")
print("="*60)

AUTOTUNE = tf.data.AUTOTUNE
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.20),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="augmentation")

def parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    img = tf.cast(img, tf.float32)
    return img, label

def augment(img, label):
    img = augmentation(img, training=True)
    return img, label

def make_dataset(paths, lbls, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, lbls))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(1000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(X_train, y_train, training=True)
val_ds   = make_dataset(X_val,   y_val)
test_ds  = make_dataset(X_test,  y_test)
print("  ✔ Pipelines ready")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — BUILD DenseNet121
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 3 — Building DenseNet121 (3-class head)")
print("="*60)

def build_model(num_classes, trainable_base=False):
    base = DenseNet121(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = trainable_base

    inputs = layers.Input(shape=(IMG_H, IMG_W, 3))
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    return Model(inputs, out, name="TomatoDisease_DenseNet121"), base

model, base_model = build_model(num_classes, trainable_base=False)
model.summary()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — PHASE 1: Train classifier head (frozen base)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 4 — Phase 1: Training classifier head (frozen base)")
print(f"           Epochs: {EPOCHS_P1}")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(LR_P1),
    loss=focal_loss,
    metrics=["accuracy"]
)

cb_p1 = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(str(MODEL_DIR / "best_phase1.keras"), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    CSVLogger(str(LOG_DIR / "phase1_log.csv"))
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P1,
    callbacks=cb_p1
)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — PHASE 2: Fine-tune last Dense Block
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 5 — Phase 2: Fine-tuning last Dense Block")
print(f"           Epochs: {EPOCHS_P2}")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

n_trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"  Trainable base layers: {n_trainable} / {len(base_model.layers)}")

model.compile(
    optimizer=keras.optimizers.Adam(LR_P2),
    loss=focal_loss,
    metrics=["accuracy"]
)

cb_p2 = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(str(MODEL_DIR / "best_phase2.keras"), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    CSVLogger(str(LOG_DIR / "phase2_log.csv"))
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P2,
    callbacks=cb_p2
)

model.save(str(MODEL_DIR / "tomato3class_densenet_final.keras"))
print(f"\n  ✔ Final model saved → {MODEL_DIR / 'tomato3class_densenet_final.keras'}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — EVALUATE ON TEST SET & CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  STEP 6 — Evaluating on Test Set (15%)")
print("="*60)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

y_pred_probs = model.predict(test_ds, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\n  📋 Classification Report:\n")
report = classification_report(
    y_test, y_pred,
    target_names=[idx_to_name[i] for i in range(num_classes)]
)
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(f"Case Study 2 — Tomato Disease Detection (3-Class)\n")
    f.write("="*50 + "\n")
    f.write(f"Architecture: DenseNet121\n")
    f.write(f"Classes: {', '.join(classes)}\n\n")
    f.write(f"Test Accuracy : {test_acc*100:.2f}%\n")
    f.write(f"Test Loss     : {test_loss:.4f}\n\n")
    f.write(f"Epochs Phase-1: {EPOCHS_P1}\n")
    f.write(f"Epochs Phase-2: {EPOCHS_P2}\n\n")
    f.write(report)

# ── Training Curves ───────────────────────────────────────────────────────────
def merge_histories(h1, h2):
    merged = {}
    for k in h1.history:
        merged[k] = h1.history[k] + h2.history[k]
    return merged

hist        = merge_histories(history1, history2)
ep_range    = range(1, len(hist["accuracy"]) + 1)
phase1_end  = len(history1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0D1117")
fig.suptitle("DenseNet121 — Tomato Disease (3-Class) Training History",
             fontsize=14, color="white", fontweight="bold")

for ax in axes:
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.axvline(x=phase1_end + 0.5, color="#FFD700", linewidth=1.5,
               linestyle="--", label="Phase 1→2")

axes[0].plot(ep_range, hist["accuracy"],     label="Train", color="#2196F3", lw=2)
axes[0].plot(ep_range, hist["val_accuracy"], label="Val",   color="#FF9800", lw=2)
axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy"); axes[0].legend(facecolor="#0D1117", labelcolor="white")
axes[0].grid(True, alpha=0.2)

axes[1].plot(ep_range, hist["loss"],     label="Train", color="#4CAF50", lw=2)
axes[1].plot(ep_range, hist["val_loss"], label="Val",   color="#F44336", lw=2)
axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss"); axes[1].legend(facecolor="#0D1117", labelcolor="white")
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "training_curves.png"), dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print("  ✔ Saved: training_curves.png")

# ── Confusion Matrix (FIXED — manual annotations bypass seaborn bugs) ─────────
cm = confusion_matrix(y_test, y_pred)
short_names = [n.replace("Tomato_", "") for n in [idx_to_name[i] for i in range(num_classes)]]

cm_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
cm_pct = cm.astype(float) / cm_sum * 100

annot_labels = np.array(
    [[f"{int(cnt)}\n({pct:.1f}%)" for cnt, pct in zip(row_c, row_p)]
     for row_c, row_p in zip(cm, cm_pct)]
)

fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0D1117")
ax.set_facecolor("#161B22")

vmax = cm.max() if cm.max() > 0 else 1

hm = sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=short_names,
    yticklabels=short_names,
    ax=ax,
    linewidths=1.0,
    linecolor="#333",
    vmin=0,
    vmax=vmax
)

threshold = vmax * 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        text_color = "white" if val > threshold else "#0D1117"
        ax.text(j + 0.5, i + 0.5, annot_labels[i, j],
                ha="center", va="center", color=text_color,
                fontsize=12, fontweight="bold")

ax.set_title("Confusion Matrix — DenseNet121 Test Set (3-Class Tomato)\n"
             f"(Phase-1: {EPOCHS_P1} epochs | Phase-2: {EPOCHS_P2} epochs)",
             color="white", fontsize=12, fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", color="white", fontsize=11)
ax.set_ylabel("True Label",      color="white", fontsize=11)
ax.tick_params(colors="white", labelsize=9)
plt.xticks(rotation=30, ha="right", color="white")
plt.yticks(rotation=0, color="white")
ax.collections[0].colorbar.ax.yaxis.set_tick_params(color="white")
plt.setp(ax.collections[0].colorbar.ax.yaxis.get_ticklabels(), color="white")
plt.tight_layout()

plt.savefig(str(OUTPUT_DIR / "confusion_matrix.png"), dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print("  ✔ Saved: confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  🎉  DONE — Outputs saved to: Case study 2/outputs/densenet/")
print("="*60)
print(f"  Test Accuracy    : {test_acc*100:.2f}%")
print(f"  Test Loss        : {test_loss:.4f}")
print("="*60)
