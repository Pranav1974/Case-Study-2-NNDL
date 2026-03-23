"""
Case Study 2 — EfficientNetB0 Inference & Visualization
Loads trained model, runs predictions, saves all output images.
"""
import json, random
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image as PIL_Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"c:\Users\prana\OneDrive\Desktop\SUB\SEM 6\NNDL\NNDK")
DATA_DIR   = BASE_DIR / "archive_1" / "PlantVillage"
CS2_DIR    = BASE_DIR / "Case study 2"
OUTPUT_DIR = CS2_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "models" / "tomato3class_efficientnet_final.keras"
VIZ_DIR    = OUTPUT_DIR / "predictions"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

IMG_H = 224; IMG_W = 224; SEED = 42; BATCH = 16
SAMPLES_PER_CLASS = 1000
VIZ_PER_CLS = 3
TARGET_CLASSES = ["Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight"]
BAR_COLORS = ["#2196F3","#FF9800","#4CAF50"]

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Load model ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Loading trained EfficientNetB0 model ...")
print("="*55)

# Custom focal loss needed for loading
class WeightedFocalLoss(tf.keras.losses.Loss):
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
        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t))
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"class_weights": self.class_weights_list, "gamma": self.gamma})
        return cfg

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    custom_objects={"WeightedFocalLoss": WeightedFocalLoss}
)
print(f"✔ Model loaded: {MODEL_PATH.name}")

# ── Class maps ────────────────────────────────────────────────────────────────
classes     = sorted(TARGET_CLASSES)
num_classes = len(classes)
name_to_idx = {n: i for i, n in enumerate(classes)}
idx_to_name = {i: n for n, i in name_to_idx.items()}

# ── Rebuild same balanced test split (same seed) ──────────────────────────────
print("\n  Rebuilding balanced test split (1000/class, seed=42) ...")
valid_exts = {".jpg",".jpeg",".png",".bmp"}
image_paths, labels = [], []
for cls_name in classes:
    cls_dir   = DATA_DIR / cls_name
    cls_files = [str(f) for f in cls_dir.iterdir() if f.suffix.lower() in valid_exts]
    if len(cls_files) > SAMPLES_PER_CLASS:
        cls_files = random.sample(cls_files, SAMPLES_PER_CLASS)
    image_paths.extend(cls_files)
    labels.extend([name_to_idx[cls_name]] * len(cls_files))

image_paths = np.array(image_paths)
labels      = np.array(labels, dtype=np.int32)

_, X_temp, _, y_temp = train_test_split(
    image_paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
X_test, y_test, _, _ = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
print(f"  Test set: {len(y_test)} images ({len(y_test)//num_classes} per class)")

# ── Build test dataset ────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    return tf.cast(img, tf.float32), label

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
           .map(load_img, num_parallel_calls=AUTOTUNE)
           .batch(BATCH).prefetch(AUTOTUNE))

# ── Inference ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Running inference on test set ...")
print("="*55)
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n  ✔ Test Accuracy : {test_acc*100:.2f}%")
print(f"  ✔ Test Loss     : {test_loss:.4f}")

y_pred_probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n  📋 Classification Report:\n")
report = classification_report(
    y_test, y_pred,
    target_names=[idx_to_name[i] for i in range(num_classes)]
)
print(report)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
short = [idx_to_name[i].replace("Tomato_","") for i in range(num_classes)]

fig, ax = plt.subplots(figsize=(8,6), facecolor="#0D1117")
ax.set_facecolor("#0D1117")
cmap_dark = sns.dark_palette("#2196F3", as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_dark,
            xticklabels=short, yticklabels=short, ax=ax,
            linewidths=0.5, linecolor="#333", annot_kws={"size":14})
ax.set_title("Confusion Matrix — Test Set (3-Class)",
             color="white", fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicted", color="white", fontsize=11)
ax.set_ylabel("Actual",    color="white", fontsize=11)
ax.tick_params(colors="white", labelsize=9)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "confusion_matrix_inference.png"),
            dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print("  ✔ Saved: confusion_matrix_inference.png")

# ── Per-sample prediction cards ───────────────────────────────────────────────
print("\n" + "="*55)
print(f"  Generating {VIZ_PER_CLS} prediction cards per class ...")
print("="*55)

class_to_indices = defaultdict(list)
for i, lbl in enumerate(y_test):
    class_to_indices[lbl].append(i)

summary_rows = []
for cls_idx in range(num_classes):
    cls_name  = idx_to_name[cls_idx]
    cls_short = cls_name.replace("Tomato_","")
    samples   = random.sample(class_to_indices[cls_idx],
                              min(VIZ_PER_CLS, len(class_to_indices[cls_idx])))

    for rank, idx in enumerate(samples):
        img_path   = X_test[idx]
        true_lbl   = int(y_test[idx])
        pred_lbl   = int(y_pred[idx])
        probs      = y_pred_probs[idx]
        confidence = float(probs[pred_lbl])
        correct    = (true_lbl == pred_lbl)

        img_arr = np.array(PIL_Image.open(img_path).convert("RGB").resize((IMG_H, IMG_W)))
        status  = "✅ CORRECT" if correct else "❌ WRONG"
        s_color = "#4CAF50" if correct else "#F44336"

        fig, (ax_img, ax_bar) = plt.subplots(
            1, 2, figsize=(11, 4),
            gridspec_kw={"width_ratios": [1, 1.4]},
            facecolor="#0D1117"
        )
        ax_img.imshow(img_arr); ax_img.axis("off")
        ax_img.set_title(
            f"{status}\nTrue : {cls_name.replace('Tomato_','')}\n"
            f"Pred : {idx_to_name[pred_lbl].replace('Tomato_','')}",
            color=s_color, fontsize=9, fontweight="bold", pad=6)

        ax_bar.set_facecolor("#161B22")
        snames = [idx_to_name[i].replace("Tomato_","") for i in range(num_classes)]
        bars = ax_bar.barh(snames, probs*100, color=BAR_COLORS, edgecolor="#333", height=0.5)
        for bar, p in zip(bars, probs):
            ax_bar.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                        f"{p*100:.1f}%", va="center", color="white",
                        fontsize=10, fontweight="bold")
        ax_bar.set_xlim(0, 118)
        ax_bar.axvline(x=50, color="#555", linewidth=0.8, linestyle="--")
        ax_bar.set_xlabel("Confidence (%)", color="white", fontsize=9)
        ax_bar.tick_params(colors="white", labelsize=9)
        ax_bar.set_title("EfficientNetB0 — Softmax Output", color="white", fontsize=9, pad=6)
        for spine in ax_bar.spines.values(): spine.set_edgecolor("#444")

        fig.suptitle(
            f"EfficientNet Prediction | {cls_name} | Conf: {confidence*100:.1f}%",
            fontsize=11, color=s_color, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_name = f"{cls_short}_sample{rank+1}.png"
        plt.savefig(str(VIZ_DIR / out_name), dpi=130,
                    bbox_inches="tight", facecolor="#0D1117")
        plt.close()

        tag = "✅" if correct else "❌"
        print(f"   {tag} [{cls_short:<30}] conf={confidence*100:.1f}%  → {out_name}")
        summary_rows.append((img_arr, cls_name, idx_to_name[pred_lbl], probs, correct))

# ── Summary grid ──────────────────────────────────────────────────────────────
n = len(summary_rows)
grid_rows = (n + 1) // 2
fig, axes = plt.subplots(grid_rows, 4,
                         figsize=(16, grid_rows * 3.5),
                         facecolor="#0D1117")
if grid_rows == 1: axes = axes[np.newaxis, :]
fig.suptitle("EfficientNetB0 — All Prediction Cards Summary",
             fontsize=14, color="white", fontweight="bold", y=1.01)

for i, (img_arr, true_n, pred_n, probs, ok) in enumerate(summary_rows):
    row = i // 2; col_img = (i % 2)*2; col_bar = col_img+1
    sc  = "#4CAF50" if ok else "#F44336"

    ax_i = axes[row, col_img]
    ax_i.imshow(img_arr); ax_i.axis("off")
    ax_i.set_title(f"{'✅' if ok else '❌'} {true_n.replace('Tomato_','')}", color=sc, fontsize=7, pad=3)

    ax_b = axes[row, col_bar]
    ax_b.set_facecolor("#161B22")
    snames = [idx_to_name[j].replace("Tomato_","") for j in range(num_classes)]
    bars = ax_b.barh(snames, probs*100, color=BAR_COLORS, edgecolor="#333", height=0.45)
    for bar, p in zip(bars, probs):
        ax_b.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                  f"{p*100:.0f}%", va="center", color="white", fontsize=6)
    ax_b.set_xlim(0, 120); ax_b.tick_params(colors="white", labelsize=6)
    ax_b.set_title(f"Pred: {pred_n.replace('Tomato_','')}", color="#FFB74D", fontsize=7, pad=3)
    for spine in ax_b.spines.values(): spine.set_edgecolor("#444")

for j in range(n, grid_rows*2):
    r, c = divmod(j, 2)
    if r < grid_rows: axes[r, c*2].axis("off"); axes[r, c*2+1].axis("off")

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "prediction_summary_grid.png"),
            dpi=130, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print("\n  ✔ Saved: prediction_summary_grid.png")

# ── Final ─────────────────────────────────────────────────────────────────────
correct_count = sum(1 for *_, ok in summary_rows if ok)
print("\n" + "="*55)
print("  🎉  INFERENCE COMPLETE")
print("="*55)
print(f"  Test Accuracy   : {test_acc*100:.2f}%")
print(f"  Test Loss       : {test_loss:.4f}")
print(f"  Cards Correct   : {correct_count}/{len(summary_rows)}")
print(f"\n  📁 outputs/predictions/  ← {n} prediction cards")
print(f"  📁 outputs/prediction_summary_grid.png")
print(f"  📁 outputs/confusion_matrix_inference.png")
print("="*55)
