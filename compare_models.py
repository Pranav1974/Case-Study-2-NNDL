import os
import re
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\Users\prana\OneDrive\Desktop\SUB\SEM 6\NNDL\NNDK\Case study 2\outputs")
EFF_REPORT = BASE_DIR / "efficientnet" / "classification_report.txt"
DEN_REPORT = BASE_DIR / "densenet" / "classification_report.txt"

print(f"Waiting for DenseNet report to be generated...")

# Wait for the file to be completely written
while not DEN_REPORT.exists():
    time.sleep(10)

# Give it a couple more seconds to ensure writing finishes
time.sleep(2)

def extract_metrics(report_path):
    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Extract accuracy
    acc_match = re.search(r"Test Accuracy\s*:\s*([\d\.]+)%", content)
    acc = float(acc_match.group(1)) if acc_match else 0.0

    # Extract loss
    loss_match = re.search(r"Test Loss\s*:\s*([\d\.]+)", content)
    loss = float(loss_match.group(1)) if loss_match else 0.0

    # Extract macro avg F1
    f1_match = re.search(r"macro avg\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)", content)
    f1 = float(f1_match.group(1)) if f1_match else 0.0

    return acc, loss, f1 * 100

eff_acc, eff_loss, eff_f1 = extract_metrics(EFF_REPORT)
den_acc, den_loss, den_f1 = extract_metrics(DEN_REPORT)

print("\n========================================")
print("     EfficientNetB0 vs DenseNet121      ")
print("========================================")
print(f"         Test Accuracy | Test Loss | Macro F1 ")
print(f"EfficientNet : {eff_acc:>6.2f}% | {eff_loss:>9.4f} | {eff_f1:>6.2f}%")
print(f"DenseNet121  : {den_acc:>6.2f}% | {den_loss:>9.4f} | {den_f1:>6.2f}%")
print("========================================\n")

# Create a comparison bar chart
labels = ['Test Accuracy', 'Macro F1-Score']
eff_scores = [eff_acc, eff_f1]
den_scores = [den_acc, den_f1]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0D1117")
ax.set_facecolor("#161B22")

rects1 = ax.bar(x - width/2, eff_scores, width, label='EfficientNetB0 (5 ep)', color='#2196F3', edgecolor='#888', linewidth=1)
rects2 = ax.bar(x + width/2, den_scores, width, label='DenseNet121 (10 ep)', color='#4CAF50', edgecolor='#888', linewidth=1)

ax.set_ylabel('Percentage (%)', color='white', fontsize=12)
ax.set_title('Model Performance Comparison (3-Class Tomato Disease)', color='white', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, color='white', fontsize=11)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

ax.legend(facecolor="#0D1117", labelcolor="white", fontsize=10)
ax.set_ylim(80, 100) # Assuming both models perform > 80%

def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
fig_path = BASE_DIR / "model_comparison.png"
plt.savefig(str(fig_path), dpi=150, bbox_inches='tight', facecolor="#0D1117")
print(f"✔ Saved graphical comparison to outputs/model_comparison.png")
