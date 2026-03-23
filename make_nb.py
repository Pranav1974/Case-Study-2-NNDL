import json
import re

with open('train_tomato_densenet.py', 'r', encoding='utf-8') as f:
    code = f.read()

blocks = re.split(r'# ══════════════════════════════════════════════════════════════════════════════+\n?', code)

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(text):
    if not text.strip(): return
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in text.strip().split('\n')]
    })

add_md('# Case Study 2 — Tomato Disease Detection (DenseNet121)\nTransfer Learning + Fine-Tuning with `DenseNet121`')
add_code(blocks[0])

for block in blocks[1:]:
    if 'STEP 1' in block:
        add_md('## 1. Dataset Loading & Class Balancing')
    elif 'STEP 2' in block:
        add_md('## 2. Data Augmentation & `tf.data` Pipelines')
    elif 'STEP 3' in block:
        add_md('## 3. Build DenseNet121 Model')
    elif 'STEP 4' in block:
        add_md('## 4. Phase 1: Train Classifier Head (Frozen Base)')
    elif 'STEP 5' in block:
        add_md('## 5. Phase 2: Fine-tune Last Dense Block')
    elif 'STEP 6' in block:
        add_md('## 6. Evaluate on Test Set & Plot Curves\nIncludes the generation of the manually-annotated Confusion Matrix to avoid rendering bugs.')
    elif 'STEP 7' in block:
        add_md('## 7. Model Predictions & Bar Charts Visualization')
    elif 'FINAL SUMMARY' in block:
        add_md('## 8. Final Report')

    add_code(block)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('train_tomato_densenet.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook created successfully!")
