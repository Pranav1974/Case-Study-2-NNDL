import json
import os

# Define the notebook content
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Case Study 2 — Model Comparison & Side-by-Side Detection\n",
                "### EfficientNetB0 vs. DenseNet121\n",
                "This notebook perform real-time disease detection on sample leaf images using both trained models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, json, random\n",
                "import numpy as np\n",
                "import tensorflow as tf\n",
                "from pathlib import Path\n",
                "from PIL import Image\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# --- Configuration ---\n",
                "BASE = Path(r'c:/Users/prana/OneDrive/Desktop/SUB/SEM 6/NNDL/NNDK/Case study 2')\n",
                "DATA = Path(r'c:/Users/prana/OneDrive/Desktop/SUB/SEM 6/NNDL/NNDK/archive_1/PlantVillage')\n",
                "\n",
                "EFF_PATH = BASE / 'outputs' / 'efficientnet' / 'models' / 'tomato3class_efficientnet_final.keras'\n",
                "DEN_PATH = BASE / 'outputs' / 'densenet' / 'models' / 'tomato3class_densenet_final.keras'\n",
                "LBL_PATH = BASE / 'outputs' / 'densenet' / 'label_mapping.json'\n",
                "\n",
                "with open(LBL_PATH, 'r') as f:\n",
                "    label_map = json.load(f)\n",
                "\n",
                "print('Loading Models...')\n",
                "eff_model = tf.keras.models.load_model(str(EFF_PATH), compile=False)\n",
                "den_model = tf.keras.models.load_model(str(DEN_PATH), compile=False)\n",
                "print('Models Loaded Successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Visual Model Comparison (Metrics)\n",
                "Comparison chart saved during automated evaluation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "comp_img = BASE / 'outputs' / 'model_comparison.png'\n",
                "if comp_img.exists():\n",
                "    plt.figure(figsize=(10, 8))\n",
                "    img = Image.open(comp_img)\n",
                "    plt.imshow(img)\n",
                "    plt.axis('off')\n",
                "    plt.show()\n",
                "else:\n",
                "    print('Comparison chart not found.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Real-Time Leaf Detection\n",
                "Pick a random leaf and compare how both models handle them."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_prediction(model, img_arr, architecture='eff'):\n",
                "    if architecture == 'eff':\n",
                "        # EfficientNet pre-processing is built-in\n",
                "        proc = img_arr.astype('float32')\n",
                "    else:\n",
                "        # DenseNet pre-processing function\n",
                "        proc = tf.keras.applications.densenet.preprocess_input(img_arr.astype('float32'))\n",
                "    \n",
                "    preds = model.predict(np.expand_dims(proc, axis=0), verbose=0)[0]\n",
                "    idx = np.argmax(preds)\n",
                "    return label_map[str(idx)].replace('Tomato_', ''), preds[idx]\n",
                "\n",
                "classes = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight']\n",
                "fig, axes = plt.subplots(3, 1, figsize=(10, 15))\n",
                "\n",
                "for i, cls in enumerate(classes):\n",
                "    cls_dir = DATA / cls\n",
                "    test_path = random.choice([f for f in cls_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']])\n",
                "    \n",
                "    img_pil = Image.open(test_path).convert('RGB').resize((224, 224))\n",
                "    img_arr = np.array(img_pil)\n",
                "    \n",
                "    e_name, e_conf = get_prediction(eff_model, img_arr, 'eff')\n",
                "    d_name, d_conf = get_prediction(den_model, img_arr, 'den')\n",
                "    \n",
                "    axes[i].imshow(img_arr)\n",
                "    axes[i].axis('off')\n",
                "    \n",
                "    title = f'Actual: {cls[7:]}\\n' \\\n",
                "            f'EfficientNet Detection: {e_name} ({e_conf*100:.1f}%) \\n' \\\n",
                "            f'DenseNet Detection: {d_name} ({d_conf*100:.1f}%)'\n",
                "    axes[i].set_title(title, fontsize=10, fontweight='bold', pad=10)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
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
            "pygments_lexer": "ipython3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the file as JSON
with open('model_comparison_detection.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Dual-Model Comparison Notebook created successfully!')
