import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    cm = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 8]])
    annot_labels = np.array([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=annot_labels, fmt='', ax=ax)
    print("Num texts:", len(ax.texts))
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            text_color = "white" if val > 2 else "black"
            ax.texts[i * 3 + j].set_color(text_color)
except Exception as e:
    import traceback
    traceback.print_exc()
