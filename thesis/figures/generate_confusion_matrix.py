import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

cm = np.array([[55, 40], [4, 38]])
labels = ['AI', 'Non-AI']

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size': 16}, ax=ax)

ax.set_xlabel('Predicted label', fontsize=12)
ax.set_ylabel('True label', fontsize=12)
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Done.")
