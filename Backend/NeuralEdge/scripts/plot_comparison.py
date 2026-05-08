import matplotlib.pyplot as plt
import numpy as np
import os

# Data
models = [
    "Logistic\nRegression", 
    "Price-only\nLSTM", 
    "Early\nModel 3", 
    "Calibrated\nModel 3", 
    "NeuralEdge\nChallenger"
]

accuracy = [50.66, 50.48, 55.34, 57.46, 63.96]
f1_score = [0.00, 67.00, 5.60, 53.98, 62.09] # Scaled to percentage for visual parity
roc_auc = [52.08, 52.11, 50.00, 65.40, 65.40] # Scaled to percentage for visual parity

x = np.arange(len(models))
width = 0.25

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, accuracy, width, label='Test Accuracy (%)', color='#2c3e50')
rects2 = ax.bar(x, f1_score, width, label='F1-Score (Scaled x100)', color='#3498db')
rects3 = ax.bar(x + width, roc_auc, width, label='ROC-AUC (Scaled x100)', color='#e74c3c')

# Labels and formatting
ax.set_ylabel('Performance Metrics', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison Across Iterations', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper left')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Grid and layout
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 80) # Set reasonable Y limit to leave space for labels
fig.tight_layout()

# Save
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots/model_comparison.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
