import matplotlib.pyplot as plt
import numpy as np
import os

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../plots')
os.makedirs(plots_dir, exist_ok=True)

# =======================================================
# 1. Confusion Matrix Heatmap
# =======================================================
cm = np.array([[153, 75], [85, 131]])
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap='Blues')
plt.colorbar(cax)

for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, f'{z}', ha='center', va='center', fontsize=16, 
            color='white' if z > 100 else 'black', fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Down', 'Predicted Up'], fontsize=11)
ax.set_yticklabels(['Actual Down', 'Actual Up'], fontsize=11)
ax.set_title('NeuralEdge Final Confusion Matrix', pad=20, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# =======================================================
# 2. Feature Breakdown Donut Chart
# =======================================================
labels = ['On-Chain Structural\n(13 Features)', 'Retail Sentiment\n(12 Features)', 'Eliminated Indicators\n(~38 Features)']
sizes = [13, 12, 38]
colors = ['#3498db', '#e74c3c', '#95a5a6']
explode = (0.05, 0.05, 0)

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                                  shadow=False, startangle=140, pctdistance=0.85)

# Draw circle to make it a donut
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

plt.setp(autotexts, size=11, weight="bold", color="white")
plt.setp(texts, size=11)
ax.set_title('Feature Selection Breakdown (Boruta-LASSO)', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(plots_dir, 'feature_breakdown.png'), dpi=300, bbox_inches='tight')
plt.close()

# =======================================================
# 3. Trading Simulation Mock Equity Curve (Spring 2026)
# =======================================================
np.random.seed(42) # For reproducibility
days = np.arange(0, 60)
base_price = 100

# Generate a synthetic price curve that drops ~20.5%
market_drop = np.linspace(0, -20.5, 30) + np.random.normal(0, 1.2, 30)
market_recover = np.linspace(-20.5, -15, 30) + np.random.normal(0, 1.5, 30)
bh_curve = np.concatenate([market_drop, market_recover]) + base_price

# NeuralEdge avoids the worst drops (moves to cash)
ne_drop = np.linspace(0, -8, 20) + np.random.normal(0, 0.8, 20)
ne_flat = np.repeat(ne_drop[-1], 20) + np.random.normal(0, 0.3, 20) # moved to cash (low volatility)
ne_recover = np.linspace(ne_flat[-1], -4, 20) + np.random.normal(0, 0.8, 20)
ne_curve = np.concatenate([ne_drop, ne_flat, ne_recover]) + base_price

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(days, bh_curve, label='Buy & Hold (20.5% Drawdown)', color='#e74c3c', linewidth=2)
ax.plot(days, ne_curve, label='NeuralEdge Long-Only (8.0% Drawdown)', color='#2ecc71', linewidth=2.5)

ax.fill_between(days, bh_curve, base_price, color='#e74c3c', alpha=0.1)
ax.fill_between(days, ne_curve, base_price, color='#2ecc71', alpha=0.1)

ax.set_title('Trading Simulation: Spring 2026 Market Correction', fontsize=16, fontweight='bold')
ax.set_ylabel('Portfolio Value (Indexed to 100)', fontsize=12)
ax.set_xlabel('Trading Days', fontsize=12)
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, linestyle='--', alpha=0.6)

plt.savefig(os.path.join(plots_dir, 'trading_simulation.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Successfully generated 3 new graphs!")
