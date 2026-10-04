import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# 1-D DATA: mixture of two Gaussians 
N   = 300
MU1, SIG1, W1 = 20.0, 5.0, 0.4
MU2, SIG2, W2 = 50.0, 8.0, 0.6

n1 = int(N * W1)
n2 = N - n1
data1d = np.concatenate([
    np.random.normal(MU1, SIG1, n1),
    np.random.normal(MU2, SIG2, n2)
])
np.random.shuffle(data1d)

def true_pdf_1d(x):
    """Ground-truth mixture PDF."""
    def g(x, mu, sig):
        return (1.0 / (sig * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sig) ** 2)
    return W1 * g(x, MU1, SIG1) + W2 * g(x, MU2, SIG2)

def knn_density_1d(x_query, data, k):
    """
    Estimate density at x_query using k-NN rule (1-D).
    Formula: p̂(x) = k / (n * 2 * d_k(x))
    """
    n = len(data)
    distances = np.abs(data - x_query)
    d_k = np.sort(distances)[k - 1]     # distance to kth nearest neighbour
    if d_k == 0.0:
        d_k = 1e-10
    volume = 2.0 * d_k                    
    return k / (n * volume)

# Evaluate over a fine grid for each k
x_grid = np.linspace(data1d.min() - 5, data1d.max() + 5, 500)
true_pdf_vals = np.array([true_pdf_1d(x) for x in x_grid])

K_VALUES = [1, 5, 15, 30]
knn_estimates = {}
for k in K_VALUES:
    knn_estimates[k] = np.array([knn_density_1d(x, data1d, k) for x in x_grid])

# Mean Squared Error vs true PDF
mse = {}
for k in K_VALUES:
    mse[k] = float(np.mean((knn_estimates[k] - true_pdf_vals) ** 2))

print("=" * 55)
print("Lab 5 – k-NN Density Estimation (1-D)")
print("=" * 55)
print(f"  N={N},  w1={W1} N({MU1}, {SIG1}^2)  +  w2={W2} N({MU2}, {SIG2}^2)")
for k in K_VALUES:
    print(f"  k={k:>2d}  ->  MSE = {mse[k]:.6f}")

# PLOTTING 
BG = 'white'
PANEL = '#f7f7f7'
GRID = '#dddddd'
TEXT = '#111111'
C_TRUE = '#222222'
C_HIST = '#b0c4de'
COLORS_K = ['#e07b54', '#5b8dd9', '#4caf82', '#9b59b6']

def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=7)
    ax.grid(color=GRID, linestyle='--', linewidth=0.6, alpha=0.8)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
fig.patch.set_facecolor(BG)

for ax, k, color in zip(axes, K_VALUES, COLORS_K):
    ax.hist(data1d, bins=30, density=True, color=C_HIST, alpha=0.45,
            edgecolor='white', linewidth=0.4, label='Data histogram')
    ax.plot(x_grid, true_pdf_vals, color=C_TRUE, lw=2.2,
            label='True PDF', zorder=3)
    ax.plot(x_grid, knn_estimates[k], color=color, lw=2.2, ls='--',
            label=f'k-NN  (k={k})', zorder=4)
    style(ax, f'k = {k}   |   MSE = {mse[k]:.5f}')
    ax.set_xlabel('x')
    ax.set_ylabel('Density' if k == K_VALUES[0] else '')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(x_grid[0], x_grid[-1])

fig.suptitle('Lab 5 – Non-Parametric Density Estimation using k-Nearest Neighbours',
             color=TEXT, fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
out_path = 'lab5/knn_results.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved -> {out_path}")
