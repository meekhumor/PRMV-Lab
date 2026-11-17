import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Dataset
iris = load_iris()
X = iris.data          # shape: (150, 4)  sepal_len | sepal_wid | petal_len | petal_wid
y = iris.target            
labels = iris.target_names    # ['setosa', 'versicolor', 'virginica']
feat_names = iris.feature_names

# Step 1: Standardize (zero mean) 
X_mean = np.mean(X, axis=0)
X_std  = X - X_mean        # center the data 

# Step 2: Covariance Matrix 
cov_matrix = np.cov(X_std, rowvar=False)   # 4x4 covariance matrix

# Step 3: Eigen Decomposition 
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]      

# Step 4: Variance Explained 
var_explained = eigenvalues / np.sum(eigenvalues) * 100
cumulative_var = np.cumsum(var_explained)

# Step 5: Project to 2D 
W = eigenvectors[:, :2]    # 4x2 projection matrix
X_pca = X_std @ W              # (150, 2)


print("\nCovariance Matrix (4x4):")
print(np.round(cov_matrix, 3))

print("\nEigenvalues (sorted descending):")
for i, ev in enumerate(eigenvalues):
    print(f"   PC{i+1}: {ev:.4f}  ->  Variance Explained = {var_explained[i]:.2f}%")

print(f"\nVariance retained with PC1 + PC2: {cumulative_var[1]:.2f}%")
print(f"Variance lost  with PC3 + PC4:   {100 - cumulative_var[1]:.2f}%")

print("\nPrincipal Components (Eigenvectors):")
for i in range(2):
    print(f"   PC{i+1}: {np.round(eigenvectors[:, i], 4)}")

print("\nReduced Data (first 5 rows, 4D -> 2D):")
print(np.round(X_pca[:5], 4))
print("=" * 60)

# Plotting 
BG = 'white'
PANEL = '#f7f7f7'
GRID = '#dddddd'
TEXT = '#111111'
COLORS = ['#e07b54', '#5b8dd9', '#4caf82']

def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=7)
    ax.grid(color=GRID, linestyle='--', linewidth=0.6, alpha=0.8)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(BG)

# Plot 1: Eigenvalues Bar Chart 
ax1 = axes[0]
pc_labels = [f'PC{i+1}' for i in range(4)]
bar_colors = ['#5b8dd9', '#4caf82', '#e07b54', '#9b59b6']
bars = ax1.bar(pc_labels, eigenvalues, color=bar_colors, edgecolor='white',
               linewidth=0.8, width=0.5)
for bar, ev, vr in zip(bars, eigenvalues, var_explained):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{ev:.3f}\n({vr:.1f}%)', ha='center', va='bottom',
             fontsize=8.5, color=TEXT, fontweight='bold')
style(ax1, 'Eigenvalues per Principal Component')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigenvalue (Variance)')
ax1.set_ylim(0, eigenvalues[0] * 1.25)

# Plot 2: Cumulative Variance Explained 
ax2 = axes[1]
ax2.plot(range(1, 5), cumulative_var, marker='o', color='#5b8dd9',
         lw=2.2, markersize=8, zorder=4)
ax2.fill_between(range(1, 5), cumulative_var, alpha=0.15, color='#5b8dd9')
ax2.axhline(y=cumulative_var[1], color='#e07b54', lw=1.5, ls='--',
            label=f'PC1+PC2 = {cumulative_var[1]:.1f}%')
ax2.axhline(y=95, color='#4caf82', lw=1.2, ls=':', label='95% threshold')
for i, cv in enumerate(cumulative_var):
    ax2.text(i + 1, cv + 1.2, f'{cv:.1f}%', ha='center',
             fontsize=8.5, color=TEXT, fontweight='bold')
style(ax2, 'Cumulative Variance Explained')
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Cumulative Variance (%)')
ax2.set_xticks(range(1, 5))
ax2.set_ylim(0, 110)
ax2.legend(fontsize=8.5)

# Plot 3: 2D PCA Scatter 
ax3 = axes[2]
for cls, color, lbl in zip(range(3), COLORS, labels):
    mask = y == cls
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=color, label=lbl.capitalize(),
                s=55, alpha=0.85, edgecolors='white', linewidth=0.5, zorder=4)
style(ax3, 'PCA Projection')
ax3.set_xlabel(f'PC1  ({var_explained[0]:.1f}% variance)')
ax3.set_ylabel(f'PC2  ({var_explained[1]:.1f}% variance)')
ax3.legend(fontsize=9, loc='upper right')

fig.suptitle('Lab 8 - Dimensionality Reduction using PCA on Iris Dataset',
             color=TEXT, fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()

out_path = 'lab8/pca_results.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved -> {out_path}")
