import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load Dataset 
iris = load_iris()
X = iris.data        # features: sepal length, width, petal length, width
y = iris.target      # classes: 0 (setosa), 1 (versicolor), 2 (virginica)
labels = iris.target_names

# 2. Step 1: Center Data 
X_mean = np.mean(X, axis=0)
X_std  = X - X_mean     

# 3. Step 2: Covariance Matrix 
cov_matrix = np.cov(X_std, rowvar=False)

# 4. Step 3: Eigendecomposition 
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues & eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 5. Step 4: Percentage Variance Explained 
var_explained = eigenvalues / np.sum(eigenvalues) * 100
cumulative_var = np.cumsum(var_explained)

print("PCA (Principal Component Analysis) on Iris Dataset")
print("\nCovariance Matrix:")
print(np.round(cov_matrix, 3))

print("\nEigenvalues & Variance Explained:")
for i, ev in enumerate(eigenvalues):
    print(f"  PC{i+1}: {ev:.4f}  -->  Variance Explained = {var_explained[i]:.2f}%")

print(f"\nVariance retained with PC1 + PC2: {cumulative_var[1]:.2f}%")
print(f"Variance lost with PC3 + PC4: {100 - cumulative_var[1]:.2f}%")

# 6. Step 5: Project Data to 2D Space
W = eigenvectors[:, :2]    
X_pca = X_std @ W          

print("\nTop 2 Principal Components (Eigenvectors):")
print("PC1:", np.round(eigenvectors[:, 0], 4))
print("PC2:", np.round(eigenvectors[:, 1], 4))

# 7. Visualization: Eigenvalues, Variance, & 2D Projection
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].bar([f'PC{i+1}' for i in range(4)], eigenvalues)
axes[0].set_title('Eigenvalues')

axes[1].plot(range(1, 5), cumulative_var, 'o-')
axes[1].axhline(cumulative_var[1], color='red', ls='--')
axes[1].axhline(95, color='green', ls=':')
axes[1].set_title('Cumulative Variance')
axes[1].set_xlabel('Components')
axes[1].set_ylabel('Variance (%)')
axes[1].grid()

for cls, col, lbl in zip(range(3), ['red', 'blue', 'green'], labels):
    axes[2].scatter(X_pca[y==cls, 0], X_pca[y==cls, 1], c=col, label=lbl, s=40)

axes[2].set_title('2D PCA Projection')
axes[2].set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
axes[2].set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()