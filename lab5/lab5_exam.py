import numpy as np
import matplotlib.pyplot as plt

# 1. Synthetic Data: Mixture of 2 Gaussians 
# p(x) = w1*N(μ1, σ1²) + w2*N(μ2, σ2²)
np.random.seed(42)
N = 300
MU1, SIG1, W1 = 20.0, 5.0, 0.4    
MU2, SIG2, W2 = 50.0, 8.0, 0.6    

n1 = int(N * W1)
n2 = N - n1
data = np.concatenate([
    np.random.normal(MU1, SIG1, n1),
    np.random.normal(MU2, SIG2, n2)
])
np.random.shuffle(data)

def true_pdf(x):
    g1 = (1.0 / (SIG1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - MU1) / SIG1)**2)
    g2 = (1.0 / (SIG2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - MU2) / SIG2)**2)
    return W1 * g1 + W2 * g2

# 2. k-NN Density Estimator Function 
def knn_density(x_query, data, k):
    n = len(data)
    distances = np.abs(data - x_query)   # Dist from query x to all data points
    d_k = np.sort(distances)[k - 1]      # Dist to k-th nearest neighbor
    if d_k == 0.0:
        d_k = 1e-10
    volume = 2.0 * d_k                     
    return k / (n * volume)              # p̂(x) = k / (n * V_k)

# 3. Evaluation on Grid & MSE Calculation 
x_grid = np.linspace(data.min() - 5, data.max() + 5, 500)
true_vals = true_pdf(x_grid)
K_VALUES = [1, 5, 15, 30]

print("k-NN Density Estimation")
print(f"Data: N={N}, Components: 0.4*N({MU1},{SIG1}²) + 0.6*N({MU2},{SIG2}²)\n")

# 4. Visualization: Compare k values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, k in zip(axes, K_VALUES):
    estimates = np.array([knn_density(x, data, k) for x in x_grid])
    mse = np.mean((estimates - true_vals)**2)
    print(f'k={k} | MSE={mse:.6f}')

    ax.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    ax.plot(x_grid, true_vals, 'k-', label='True PDF')
    ax.plot(x_grid, estimates, 'r--', label=f'k-NN (k={k})')
    ax.set_title(f'k={k}, MSE={mse:.5f}')
    ax.set_xlabel('x')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()