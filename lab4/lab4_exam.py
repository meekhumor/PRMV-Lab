import math
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Generation 
np.random.seed(42)
TRUE_MU = 50.0   # Ground truth mean μ
TRUE_SIGMA = 15.0   # Ground truth standard deviation σ
N = 200    

data = [np.random.normal(TRUE_MU, TRUE_SIGMA) for _ in range(N)]

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 2. CASE 1: Unknown μ, Known σ 
known_sigma = TRUE_SIGMA
mu_hat1 = sum(data) / N  # μ̂ = (1/N) * Σ x_k

# Log-likelihood: l(μ) = Σ [-0.5*ln(2πσ²) - (x-μ)² / (2σ²)]
def ll_case1(mu):
    return sum(-0.5 * math.log(2 * math.pi * known_sigma**2) - (x - mu)**2 / (2 * known_sigma**2) for x in data)

print("CASE 1: Unknown μ, Known σ")
print(f"True μ = {TRUE_MU:.2f}, Known σ = {known_sigma:.2f}")
print(f"MLE Estimate μ̂ = {mu_hat1:.4f}")
print(f"Log-Likelihood l(μ̂) = {ll_case1(mu_hat1):.4f}\n")

# 3. CASE 2: Unknown μ AND σ² 
# μ̂ = (1/N) * Σ x_k
mu_hat2 = sum(data) / N  # μ̂ = (1/N) * Σ x_k
var_hat2   = sum((x - mu_hat2)**2 for x in data) / N  # σ̂² = (1/N) * Σ (x - μ̂)²
sigma_hat2 = math.sqrt(var_hat2)

# Log-likelihood: l(μ, σ) = Σ [-0.5*ln(2πσ²) - (x-μ)² / (2σ²)]
def ll_case2(mu, sig):
    return sum(-0.5 * math.log(2 * math.pi * sig**2) - (x - mu)**2 / (2 * sig**2) for x in data)

print("CASE 2: Unknown μ and σ²")
print(f"True μ = {TRUE_MU:.2f}, True σ = {TRUE_SIGMA:.2f}")
print(f"MLE Estimate μ̂ = {mu_hat2:.4f}")
print(f"MLE Estimate σ̂ = {sigma_hat2:.4f}")
print(f"Log-Likelihood l(μ̂, σ̂) = {ll_case2(mu_hat2, sigma_hat2):.4f}")

# 4. Visualization: True PDF vs MLE Estimated PDF
xs = np.linspace(TRUE_MU-4*TRUE_SIGMA, TRUE_MU+4*TRUE_SIGMA, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for ax, true_s, mle_m, mle_s, title in [
    (ax1, known_sigma, mu_hat1, known_sigma, 'Unknown μ, Known σ'),
    (ax2, TRUE_SIGMA, mu_hat2, sigma_hat2, 'Unknown μ and σ²')
]:
    ax.plot(xs, gaussian_pdf(xs, TRUE_MU, true_s), 'k-', label='True PDF')
    ax.plot(xs, gaussian_pdf(xs, mle_m, mle_s), 'r--', label='MLE PDF')
    ax.axvline(TRUE_MU, color='black', ls=':')
    ax.axvline(mle_m, color='red', ls=':')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('p(x | θ)')
    ax.legend()
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.show()