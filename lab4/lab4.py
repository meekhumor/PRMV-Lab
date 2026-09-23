import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
TRUE_MU = 50.0
TRUE_SIGMA = 15.0
N = 200
data = [np.random.normal(TRUE_MU, TRUE_SIGMA) for _ in range(N)]

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) \
           * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


# CASE 1: Unknown μ, Σ KNOWN 
known_sigma = TRUE_SIGMA   

n = len(data)
mu_hat = 0.0
for x in data:
    mu_hat += x
mu_hat /= n

# Log-likelihood l(μ)  [Eq. 8]
def ll_case1(mu_cand):
    ll = 0.0
    for x in data:
        ll += -0.5 * math.log(2 * math.pi * known_sigma**2) \
              - (x - mu_cand)**2 / (2 * known_sigma**2)
    return ll

# Derivative ∂l/∂μ  [Eq. 10]  → 0 at μ̂
def dl_dmu(mu_cand):
    return sum((x - mu_cand) for x in data) / known_sigma**2

print("CASE 1: Unknown μ, Σ known")
print(f"True  μ = {TRUE_MU}")
print(f"True σ (known) = {known_sigma}")
print(f"MLE μ̂  [Eq.16]= {mu_hat:.4f}")
print(f"l(μ̂) [Eq.5,12] = {ll_case1(mu_hat):.4f}")

# CASE 2: Unknown μ AND σ²

# MLE for μ  [Eq. 16]
mu_hat2 = 0.0
for x in data:
    mu_hat2 += x
mu_hat2 /= n

# MLE for σ²  [Eq. 17]
var_hat = 0.0
for x in data:
    var_hat += (x - mu_hat2) ** 2
var_hat    /= n
sigma_hat2  = var_hat ** 0.5

# Log-likelihood l(μ, σ²)  [Eq. 8]
def ll_case2(mu_cand, sig_cand):
    ll = 0.0
    for x in data:
        ll += -0.5 * math.log(2 * math.pi * sig_cand**2) \
              - (x - mu_cand)**2 / (2 * sig_cand**2)
    return ll

# Derivatives  [Eq. 13]
def dl_dmu2(mu_cand, sig_cand):
    return sum((x - mu_cand) for x in data) / sig_cand**2

def dl_dsigma(mu_cand, sig_cand):
    return -n / sig_cand + sum((x - mu_cand)**2 for x in data) / sig_cand**3


print("CASE 2: Unknown μ AND σ²")
print(f"True μ = {TRUE_MU}")
print(f"True σ = {TRUE_SIGMA}")
print(f"MLE μ̂  [Eq.16] = {mu_hat2:.4f}")
print(f"MLE σ̂  [Eq.17] = {sigma_hat2:.4f}")
print(f"l(μ̂,σ̂) [Eq.5,12] = {ll_case2(mu_hat2, sigma_hat2):.4f}")


BG='white'; PANEL='#f7f7f7'; GRID='#dddddd'; TEXT='#111111'

def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.grid(color=GRID, linestyle='--', linewidth=0.6, alpha=0.8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

xs = np.linspace(TRUE_MU - 5*TRUE_SIGMA, TRUE_MU + 5*TRUE_SIGMA, 600)

# Plot 1: Case 1 
true_pdf1 = [gaussian_pdf(x, TRUE_MU, known_sigma) for x in xs]
mle_pdf1  = [gaussian_pdf(x, mu_hat,  known_sigma) for x in xs] 

ax1.plot(xs, true_pdf1, color='black', lw=2.5,
         label=f'True  μ={TRUE_MU}, σ={known_sigma}  (known)')
ax1.plot(xs, mle_pdf1,  color='#e07b54', lw=2.5, ls='--',
         label=f'MLE   μ̂={mu_hat:.2f}, σ={known_sigma}  (fixed)')
ax1.axvline(TRUE_MU, color='black',   lw=1.2, ls=':', alpha=0.6)
ax1.axvline(mu_hat,  color='#e07b54', lw=1.2, ls=':', alpha=0.9)
style(ax1, 'Case 1: Unknown μ, Σ Known\n[Eq. 11, 16]')
ax1.set_xlabel('x')
ax1.set_ylabel('p(x | θ)')
ax1.legend(fontsize=9)

# Plot 2: Case 2 
true_pdf2 = [gaussian_pdf(x, TRUE_MU,   TRUE_SIGMA) for x in xs]
mle_pdf2  = [gaussian_pdf(x, mu_hat2, sigma_hat2)   for x in xs]

ax2.plot(xs, true_pdf2, color='black', lw=2.5,
         label=f'True  μ={TRUE_MU}, σ={TRUE_SIGMA}')
ax2.plot(xs, mle_pdf2,  color='#5b8dd9', lw=2.5, ls='--',
         label=f'MLE   μ̂={mu_hat2:.2f}, σ̂={sigma_hat2:.2f}')
ax2.axvline(TRUE_MU,   color='black',   lw=1.2, ls=':', alpha=0.6)
ax2.axvline(mu_hat2,   color='#5b8dd9', lw=1.2, ls=':', alpha=0.9)
style(ax2, 'Case 2: Unknown μ and σ²\n[Eq. 16, 17]')
ax2.set_xlabel('x')
ax2.set_ylabel('p(x | θ)')
ax2.legend(fontsize=9)

fig.suptitle(f'Lab 4 – MLE Parameter Estimation\n'
              '',
             color=TEXT, fontsize=12, fontweight='bold', y=1.03)
plt.tight_layout()
out = 'lab4/mle_results.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)

