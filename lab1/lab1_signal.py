import numpy as np
import matplotlib.pyplot as plt

# Time vector (1 second duration)
t = np.linspace(0, 1, 1000)

# All famous basic signals (pure, no noise)
famous_signals = {
    "Sine Wave": np.sin(2 * np.pi * 5 * t),
    "Cosine Wave": np.cos(2 * np.pi * 5 * t),
    "Square Wave": np.sign(np.sin(2 * np.pi * 5 * t)),
    "Triangular Wave": 2 * np.abs(2 * (5 * t - np.floor(0.5 + 5 * t))) - 1,
    "Sawtooth Wave": 2 * (5 * t - np.floor(0.5 + 5 * t))
}

means = []
variances = []
histograms = []
bins_range = np.linspace(-1.5, 1.5, 30)

# Feature Extraction
for name, sig in famous_signals.items():
    m = np.mean(sig)
    v = np.var(sig)
    means.append(m)
    variances.append(v)
    
    counts, bin_edges = np.histogram(sig, bins=bins_range)
    histograms.append((counts, bin_edges))
    print(f"{name:16s} | Mean: {m:7.4f} | Variance: {v:7.4f}")

# Visualization: 
num_signals = len(famous_signals)
fig, axes = plt.subplots(num_signals + 1, 2, figsize=(11, 14))

for i, (name, sig) in enumerate(famous_signals.items()):
    # Waveform Plot
    axes[i, 0].plot(t[:300], sig[:300], color='tab:blue', linewidth=1.5)
    axes[i, 0].set_title(f'{name}', fontsize=9, pad=3)
    axes[i, 0].set_ylabel('Amp', fontsize=8)
    axes[i, 0].tick_params(labelsize=8)
    axes[i, 0].grid(True, linestyle='--', alpha=0.5)
    
    # Histogram Plot
    counts, bin_edges = histograms[i]
    axes[i, 1].bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color='purple', edgecolor='black', alpha=0.7)
    axes[i, 1].set_title(f'{name} Histogram', fontsize=9, pad=3)
    axes[i, 1].set_ylabel('Count', fontsize=8)
    axes[i, 1].tick_params(labelsize=8)

axes[num_signals - 1, 0].set_xlabel('Time (s)', fontsize=8)
axes[num_signals - 1, 1].set_xlabel('Amplitude', fontsize=8)

signal_labels = list(famous_signals.keys())

# Mean per Signal
bars1 = axes[num_signals, 0].bar(signal_labels, means, color='skyblue', edgecolor='black', width=0.5)
axes[num_signals, 0].set_title('Mean per Signal', fontsize=9, pad=3)
axes[num_signals, 0].set_ylabel('Mean', fontsize=8)
axes[num_signals, 0].tick_params(axis='x', rotation=15, labelsize=8)
axes[num_signals, 0].tick_params(axis='y', labelsize=8)
axes[num_signals, 0].axhline(0, color='gray', linewidth=0.8)

# Variance per Signal
bars2 = axes[num_signals, 1].bar(signal_labels, variances, color='lightgreen', edgecolor='black', width=0.5)
axes[num_signals, 1].set_title('Variance per Signal', fontsize=9, pad=3)
axes[num_signals, 1].set_ylabel('Variance', fontsize=8)
axes[num_signals, 1].tick_params(axis='x', rotation=15, labelsize=8)
axes[num_signals, 1].tick_params(axis='y', labelsize=8)
plt.subplots_adjust(hspace=0.65, wspace=0.35, left=0.09, right=0.95, top=0.96, bottom=0.06)
plt.show()

