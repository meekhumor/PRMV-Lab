import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
digits = load_digits()
X = digits.data          # (1797, 64)  – flattened 8x8 pixel grids
y = digits.target        # (1797,)     – digit labels 0-9
images = digits.images   # (1797, 8, 8)

print("=" * 60)
print("  Lab 11 – OCR System using SVM on Handwritten Digits")
print("=" * 60)
print(f"\nDataset shape   : {X.shape}")
print(f"Number of classes : {len(np.unique(y))}  ->  {np.unique(y)}")
print(f"Samples per class :")
for cls in np.unique(y):
    print(f"   Digit {cls} : {np.sum(y == cls)} samples")

# ─────────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. Train SVM Classifier
# ─────────────────────────────────────────────
clf = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
clf.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=[str(d) for d in range(10)])

print(f"\nTest Accuracy : {acc * 100:.2f}%")
print("\nClassification Report :")
print(report)
print("Confusion Matrix :")
print(cm)
print("=" * 60)

# ─────────────────────────────────────────────
# 5. Visualise
# ─────────────────────────────────────────────
BG      = '#0f1117'
PANEL   = '#1a1d27'
GRID    = '#2a2d3a'
TEXT    = '#e8eaf6'
ACCENT  = '#7c83fd'
COLORS  = ['#7c83fd', '#fd7c83', '#83fd7c', '#fdcf7c',
           '#7cfdf4', '#cf7cfd', '#fd7ccf', '#7cfdcf',
           '#fdfd7c', '#fd7c7c']

def ax_style(ax, title=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=6)

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle('Lab 11 – OCR System: Handwritten Digit Recognition (SVM)',
             color=TEXT, fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# ── Panel A: sample digit images (2 rows x 10 digits) ──────────────
ax_samples = fig.add_subplot(gs[0, :])
ax_samples.axis('off')
ax_samples.set_title('Panel A – Sample Digit Images from Dataset',
                     color=TEXT, fontsize=9, fontweight='bold', loc='left', pad=4)

inner_gs = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs[0, :],
                                            hspace=0.1, wspace=0.15)
for digit in range(10):
    idx   = np.where(digits.target == digit)[0][0]
    idx2  = np.where(digits.target == digit)[0][1]
    for row, sample_idx in enumerate([idx, idx2]):
        ax_img = fig.add_subplot(inner_gs[row, digit])
        ax_img.imshow(digits.images[sample_idx], cmap='gray', interpolation='nearest')
        ax_img.axis('off')
        if row == 0:
            ax_img.set_title(str(digit), color=COLORS[digit],
                             fontsize=9, fontweight='bold', pad=2)

# ── Panel B: Confusion Matrix heatmap ──────────────────────────────
ax_cm = fig.add_subplot(gs[1, :2])
cmap_custom = LinearSegmentedColormap.from_list(
    'ocr', ['#0f1117', '#2a1d5c', ACCENT], N=256)
im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap_custom)
plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
ticks = np.arange(10)
ax_cm.set_xticks(ticks); ax_cm.set_yticks(ticks)
ax_cm.set_xticklabels(ticks, color=TEXT, fontsize=8)
ax_cm.set_yticklabels(ticks, color=TEXT, fontsize=8)
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        ax_cm.text(j, i, str(cm[i, j]),
                   ha='center', va='center', fontsize=7,
                   color='white' if cm[i, j] < thresh else '#0f1117',
                   fontweight='bold')
ax_style(ax_cm, 'Panel B – Confusion Matrix')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')

# ── Panel C: Per-class accuracy bar chart ──────────────────────────
ax_acc = fig.add_subplot(gs[1, 2:])
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
bars = ax_acc.bar(range(10), per_class_acc, color=COLORS,
                  edgecolor=BG, linewidth=0.6, width=0.6)
for bar, val in zip(bars, per_class_acc):
    ax_acc.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=7.5, color=TEXT, fontweight='bold')
ax_acc.axhline(y=acc * 100, color='#fdcf7c', lw=1.5, ls='--',
               label=f'Overall: {acc*100:.2f}%')
ax_style(ax_acc, 'Panel C – Per-Class Recognition Accuracy')
ax_acc.set_xlabel('Digit Class')
ax_acc.set_ylabel('Accuracy (%)')
ax_acc.set_xticks(range(10))
ax_acc.set_ylim(0, 115)
ax_acc.legend(fontsize=8.5, facecolor=PANEL, labelcolor=TEXT)

# ── Panel D: PCA 2-D scatter of test features ──────────────────────
ax_pca = fig.add_subplot(gs[2, :2])
pca = PCA(n_components=2, random_state=42)
X_test_2d = pca.fit_transform(X_test)
for digit in range(10):
    mask = y_test == digit
    ax_pca.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
                   color=COLORS[digit], label=str(digit),
                   s=18, alpha=0.75, edgecolors='none')
ax_style(ax_pca, 'Panel D – PCA 2-D Feature Space (Test Set)')
ax_pca.set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax_pca.set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax_pca.legend(fontsize=7, ncol=5, facecolor=PANEL, labelcolor=TEXT,
              loc='upper right', markerscale=1.4)

# ── Panel E: sample predictions (correct / wrong) ──────────────────
ax_pred = fig.add_subplot(gs[2, 2:])
ax_pred.axis('off')
ax_pred.set_facecolor(PANEL)
ax_pred.set_title('Panel E – Sample Predictions  correct / wrong',
                  color=TEXT, fontsize=9, fontweight='bold', loc='left', pad=4)

correct_idx = np.where(y_pred == y_test)[0][:5]
wrong_idx   = np.where(y_pred != y_test)[0][:5]
show_idx    = list(correct_idx) + list(wrong_idx)
labels_flag = ['correct'] * len(correct_idx) + ['wrong'] * len(wrong_idx)

inner_pred = gridspec.GridSpecFromSubplotSpec(
    2, max(len(correct_idx), len(wrong_idx)),
    subplot_spec=gs[2, 2:], hspace=0.35, wspace=0.25)

for col, (si, flag) in enumerate(zip(show_idx, labels_flag)):
    row_idx = 0 if flag == 'correct' else 1
    col_idx = col % len(correct_idx)
    ax_p = fig.add_subplot(inner_pred[row_idx, col_idx])
    orig_img = scaler.inverse_transform(X_test[si].reshape(1, -1)).reshape(8, 8)
    ax_p.imshow(orig_img, cmap='gray', interpolation='nearest')
    ax_p.axis('off')
    color = '#83fd7c' if flag == 'correct' else '#fd7c83'
    mark  = 'OK' if flag == 'correct' else 'ERR'
    title = f'T:{y_test[si]} P:{y_pred[si]} {mark}'
    ax_p.set_title(title, color=color, fontsize=7, fontweight='bold', pad=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = 'lab11/ocr_results.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved -> {out_path}")
