import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Linear Discriminant Classifier
class LinearDiscriminantClassifier:
    def __init__(self):
        self.means  = {}   # μ_c 
        self.priors = {}   # P(w_c) 
        self.w_vecs = {}   # w_c = μ_c / σ²   
        self.w_bias = {}   # w_c0 = -||μ_c||² / (2σ²) + ln P(w_c)  
        self.sigma2 = None # σ² 

    def fit(self, X, y):
        classes = np.unique(y)
        N = len(y)

        # Estimate class means and priors
        for c in classes:
            Xc = X[y == c]
            self.means[c]  = np.mean(Xc, axis=0)   # μ_c = (1/N_c) Σ x_i
            self.priors[c] = len(Xc) / N           # P(w_c) = N_c / N

        # Pooled scalar variance: σ² = Σ_c Σ_{x∈w_c} ||x - μ_c||² / (N * D)
        pooled = sum(np.sum((X[y == c] - self.means[c])**2) for c in classes)
        self.sigma2 = pooled / (N * X.shape[1])

        # Weight vector and bias for each class:
        # w_c = μ_c / σ²
        # w_c0 = -||μ_c||² / (2σ²) + ln P(w_c)
        for c in classes:
            mu = self.means[c]
            self.w_vecs[c] = mu / self.sigma2
            self.w_bias[c] = (-0.5 / self.sigma2) * (mu @ mu) + np.log(self.priors[c])

    def predict(self, X):
        # Discriminant function: g_c(x) = w_c^T * x + w_c0
        g0 = X @ self.w_vecs[0] + self.w_bias[0]   # Score for class 0
        g1 = X @ self.w_vecs[1] + self.w_bias[1]   # Score for class 1

        return (g1 > g0).astype(int)


# 2. Gaussian Bayes Classifier 
class GaussianBayesClassifier:
    def __init__(self):
        self.means = {}   # μ_c 
        self.covs = {}    # Σ_c 
        self.priors = {}  # P(w_c) 

    def fit(self, X, y):
        N = len(y)
        for c in np.unique(y):
            Xc = X[y == c]
            self.means[c]  = np.mean(Xc, axis=0)
            self.covs[c]   = np.atleast_2d(np.cov(Xc, rowvar=False)) + 1e-6 * np.eye(2)
            self.priors[c] = len(Xc) / N

    def _pdf(self, X, mean, cov):
        # Multivariate Gaussian PDF:
        # p(x | w_c) = 1/sqrt((2π)^D |Σ_c|) * exp(-0.5 * (x-μ)^T Σ^-1 (x-μ))
        D = X.shape[1]
        norm = 1.0 / np.sqrt(((2 * np.pi)**D) * np.linalg.det(cov))
        diff = X - mean
        # Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        maha = np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)
        return norm * np.exp(-0.5 * maha)

    def predict(self, X):
        # ŷ = argmax_c [ p(x | w_c) * P(w_c) ]
        post = np.column_stack([
            self._pdf(X, self.means[c], self.covs[c]) * self.priors[c]
            for c in [0, 1]
        ])
        return np.argmax(post, axis=1)


# Load images & extract features 
# Features: f1 = pixel mean,  f2 = pixel std dev
X_all, y_all = [], []
for label, folder in [(0, 'fruit'), (1, 'dog')]:
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            X_all.append([np.mean(img), np.std(img)])
            y_all.append(label)

X = np.array(X_all, dtype=np.float64)
y = np.array(y_all)
print(f"Total: {len(X)}  (Fruit={sum(y==0)}, Dog={sum(y==1)})")

# Train / Test split 
np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# Train both classifiers 
lda = LinearDiscriminantClassifier(); lda.fit(X_train, y_train)
bayes = GaussianBayesClassifier(); bayes.fit(X_train, y_train)

# Evaluate both 
for name, clf in [('LDA', lda), ('Bayes', bayes)]:
    pred = clf.predict(X_test)
    acc  = np.mean(y_test == pred)
    tp = np.sum((y_test==1)&(pred==1)); tn = np.sum((y_test==0)&(pred==0))
    fp = np.sum((y_test==0)&(pred==1)); fn = np.sum((y_test==1)&(pred==0))
    print(f"\n{name}  Accuracy: {acc*100:.2f}%")
    print(f"  Confusion Matrix: [[TN={tn} FP={fp}] [FN={fn} TP={tp}]]")

print(f"\nLDA Parameters:")
print(f"Shared σ² = {lda.sigma2:.4f}")
print(f"Boundary vector (w1-w0) = {lda.w_vecs[1] - lda.w_vecs[0]}")
print(f"Boundary offset (b1-b0) = {lda.w_bias[1] - lda.w_bias[0]:.4f}")

x1 = np.linspace(X[:,0].min()-5, X[:,0].max()+5, 400)
x2 = np.linspace(X[:,1].min()-5, X[:,1].max()+5, 400)
xx, yy = np.meshgrid(x1, x2)
grid = np.c_[xx.ravel(), yy.ravel()]

class_names = ['Fruit', 'Dog']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, clf, name in zip(axes, [lda, bayes], ['LDA', 'Gaussian Bayes']):
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    ax.contour(xx, yy, Z, levels=[0.5], colors='black')

    for c, col in zip([0, 1], ['blue', 'orange']):
        ax.scatter(*X_train[y_train==c].T, c=col, s=8, label=class_names[c])
        ax.scatter(*clf.means[c], marker='*', s=200, c='black')

    acc = np.mean(y_test == clf.predict(X_test)) * 100
    ax.set_title(f'{name} (Accuracy: {acc:.2f}%)')
    ax.set_xlabel('Pixel Mean')
    ax.set_ylabel('Pixel Std Dev')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
