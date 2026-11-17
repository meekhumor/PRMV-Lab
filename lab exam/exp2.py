import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class GaussianBayesClassifier:
    def __init__(self):
        self.means = {}  # μ_c 
        self.covs = {}   # Σ_c  
        self.priors = {}  # P(w_c) 

    def fit(self, X, y):
        N = len(y)
        for c in [0, 1]:
            Xc = X[y == c]

            self.means[c] = np.mean(Xc, axis=0)  # μ_c = Σ x_i / N_c
            self.covs[c] = np.atleast_2d(np.cov(Xc, rowvar=False)) + 1e-6 * np.eye(2)
            self.priors[c] = len(Xc) / N  # P(w_c) = N_c / N 

    def gaussian_pdf(self, X, mean, cov):
        # Multivariate Gaussian PDF:
        # p(x | w_c) = 1 / sqrt((2π)^D * |Σ_c|) * exp(-0.5 * (x-μ)^T Σ^-1 (x-μ))
        D = X.shape[1]
        norm = 1.0 / np.sqrt(((2 * np.pi) ** D) * np.linalg.det(cov))
        diff = X - mean   # (x - μ_c)
        maha = np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)

        return norm * np.exp(-0.5 * maha)   # likelihood

    def predict(self, X):
        # P(w_c | x) ∝ p(x | w_c) * P(w_c)   
        post = np.column_stack([
            self.gaussian_pdf(X, self.means[c], self.covs[c]) * self.priors[c] for c in [0, 1]
        ])

        # Predict the class with the highest posterior
        # ŷ = argmax_c [ p(x | w_c) * P(w_c) ]
        return np.argmax(post, axis=1)


# Load images & extract features
# Features used: f1 = pixel mean, f2 = pixel std dev
X_all, y_all = [], []
for label, folder in [(0, 'flower'), (1, 'dog')]:
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            X_all.append([np.mean(img), np.std(img)]) 
            y_all.append(label)

X = np.array(X_all)
y = np.array(y_all)
print(f"Total: {len(X)}  (Flower={sum(y==0)}, Dog={sum(y==1)})")

# Train / Test split
np.random.seed(42)
idx = np.random.permutation(len(X))   
split = int(0.8 * len(X))              
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# Train & Predict 
clf = GaussianBayesClassifier()
clf.fit(X_train, y_train)    
y_pred = clf.predict(X_test) 

acc  = np.mean(y_test == y_pred)
tp = np.sum((y_test == 1) & (y_pred == 1))   
tn = np.sum((y_test == 0) & (y_pred == 0))   
fp = np.sum((y_test == 0) & (y_pred == 1))   
fn = np.sum((y_test == 1) & (y_pred == 0))   
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec:.4f}  Recall : {rec:.4f}  F1 : {f1:.4f}")
print(f"Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")

# 1D Gaussian plot (Pixel Mean & Pixel Std Dev)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (ax, title) in enumerate(zip(axes, ['Pixel Mean', 'Pixel Std Dev'])):
    mu0, mu1 = clf.means[0][i], clf.means[1][i]
    s0, s1 = np.sqrt(clf.covs[0][i,i]), np.sqrt(clf.covs[1][i,i])

    x = np.linspace(min(mu0-4*s0, mu1-4*s1), max(mu0+4*s0, mu1+4*s1), 500)
    for mu, s, color, label in [(mu0,s0,'b','Flower'), (mu1,s1,'r','Dog')]:
        pdf = np.exp(-0.5*((x-mu)/s)**2) / (s*np.sqrt(2*np.pi))
        ax.plot(x, pdf, color, label=label)
        ax.fill_between(x, pdf, alpha=0.15, color=color)

    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()