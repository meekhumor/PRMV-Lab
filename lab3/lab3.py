import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class GaussianBayesClassifier:
    def __init__(self):
        self.classes = np.array([0, 1])
        self.means   = {}
        self.covs    = {}
        self.priors  = {}

    def fit(self, X, y):
        N = len(y)
        for c in self.classes:
            X_c = X[y == c]
            self.means[c]  = np.mean(X_c, axis=0)
            cov = np.cov(X_c, rowvar=False)
            self.covs[c]   = np.atleast_2d(cov) + 1e-6 * np.eye(2)
            self.priors[c] = len(X_c) / N

    def _gaussian_pdf(self, X, mean, cov):
        D = X.shape[1]
        norm = 1.0 / np.sqrt(((2 * np.pi) ** D) * np.linalg.det(cov))
        diff = X - mean
        maha = np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)
        return norm * np.exp(-0.5 * maha)

    def predict(self, X):
        posts = np.zeros((len(X), 2))
        for idx, c in enumerate(self.classes):
            posts[:, idx] = self._gaussian_pdf(X, self.means[c], self.covs[c]) * self.priors[c]
        return np.argmax(posts, axis=1)


class LinearDiscriminantClassifier:
    def __init__(self):
        self.means   = {}   # mu_i per class
        self.priors  = {}   # P(omega_i)
        self.sigma2  = None 
        self.w_vecs  = {}   # w_i = mu_i / sigma^2
        self.w_bias  = {}   # w_i0

    def fit(self, X, y):
        classes = np.unique(y)
        N = len(y)

        # class means and priors 
        for c in classes:
            X_c = X[y == c]
            self.means[c]  = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / N

        pooled_var = 0.0
        for c in classes:
            X_c = X[y == c]
            pooled_var += np.sum((X_c - self.means[c]) ** 2)
        self.sigma2 = pooled_var / (N * X.shape[1])  # scalar σ²

        # w_i and w_i0 for each class 
        for c in classes:
            mu = self.means[c]
            self.w_vecs[c] = mu / self.sigma2
            self.w_bias[c] = (-0.5 / self.sigma2) * (mu @ mu) + np.log(self.priors[c])

    def _discriminant(self, X, c):
        return X @ self.w_vecs[c] + self.w_bias[c]

    def predict(self, X):
        g0 = self._discriminant(X, 0)
        g1 = self._discriminant(X, 1)
        # classify as class 1 if g1 > g0, else class 0
        return (g1 > g0).astype(int)

    @property
    def w(self):
        return self.w_vecs[1] - self.w_vecs[0]

    @property
    def b(self):
        return self.w_bias[1] - self.w_bias[0]

    def decision_boundary_x2(self, x1_vals):
        w = self.w
        if abs(w[1]) < 1e-10:
            return None
        return -(w[0] * x1_vals + self.b) / w[1]


def evaluate(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    cm = np.array([[tn, fp], [fn, tp]])
    return acc, cm


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir0 = os.path.join(base_dir, 'fruit')   # class 0
    dir1 = os.path.join(base_dir, 'dog')      # class 1
    class_names = ['Fruit', 'Dog']

    # Feature extraction 
    X_all, y_all = [], []
    for label, directory in [(0, dir0), (1, dir1)]:
        files = sorted(f for f in os.listdir(directory)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for fname in files:
            img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_GRAYSCALE)
            X_all.append([np.mean(img), np.std(img)])
            y_all.append(label)

    X = np.array(X_all, dtype=np.float64)
    y = np.array(y_all)

    # Train / Test split 
    np.random.seed(42)
    idx   = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
 
    lda   = LinearDiscriminantClassifier()
    bayes = GaussianBayesClassifier()
    lda.fit(X_train, y_train)
    bayes.fit(X_train, y_train)

    lda_pred   = lda.predict(X_test)
    bayes_pred = bayes.predict(X_test)

    lda_acc,   lda_cm   = evaluate(y_test, lda_pred)
    bayes_acc, bayes_cm = evaluate(y_test, bayes_pred)


    print(f"{'Linear Discriminant (LDA)':<28} {lda_acc*100:>9.2f}%")
    print(f"{'Gaussian Bayes':<28} {bayes_acc*100:>9.2f}%")

    print("\nLDA Confusion Matrix:")
    print(f"  {lda_cm}")
    print("\nGaussian Bayes Confusion Matrix:")
    print(f"  {bayes_cm}")

    print("\nGaussian Parameters")
    for c, name in enumerate(class_names):
        prior = bayes.priors[c]
        mean  = bayes.means[c]
        var   = np.diag(bayes.covs[c])
        print(f"\n{name}")
        print(f"  Prior    : {prior:.3f}")
        print(f"  Mean     : {mean}")
        print(f"  Variance : {var}")

    print("\nLDA Parameters")
    print(f"  Shared variance σ²  : {lda.sigma2:.4f}")
    for c, name in enumerate(class_names):
        print(f"\n  Class {name}")
        print(f"    w_{c}   = μ_{c}/σ²  : {lda.w_vecs[c]}")
        print(f"    w_{c}0  = bias     : {lda.w_bias[c]:.4f}")
    print(f"\n  Boundary (w1-w2)     : {lda.w}")
    print(f"  Boundary offset      : {lda.b:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Decision Boundaries: LDA vs Gaussian Bayes Classifier',
                 fontsize=14, fontweight='bold', y=1.01)

    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x2_min, x2_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 400),
                         np.linspace(x2_min, x2_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    colors_train = ['#4e79a7', '#f28e2b']  
    colors_test  = ['#76b7e8', '#ffbb77']

    for ax, clf, title, acc in [
        (axes[0], lda,   f'Linear Discriminant Classifier\nTest Accuracy: {lda_acc*100:.2f}%',   lda_acc),
        (axes[1], bayes, f'Gaussian Bayes Classifier\nTest Accuracy: {bayes_acc*100:.2f}%', bayes_acc),
    ]:
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.18,
                    colors=['#4e79a7', '#f28e2b'], levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    
        for c in [0, 1]:
            mask = y_train == c
            ax.scatter(X_train[mask, 0], X_train[mask, 1],
                       c=colors_train[c], s=8, alpha=0.4,
                       label=f'{class_names[c]}')

        correct = y_test == clf.predict(X_test)
        for c in [0, 1]:
            mask = y_test == c
            ax.scatter(X_test[mask, 0], X_test[mask, 1],
                       c=colors_test[c], s=20, alpha=0.9, edgecolors='black',
                       linewidths=0.4)

        for c in [0, 1]:
            m = (lda if clf is lda else bayes).means[c]
            ax.scatter(*m, marker='*', s=300, c='white',
                       edgecolors='black', linewidths=1.2, zorder=6,
                       label=f'Mean {class_names[c]}')

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel('Feature 1: Pixel Mean',     fontsize=10)
        ax.set_ylabel('Feature 2: Pixel Std Dev',  fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right', markerscale=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'decision_boundaries.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
