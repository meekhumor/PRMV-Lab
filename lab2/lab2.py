import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class GaussianBayesClassifier2Class:
    def __init__(self, priors=None):
        self.priors = priors
        self.classes = np.array([0, 1])
        self.means = {}
        self.covs = {}
        self.estimated_priors = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        N = len(y)
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            cov = np.cov(X_c, rowvar=False) if X.shape[1] > 1 else np.atleast_2d(np.var(X_c, ddof=1))
            self.covs[c] = np.atleast_2d(cov) + 1e-6 * np.eye(np.atleast_2d(cov).shape[0])
            self.estimated_priors[c] = len(X_c) / N
        if self.priors is None:
            self.priors = [self.estimated_priors[0], self.estimated_priors[1]]
        else:
            self.priors = np.array(self.priors, dtype=np.float64) / np.sum(self.priors)

    def gaussian_pdf(self, X, mean, cov):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        D = X.shape[1]
        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** D) * np.linalg.det(cov))
        diff = X - mean
        mahalanobis = np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)
        return norm_const * np.exp(-0.5 * mahalanobis)

    def predict_proba(self, X):
        # P(w_i | x) = p(x | w_i) * P(w_i) / p(x)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        likelihoods = np.zeros((len(X), 2))
        for idx, c in enumerate(self.classes):
            likelihoods[:, idx] = self.gaussian_pdf(X, self.means[c], self.covs[c])
        joint = likelihoods * np.array([self.priors[0], self.priors[1]])
        evidence = np.sum(joint, axis=1)
        evidence = np.where(evidence == 0, 1e-12, evidence)
        posteriors = joint / evidence[:, np.newaxis]
        return posteriors, likelihoods, evidence

    def predict(self, X):
        posteriors, _, _ = self.predict_proba(X)
        return np.argmax(posteriors, axis=1)


def evaluate_classifier(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "precision": precision, "recall": recall, "f1_score": f1
    }


if __name__ == '__main__':

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir1 = os.path.join(base_dir, 'flower')
    dir2 = os.path.join(base_dir, 'dog')

    X_features, y_labels = [], []

    for label, directory in [(0, dir1), (1, dir2)]:
        files = sorted(f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for filename in files:
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            X_features.append([np.mean(img), np.std(img)])
            y_labels.append(label)

    X = np.array(X_features)
    y = np.array(y_labels)
    print(f"Total: {len(X)} images  (Flower={sum(y==0)}, Dog={sum(y==1)})")

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    clf = GaussianBayesClassifier2Class()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    m = evaluate_classifier(y_test, y_pred)
    print(f"\nTest Accuracy : {m['accuracy']*100:.2f}%")
    print(f"Precision     : {m['precision']:.4f}")
    print(f"Recall        : {m['recall']:.4f}")
    print(f"F1-Score      : {m['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{m['confusion_matrix']}")

    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('Gaussian Bayes — 1D Class-Conditional Distributions', fontsize=13, fontweight='bold')

    for i, (ax, title) in enumerate(zip(axes, ['Pixel Mean', 'Pixel Std Dev'])):
        mu0, mu1   = clf.means[0][i], clf.means[1][i]
        sig0, sig1 = np.sqrt(clf.covs[0][i, i]), np.sqrt(clf.covs[1][i, i])
        x = np.linspace(min(mu0 - 4*sig0, mu1 - 4*sig1),
                        max(mu0 + 4*sig0, mu1 + 4*sig1), 500)
        pdf0 = (1/(sig0*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu0)/sig0)**2)
        pdf1 = (1/(sig1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sig1)**2)
        ax.plot(x, pdf0, 'b-', linewidth=2.5, label=f'Flower  μ={mu0:.1f}, σ={sig0:.1f}')
        ax.plot(x, pdf1, 'r-', linewidth=2.5, label=f'Dog     μ={mu1:.1f}, σ={sig1:.1f}')
        ax.fill_between(x, pdf0, alpha=0.15, color='blue')
        ax.fill_between(x, pdf1, alpha=0.15, color='red')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature Value', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bayes_1d_gaussians.png')
    plt.savefig(out1, dpi=150)
    plt.show()

    # 6. Plot: 2D Gaussian 
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    z0 = clf.gaussian_pdf(grid, clf.means[0], clf.covs[0]).reshape(xx.shape)
    z1 = clf.gaussian_pdf(grid, clf.means[1], clf.covs[1]).reshape(xx.shape)

    ax2.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
                c='blue', s=15, alpha=0.5, label='Flower (test)')
    ax2.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
                c='red',  s=15, alpha=0.5, label='Dog (test)')

    ax2.scatter(*clf.means[0], c='blue', marker='X', s=200, edgecolors='k', zorder=5, label='Mean Flower')
    ax2.scatter(*clf.means[1], c='red',  marker='X', s=200, edgecolors='k', zorder=5, label='Mean Dog')

    ax2.set_xlabel('Pixel Mean', fontsize=11)
    ax2.set_ylabel('Pixel Std Dev', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bayes_2d_gaussians.png')
    plt.savefig(out2, dpi=150)
    plt.show()
