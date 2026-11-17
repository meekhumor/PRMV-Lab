import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

classes = ['flower', 'dog']
data = {}

# Feature Extraction
for cls in classes:
    means, variances, histograms = [], [], []

    for file in sorted(os.listdir(cls))[:150]:
        img = cv2.imread(os.path.join(cls, file), 0)
        if img is None:
            continue

        means.append(np.mean(img))
        variances.append(np.var(img))

        hist = cv2.calcHist([img], [0], None, [32], [0, 256])
        histograms.append(cv2.normalize(hist, hist).flatten())

    data[cls] = [means, variances, np.array(histograms)]

    print(f"{cls}: {len(means)} images")
    print(f"Mean: {np.mean(means):.2f}, Variance: {np.mean(variances):.2f}")


# Visualization
plt.figure(figsize=(14, 10))

# 1. Mean Distribution
plt.subplot(2, 2, 1)
for cls in classes:
    plt.hist(data[cls][0], bins=20, alpha=0.6, label=cls)
plt.title("Mean Intensity")
plt.xlabel("Mean Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()

# 2. Variance Distribution
plt.subplot(2, 2, 2)
for cls in classes:
    plt.hist(data[cls][1], bins=20, alpha=0.6, label=cls)
plt.title("Variance")
plt.xlabel("Pixel Intensity Variance")
plt.ylabel("Frequency")
plt.legend()

# 3. Average Histogram
plt.subplot(2, 2, 3)
for cls in classes:
    plt.plot(np.mean(data[cls][2], axis=0), label=cls)
plt.title("Average Histogram (32 bins)")
plt.xlabel("Histogram Bin")
plt.ylabel("Normalized Frequency")
plt.legend()

# 4. Mean vs Variance
plt.subplot(2, 2, 4)
for cls in classes:
    plt.scatter(data[cls][0], data[cls][1], alpha=0.6, label=cls)
plt.title("Mean vs Variance")
plt.xlabel("Mean Pixel Intensity")
plt.ylabel("Pixel Intensity Variance")
plt.legend()

plt.tight_layout()
plt.show()