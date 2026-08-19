import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load all images
image_paths = glob.glob('images/*.jpg')
image_names = [i + 1 for i in range(len(image_paths))]

means = []
variances = []
histograms = []

# Extract features
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('Grayscale Image', img)
    
    # 1. Mean
    means.append(np.mean(img))
    
    # 2. Variance
    variances.append(np.var(img))
    
    # 3. Histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    histograms.append(hist.flatten())

# Visualization
plt.figure(figsize=(15, 5))

# Mean per Image 
plt.subplot(1, 3, 1)
plt.bar(image_names, means)
plt.title('Mean Pixel Intensity per Image')
plt.xlabel('Images')
plt.ylabel('Mean Value')
plt.xticks(rotation=45)

# Variance per Image 
plt.subplot(1, 3, 2)
plt.bar(image_names, variances)
plt.title('Variance per Image')
plt.xlabel('Images')
plt.ylabel('Variance Value')
plt.xticks(rotation=45)

# Average Histogram Feature
plt.subplot(1, 3, 3)
plt.plot(np.mean(histograms, axis=0), color='purple')
plt.title('Average Image Histogram')
plt.xlabel('Pixel Intensity (0 - 255)')
plt.ylabel('Average Pixel Count')

plt.tight_layout()
plt.show()