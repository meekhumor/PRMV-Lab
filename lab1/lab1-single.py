import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'images/car.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Mean
mean_val = np.mean(img)

# 2. Variance
var_val = np.var(img)

# 3. Histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

print(f"Image: {image_path}")
print(f"Mean Pixel Intensity: {mean_val:.4f}")
print(f"Variance: {var_val:.4f}")

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(132)
plt.plot(hist, color='purple')
plt.title('Histogram')

plt.subplot(133)
plt.bar(['Mean', 'Variance'], [mean_val, var_val])
plt.title('Mean & Variance')

plt.tight_layout()
plt.show()