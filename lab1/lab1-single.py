import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

image_paths = glob.glob('images/*.jpg')
image_path = image_paths[0] if image_paths else 'images/car.jpg'
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
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Grayscale Image ({image_path})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(hist, color='purple')
plt.title('Image Histogram')
plt.xlabel('Pixel Intensity (0 - 255)')
plt.ylabel('Pixel Count')

plt.subplot(1, 3, 3)
bars = plt.bar(['Mean', 'Variance'], [mean_val, var_val], color=['skyblue', 'lightgreen'])
plt.title('Mean & Variance')
plt.ylabel('Value')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()