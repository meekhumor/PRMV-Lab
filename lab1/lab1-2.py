import cv2
import matplotlib.pyplot as plt
import glob
import math

# Load all images
image_paths = glob.glob('images/*.jpg')
num_images = len(image_paths)

cols = 5
rows = math.ceil(num_images / cols)

plt.figure(figsize=(15, 3 * rows))

# Subplot for each image
for i, path in enumerate(image_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    plt.subplot(rows, cols, i + 1)
    plt.plot(hist)
    plt.title(f'Image {i+1}')
    plt.xlabel('Intensity')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()
