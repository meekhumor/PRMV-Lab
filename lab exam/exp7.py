import cv2
import easyocr
import matplotlib.pyplot as plt

# 1. Load Image 
path = 'images/signboard.png'
img = cv2.imread(path)

# 2. Image Preprocessing (Grayscale + Resize + CLAHE) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
processed = clahe.apply(gray)

# 3. Initialize EasyOCR & Run Text Extraction 
reader = easyocr.Reader(['hi', 'en'], gpu=False)
results = reader.readtext(processed, detail=1, paragraph=False)

# 4. Print Extracted Text 
print("Extracted Text: ")
for box, text, confidence in results:
    print(f"{text:<25} (confidence: {confidence:.2f})")

# 5. Visualization 
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Optical Character Recognition (OCR)', fontweight='bold')
plt.axis("off")
plt.tight_layout()
plt.show()
