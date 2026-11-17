import cv2
import easyocr
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  OPTICAL CHARACTER RECOGNITION (OCR) SYSTEM USING EasyOCR
#
#  Preprocessing & Feature Enhancement Pipeline:
#  1. Grayscale Conversion:
#     Y = 0.299*R + 0.587*G + 0.114*B (reduces dimensionality from 3 channels to 1)
#
#  2. 2x Bilinear Interpolation Upscaling:
#     Enlarges character contours and edge details for higher detection accuracy
#
#  3. CLAHE (Contrast Limited Adaptive Histogram Equalization):
#     - Divides image into 8x8 contextual tiles
#     - Clips histogram peak at clipLimit=2.0 to prevent noise amplification
#     - Enhances local contrast between text characters and signboard background
#
#  4. Multilingual OCR Recognition:
#     - EasyOCR Reader with Hindi ('hi') and English ('en') language models
# ══════════════════════════════════════════════════════════════════

# ── 1. Load Image ─────────────────────────────────────────────────
filename = 'images/signboard.png'
img = cv2.imread(filename)

if img is None:
    raise ValueError(f"Could not read image: {filename}")

# ── 2. Image Preprocessing (Grayscale + Resize + CLAHE) ───────────
# Step A: Convert BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step B: 2x Upscaling (fx=2, fy=2) to improve character resolution
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Step C: CLAHE (Adaptive Histogram Equalization for localized contrast boost)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
processed = clahe.apply(gray)

# ── 3. Initialize EasyOCR & Run Text Extraction ───────────────────
# ['hi', 'en']: Hindi and English support
reader = easyocr.Reader(['hi', 'en'], gpu=False)

# detail=1: returns [(bbox, text, confidence), ...]
results = reader.readtext(processed, detail=1, paragraph=False)

# ── 4. Print Extracted Text ───────────────────────────────────────
print("=" * 45)
print("           EXTRACTED TEXT")
print("=" * 45)
for box, text, confidence in results:
    print(f"{text:<25} (confidence: {confidence:.2f})")
print("=" * 45)

# ── 5. Visualization ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Optical Character Recognition (OCR) — Input Signboard', fontweight='bold')
plt.axis("off")
plt.tight_layout()
plt.show()
