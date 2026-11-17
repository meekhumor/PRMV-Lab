import matplotlib.pyplot as plt
from skimage import data
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════
#  OBJECT DETECTION SYSTEM USING YOLO (You Only Look Once - v8)
#
#  Mathematical Formulation & Key Concepts:
#  1. Single-Stage Detection:
#     - Treats object detection as a single regression problem from
#       image pixels to bounding box coordinates & class probabilities.
#
#  2. Confidence Score:
#     Confidence = P(Object) * IoU_truth_pred
#     Measures both the probability that class appears in box and how
#     well predicted box fits the ground truth object.
#
#  3. Intersection over Union (IoU):
#     IoU = Area(B_pred ∩ B_truth) / Area(B_pred ∪ B_truth)
#     - Used in Non-Maximum Suppression (NMS) to eliminate duplicate boxes.
#     - iou=0.5: if two boxes overlap by > 50%, suppress lower confidence.
#
#  4. Bounding Box Representation:
#     xyxy = [x1, y1, x2, y2] (top-left and bottom-right corner pixel coords)
# ══════════════════════════════════════════════════════════════════

# ── 1. Load Input Image ───────────────────────────────────────────
# data.cat() provides a standard built-in test image from skimage (RGB)
image = data.cat()

# ── 2. Load Pretrained YOLO Model ─────────────────────────────────
# yolov8s.pt: YOLOv8 Small model pretrained on COCO dataset (80 classes)
model = YOLO('yolov8s.pt')

# ── 3. Run Inference ──────────────────────────────────────────────
# conf = 0.35 : Minimum confidence threshold to retain detection
# iou  = 0.50 : Non-Maximum Suppression (NMS) overlap threshold
# imgsz = 640 : Input resolution resized for inference
results = model(image, conf=0.35, iou=0.50, imgsz=640)

# First (and only) image result
result = results[0]
print(f"Successfully detected {len(result.boxes)} object(s).\n")

# ── 4. Extract Bounding Boxes, Confidence, & Class Labels ──────────
for i, box in enumerate(result.boxes):
    # Bounding box corners: [x1, y1, x2, y2]
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Confidence score ∈ [0, 1]
    confidence = float(box.conf[0])

    # Class ID and class name
    class_id = int(box.cls[0])
    class_name = result.names[class_id]

    print(f"Object {i+1}: {class_name.capitalize()} | "
          f"Confidence: {confidence:.2f} | "
          f"Box: (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})")

# ── 5. Visualization ──────────────────────────────────────────────
# result.plot() automatically draws bounding boxes & labels on the image
annotated_image = result.plot()

plt.figure(figsize=(8, 8))
plt.imshow(annotated_image)
plt.axis('off')
plt.title('Object Detection — Bounding Boxes & Class Predictions', fontweight='bold')
plt.tight_layout()
plt.show()
