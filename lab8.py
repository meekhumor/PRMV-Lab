import matplotlib.pyplot as plt
from skimage import data
from ultralytics import YOLO

image = data.cat()
model = YOLO('yolov8s.pt')

results = model(image, conf=0.35, iou=0.5, imgsz=640)  # conf = confidence threshold, iou = NMS overlap threshold

result = results[0]
print(f'Successfully detected {len(result.boxes)} object(s).\n')

for i, box in enumerate(result.boxes):  # Extract bounding box coordinates [x1, y1, x2, y2], confidence score, and class label
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = result.names[class_id]

    print(
        f'Object {i+1}: {class_name.capitalize()} | Confidence: {confidence:.2f}'
        f' | Box: (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})'
    )

annotated_image = result.plot()

plt.figure(figsize=(8, 8))
plt.imshow(annotated_image)
plt.axis('off')
plt.title(
    'Object Detection - Bounding Boxes & Class Predictions',
    fontsize=14,
    fontweight='bold',
)
plt.tight_layout()
plt.show()
