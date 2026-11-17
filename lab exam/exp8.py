import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
source = "input.mov" 
conf = 0.35

cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame, conf=conf, verbose=False)

    # Draw bounding boxes and labels
    count = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f"{model.names[cls]} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1

    # Display FPS and detection count
    cv2.putText(frame, f"Detections: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()