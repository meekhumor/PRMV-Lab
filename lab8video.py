import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# HSV range for orange
lower = np.array([5,  120,  80])
upper = np.array([25, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(frame, "orange", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    cv2.imshow("Orange Tracker  [q to quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

