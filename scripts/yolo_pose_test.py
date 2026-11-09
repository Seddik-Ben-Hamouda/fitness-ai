"""Minimal webcam preview of the YOLO pose model's body-joint detections."""

import cv2
from ultralytics import YOLO

# Load the pose model weights stored in the project folder.
model = YOLO("yolo26m-pose.pt")

# Open the default webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Get a new image and stop if the camera cannot provide one.
    success, frame = cap.read()

    if not success:
        print("Cannot access webcam")
        break

    # Detect body joints and draw them on a copy of the image.
    results = model(frame, device=0, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Fitness AI - YOLO Pose", annotated_frame)

    # Q closes the preview window.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()