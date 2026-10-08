import cv2
from ultralytics import YOLO

# Pretrained pose model
model = YOLO("yolo26m-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Cannot access webcam")
        break

    # Run pose estimation
    results = model(frame, device=0, verbose=False)

    # Draw detected pose
    annotated_frame = results[0].plot()

    cv2.imshow("Fitness AI - YOLO Pose", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()