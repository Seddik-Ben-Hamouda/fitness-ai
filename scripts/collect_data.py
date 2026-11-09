"""Record short, labeled pose examples from the webcam for the classifier experiment."""

import cv2
import os
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Configuration: model file, output folder, and number of frames per example.
# -----------------------------
MODEL_PATH = "yolo26m-pose.pt"
DATASET_PATH = "dataset"

SEQUENCE_LENGTH = 30

# -----------------------------
# Load the pose detector once so it can be reused for each camera frame.
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Ask which label to save and create its output folder.
# -----------------------------
exercise = input("Enter exercise (pushup/squat): ").strip().lower()

if exercise not in ["pushup", "squat"]:
    print("Invalid exercise.")
    exit()

save_path = os.path.join(DATASET_PATH, exercise)
os.makedirs(save_path, exist_ok=True)

# -----------------------------
# Open the webcam and prepare a rolling buffer for one labeled example.
# -----------------------------
cap = cv2.VideoCapture(0)

sequence = []
sample_number = len(os.listdir(save_path))

print()
print("Controls:")
print("S = Save sequence")
print("Q = Quit")
print()
print(f"Collecting data for: {exercise}")

while cap.isOpened():

    # Read the next camera image; stop cleanly if the webcam is unavailable.
    success, frame = cap.read()

    if not success:
        print("Cannot access webcam")
        break

    # Detect body joints in this frame, then draw the detection for the preview.
    results = model(frame, device=0, verbose=False)

    annotated_frame = results[0].plot()

    # --------------------------------
    # Turn detected joint coordinates into one normalized frame of data.
    # --------------------------------
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:

        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        # Normalize coordinates so examples from different camera sizes compare.
        h, w = frame.shape[:2]

        keypoints[:, 0] /= w
        keypoints[:, 1] /= h

        # Flatten 17 (x, y) points into 34 model input values.
        keypoints = keypoints.flatten()

        # Keep only the most recent frames for the next saved sequence.
        sequence.append(keypoints)

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

    # --------------------------------
    # Show the selected exercise and how much of the example is ready.
    # --------------------------------
    cv2.putText(
        annotated_frame,
        f"Exercise: {exercise}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Fitness AI - Data Collection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------
    # S saves only a complete sequence, then starts collecting the next one.
    # --------------------------------
    if key == ord("s"):

        if len(sequence) == SEQUENCE_LENGTH:

            filename = os.path.join(
                save_path,
                f"{exercise}_{sample_number}.npy"
            )

            np.save(filename, np.array(sequence))

            print(f"Saved: {filename}")

            sample_number += 1
            sequence = []

        else:
            print("Not enough frames yet.")

    # Q exits the camera loop.
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()