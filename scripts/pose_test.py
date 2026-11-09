"""Early webcam demo that draws a MediaPipe body skeleton on the video."""

import cv2
import mediapipe as mp

# Short names for MediaPipe's pose detector and drawing helper.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Keep the detector open for the full camera session.
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        # Read an image from the camera before trying to detect a pose.
        success, frame = cap.read()

        if not success:
            print("Cannot access webcam")
            break

        # OpenCV camera frames are BGR; MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find body landmarks in the converted frame.
        results = pose.process(rgb_frame)

        # Draw the landmark dots and connecting lines when a person is found.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Fitness AI - Pose Detection", frame)

        # Q closes the preview and ends the webcam loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()