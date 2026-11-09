"""Count push-ups by following shoulder, elbow, and wrist movement in video."""

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mqtt_publisher import MqttRepPublisher
from squat_counter import VIDEO_SUFFIXES, console_quit_requested, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# YOLO's 17-point pose order: left arm and right arm (shoulder, elbow, wrist).
ELBOW_SIDES = ((5, 7, 9), (6, 8, 10))


def parse_args() -> argparse.Namespace:
    """Read the video source, angle thresholds, and optional output settings."""
    parser = argparse.ArgumentParser(
        description="Count completed push-ups using YOLO pose keypoints."
    )
    parser.add_argument(
        "--source",
        help="Video file path, camera index (0), or 'camera'. If omitted, prompt.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "yolo26m-pose.pt",
        help="Path to YOLO pose model weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or CUDA device number such as 0.",
    )
    parser.add_argument(
        "--bottom-angle",
        type=float,
        default=100.0,
        help="Maximum elbow angle (degrees) to count the push-up as lowered.",
    )
    parser.add_argument(
        "--top-angle",
        type=float,
        default=155.0,
        help="Minimum elbow angle (degrees) to count as returned to the top.",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for shoulder, elbow, and wrist keypoints.",
    )
    parser.add_argument(
        "--result-seconds",
        type=float,
        default=5.0,
        help="How long to keep the final video count visible after playback ends.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process a video without opening a display window.",
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Publish each completed repetition to the configured MQTT broker.",
    )
    return parser.parse_args()


def elbow_angle_degrees(
    keypoints: np.ndarray, side: tuple[int, int, int]
) -> float:
    """Calculate the shoulder-elbow-wrist angle in degrees."""
    shoulder_index, elbow_index, wrist_index = side
    shoulder = keypoints[shoulder_index]
    elbow = keypoints[elbow_index]
    wrist = keypoints[wrist_index]
    upper_arm = shoulder - elbow
    forearm = wrist - elbow
    denominator = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
    if denominator <= 1e-6:
        raise ValueError("Detected arm landmarks do not define a usable elbow angle.")
    # Clipping prevents tiny floating-point errors from breaking arccos.
    cosine = np.clip(np.dot(upper_arm, forearm) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def choose_arm(
    keypoints: np.ndarray,
    confidences: np.ndarray | None,
    minimum_confidence: float,
) -> tuple[int, int, int] | None:
    """Choose the visible arm whose three joint-confidence scores are strongest."""
    best_side = None
    best_score = -1.0
    for side in ELBOW_SIDES:
        if max(side) >= len(keypoints):
            continue
        if confidences is None:
            score = 1.0
        else:
            side_confidences = confidences[list(side)]
            if not np.all(side_confidences >= minimum_confidence):
                continue
            score = float(np.mean(side_confidences))
        if score > best_score:
            best_score = score
            best_side = side
    return best_side


class PushupRepCounter:
    """Count only a full top-to-bottom-to-top push-up movement."""

    def __init__(self, bottom_angle: float, top_angle: float) -> None:
        if not 0 < bottom_angle < top_angle <= 180:
            raise ValueError(
                "Angles must satisfy 0 < bottom-angle < top-angle <= 180."
            )
        self.bottom_angle = bottom_angle
        self.top_angle = top_angle
        self.repetitions = 0
        self.state = "waiting"

    def update(self, angle: float) -> tuple[int, str]:
        """Advance the movement state and return the count and on-screen prompt."""
        if self.state == "waiting":
            if angle >= self.top_angle:
                self.state = "top"
                return self.repetitions, "Ready — lower into a push-up"
            return self.repetitions, "Start in a high plank"

        if self.state == "top":
            if angle < self.top_angle:
                self.state = "descending"
            return self.repetitions, "Lower toward the floor"

        if self.state == "descending":
            if angle <= self.bottom_angle:
                self.state = "bottom"
                return self.repetitions, "Good depth — push back up"
            if angle >= self.top_angle:
                self.state = "top"
                return self.repetitions, "Partial push-up — not counted"
            return self.repetitions, "Keep lowering"

        if self.state == "bottom":
            if angle >= self.top_angle:
                self.repetitions += 1
                self.state = "top"
                return self.repetitions, "Push-up counted — lower again when ready"
            return self.repetitions, "Push all the way back up"

        raise RuntimeError(f"Unexpected push-up counter state: {self.state}")


def annotate_frame(
    frame: np.ndarray,
    pose_result,
    count: int,
    angle: float | None,
    status: str,
) -> np.ndarray:
    """Add the pose, count, measured angle, and guidance to one video frame."""
    annotated = pose_result.plot() if pose_result is not None else frame.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 112), (0, 0, 0), -1)
    angle_text = (
        f"Elbow angle: {angle:.0f} deg" if angle is not None else "Elbow angle: --"
    )
    cv2.putText(
        annotated,
        f"Push-ups: {count}",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{angle_text} | {status}",
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Side view recommended | Q/Esc quit",
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def run_counter(args: argparse.Namespace, source: str) -> int:
    """Open the selected camera/video, count reps, then close resources safely."""
    # Validate settings before loading a model or opening a camera.
    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO pose model not found: {args.model}")
    if not 0 < args.keypoint_confidence <= 1:
        raise ValueError("keypoint-confidence must be greater than 0 and at most 1.")
    if args.result_seconds < 0:
        raise ValueError("result-seconds cannot be negative.")

    device_name, _ = resolve_device(args.device)
    pose_model = YOLO(str(args.model))
    # "camera", "webcam", and a numeric camera index select a live camera.
    camera_source = source.lower() in {"camera", "webcam"} or source.isdigit()
    if args.headless and camera_source:
        raise ValueError("--headless can only be used with a video file.")

    if camera_source:
        capture_source = int(source) if source.isdigit() else 0
    else:
        video_path = Path(source)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError(f"Video file not found or unsupported: {video_path}")
        capture_source = str(video_path)

    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video or camera source: {source}")

    counter = PushupRepCounter(args.bottom_angle, args.top_angle)
    angle_history: deque[float] = deque(maxlen=5)
    frame_index = 0
    valid_angle_frames = 0
    status = "Start in a high plank"
    last_output = None
    stopped_by_user = False
    mqtt_publisher = None
    print("Press Q in the video window or terminal, Esc in the window, or Ctrl+C to stop.")

    window_name = "Fitness AI - Push-up Counter"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    try:
        if args.mqtt:
            mqtt_publisher = MqttRepPublisher("push-up")

        while True:
            # Read a frame, estimate pose, and measure the best visible arm.
            success, frame = capture.read()
            if not success:
                if camera_source:
                    raise RuntimeError("Could not read a frame from the camera.")
                break

            frame_index += 1
            pose_result = pose_model.predict(
                source=frame, device=device_name, verbose=False
            )[0]
            angle = None
            if pose_result.keypoints is not None and pose_result.boxes is not None:
                all_keypoints = pose_result.keypoints.xy.cpu().numpy()
                all_boxes = pose_result.boxes.xyxy.cpu().numpy()
                all_confidences = (
                    pose_result.keypoints.conf.cpu().numpy()
                    if pose_result.keypoints.conf is not None
                    else None
                )
                if len(all_keypoints) and len(all_boxes):
                    # In a busy frame, track the largest detected person.
                    areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (
                        all_boxes[:, 3] - all_boxes[:, 1]
                    )
                    person_index = int(np.argmax(areas))
                    confidences = (
                        all_confidences[person_index]
                        if all_confidences is not None
                        else None
                    )
                    arm = choose_arm(
                        all_keypoints[person_index],
                        confidences,
                        args.keypoint_confidence,
                    )
                    if arm is not None:
                        angle = elbow_angle_degrees(
                            all_keypoints[person_index], arm
                        )
                        # A short median filter smooths noisy angle changes.
                        angle_history.append(angle)
                        smoothed_angle = float(np.median(angle_history))
                        previous_repetitions = counter.repetitions
                        _, status = counter.update(smoothed_angle)
                        # Publish only when this frame completes a new repetition.
                        if (
                            mqtt_publisher is not None
                            and counter.repetitions > previous_repetitions
                        ):
                            mqtt_publisher.publish_repetitions(
                                counter.repetitions
                            )
                        valid_angle_frames += 1
                        angle = smoothed_angle
                    else:
                        angle_history.clear()
                        status = "Turn sideways; keep shoulder, elbow and wrist visible"
                else:
                    angle_history.clear()
                    status = "No person detected — show your upper body and arms"
            else:
                angle_history.clear()
                status = "No person detected — show your upper body and arms"

            # Draw the latest count and status, then optionally show the window.
            output = annotate_frame(
                frame, pose_result, counter.repetitions, angle, status
            )
            last_output = output
            if not args.headless:
                cv2.imshow(window_name, output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or console_quit_requested():
                    stopped_by_user = True
                    break
            elif console_quit_requested():
                stopped_by_user = True
                break

        # Leave the last result visible briefly after a saved video ends.
        if (
            not camera_source
            and not args.headless
            and not stopped_by_user
            and last_output is not None
            and args.result_seconds > 0
        ):
            deadline = time.monotonic() + args.result_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                final_frame = last_output.copy()
                cv2.putText(
                    final_frame,
                    f"Video finished - {remaining:.1f}s | Q to close",
                    (12, final_frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, final_frame)
                key = cv2.waitKey(min(100, max(1, int(remaining * 1000)))) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or console_quit_requested():
                    break
    finally:
        # Release camera/window/network resources even if processing fails.
        capture.release()
        cv2.destroyAllWindows()
        if mqtt_publisher is not None:
            mqtt_publisher.close()

    print(f"Finished: {counter.repetitions} push-up(s) counted.")
    print(
        f"Frames processed: {frame_index}; valid elbow-angle frames: "
        f"{valid_angle_frames}."
    )
    print(
        "A rep counts only after the detected elbow bends to the bottom threshold "
        "and returns to the top threshold."
    )
    return 0


def main() -> int:
    """Choose a source interactively when one was not supplied on the command line."""
    args = parse_args()
    source = args.source
    if source is None:
        source = input("Enter a video file path or 'camera' to use the webcam: ").strip()
    if not source:
        raise ValueError("A video path or camera source is required.")
    return run_counter(args, source)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPush-up counter stopped.")
