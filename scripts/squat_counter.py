import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mqtt_publisher import MqttRepPublisher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
SKELETON_PAIRS = ((5, 11), (11, 13), (13, 15))

if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count completed squats using YOLO pose keypoints."
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
        default=110.0,
        help="Maximum knee angle (degrees) to count the squat as deep enough.",
    )
    parser.add_argument(
        "--standing-angle",
        type=float,
        default=155.0,
        help="Minimum knee angle (degrees) to count as fully standing.",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for shoulder, hip, knee, and ankle keypoints.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process a video without opening a display window.",
    )
    parser.add_argument(
        "--result-seconds",
        type=float,
        default=5.0,
        help="How long to keep the final video count visible after playback ends.",
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Publish each completed repetition to the configured MQTT broker.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str, torch.device]:
    device_name = (
        "0" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )
    device = torch.device(
        f"cuda:{device_name}" if device_name.isdigit() else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment has no CUDA.")
    return device_name, device


def console_quit_requested() -> bool:
    if msvcrt is None or not msvcrt.kbhit():
        return False
    return msvcrt.getwch() in {"q", "Q"}


def choose_side(
    keypoints: np.ndarray,
    confidences: np.ndarray | None,
    minimum_confidence: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    sides = ((5, 11, 13, 15), (6, 12, 14, 16))
    best_side = None
    best_score = -1.0

    for side in sides:
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

    if best_side is None:
        return None
    return keypoints, best_side


def knee_angle_degrees(
    keypoints: np.ndarray, side: tuple[int, int, int, int]
) -> float:
    _, hip_index, knee_index, ankle_index = side
    hip = keypoints[hip_index]
    knee = keypoints[knee_index]
    ankle = keypoints[ankle_index]
    upper_leg = hip - knee
    lower_leg = ankle - knee
    denominator = np.linalg.norm(upper_leg) * np.linalg.norm(lower_leg)
    if denominator <= 1e-6:
        raise ValueError("Detected leg landmarks do not define a usable knee angle.")
    cosine = np.clip(np.dot(upper_leg, lower_leg) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class SquatRepCounter:
    def __init__(self, bottom_angle: float, standing_angle: float) -> None:
        if not 0 < bottom_angle < standing_angle <= 180:
            raise ValueError(
                "Angles must satisfy 0 < bottom-angle < standing-angle <= 180."
            )
        self.bottom_angle = bottom_angle
        self.standing_angle = standing_angle
        self.repetitions = 0
        self.state = "waiting"

    def update(self, angle: float) -> tuple[int, str]:
        if self.state == "waiting":
            if angle >= self.standing_angle:
                self.state = "standing"
                return self.repetitions, "Ready — lower into a squat"
            return self.repetitions, "Stand tall to start"

        if self.state == "standing":
            if angle < self.standing_angle:
                self.state = "descending"
            return self.repetitions, "Lower into a squat"

        if self.state == "descending":
            if angle <= self.bottom_angle:
                self.state = "bottom"
                return self.repetitions, "Good depth — stand back up"
            if angle >= self.standing_angle:
                self.state = "standing"
                return self.repetitions, "Partial squat — not counted"
            return self.repetitions, "Keep lowering"

        if self.state == "bottom":
            if angle >= self.standing_angle:
                self.repetitions += 1
                self.state = "standing"
                return self.repetitions, "Squat counted — lower again when ready"
            return self.repetitions, "Stand all the way up"

        raise RuntimeError(f"Unexpected squat-counter state: {self.state}")


def annotate_frame(
    frame: np.ndarray,
    pose_result,
    count: int,
    angle: float | None,
    status: str,
) -> np.ndarray:
    annotated = pose_result.plot() if pose_result is not None else frame.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 112), (0, 0, 0), -1)
    angle_text = f"Knee angle: {angle:.0f} deg" if angle is not None else "Knee angle: --"
    cv2.putText(
        annotated,
        f"Squats: {count}",
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
    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO pose model not found: {args.model}")
    if not 0 < args.keypoint_confidence <= 1:
        raise ValueError("keypoint-confidence must be greater than 0 and at most 1.")
    if args.result_seconds < 0:
        raise ValueError("result-seconds cannot be negative.")

    device_name, _ = resolve_device(args.device)
    pose_model = YOLO(str(args.model))
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

    counter = SquatRepCounter(args.bottom_angle, args.standing_angle)
    angle_history: deque[float] = deque(maxlen=5)
    frame_index = 0
    valid_angle_frames = 0
    status = "Stand tall to start"
    last_output = None
    stopped_by_user = False
    mqtt_publisher = None
    print("Press Q in the video window or terminal, Esc in the window, or Ctrl+C to stop.")

    window_name = "Fitness AI - Squat Counter"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    try:
        if args.mqtt:
            mqtt_publisher = MqttRepPublisher("squat")

        while True:
            success, frame = capture.read()
            if not success:
                if camera_source:
                    raise RuntimeError("Could not read a frame from the camera.")
                break

            frame_index += 1
            results = pose_model.predict(
                source=frame, device=device_name, verbose=False
            )
            pose_result = results[0]
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
                    areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (
                        all_boxes[:, 3] - all_boxes[:, 1]
                    )
                    person_index = int(np.argmax(areas))
                    confidences = (
                        all_confidences[person_index]
                        if all_confidences is not None
                        else None
                    )
                    selected = choose_side(
                        all_keypoints[person_index],
                        confidences,
                        args.keypoint_confidence,
                    )
                    if selected is not None:
                        points, side = selected
                        angle = knee_angle_degrees(points, side)
                        angle_history.append(angle)
                        smoothed_angle = float(np.median(angle_history))
                        valid_angle_frames += 1
                        previous_repetitions = counter.repetitions
                        _, status = counter.update(smoothed_angle)
                        if (
                            mqtt_publisher is not None
                            and counter.repetitions > previous_repetitions
                        ):
                            mqtt_publisher.publish_repetitions(
                                counter.repetitions
                            )
                        angle = smoothed_angle
                    else:
                        angle_history.clear()
                        status = "Turn sideways; keep hip, knee and ankle visible"
                else:
                    angle_history.clear()
                    status = "No person detected — show your full body"
            else:
                angle_history.clear()
                status = "No person detected — show your full body"

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
        capture.release()
        cv2.destroyAllWindows()
        if mqtt_publisher is not None:
            mqtt_publisher.close()

    print(f"Finished: {counter.repetitions} squat(s) counted.")
    print(f"Frames processed: {frame_index}; valid knee-angle frames: {valid_angle_frames}.")
    print(
        "A rep counts only after the detected knee bends to the bottom threshold "
        "and returns to the standing threshold."
    )
    return 0


def main() -> int:
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
        print("\nSquat counter stopped.")
