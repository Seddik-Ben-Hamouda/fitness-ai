"""Show the experimental exercise classifier's prediction on video or webcam."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Windows terminal input needs msvcrt; other platforms use the window controls.
if sys.platform == "win32":
    import msvcrt
else:
    msvcrt = None

from extract_video_poses import CLASS_NAMES, SEQUENCE_LENGTH
from train_exercise_classifier import ExerciseClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSE_MODEL = PROJECT_ROOT / "yolo26m-pose.pt"
DEFAULT_CLASSIFIER = PROJECT_ROOT / "models" / "exercise_classifier.pt"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def parse_args() -> argparse.Namespace:
    """Read input source, saved model paths, and preview options."""
    parser = argparse.ArgumentParser(
        description="Visually test the exercise classifier on a video or webcam."
    )
    parser.add_argument(
        "--source",
        help="Video path, camera index (0), or 'camera'. If omitted, prompt for a source.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=DEFAULT_CLASSIFIER,
        help="Path to the trained exercise-classifier checkpoint.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=DEFAULT_POSE_MODEL,
        help="Path to YOLO pose model weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or a CUDA device number such as 0.",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        help="Optionally save the annotated final video frame to an image file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a display window; useful for automated video checks.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str, torch.device]:
    """Choose CPU/GPU and report when CUDA was requested but is unavailable."""
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


def load_models(args: argparse.Namespace, device_name: str, device: torch.device):
    """Load the trained classifier and pose detector from their saved files."""
    if not args.classifier.is_file():
        raise FileNotFoundError(f"Trained classifier not found: {args.classifier}")
    if not args.pose_model.is_file():
        raise FileNotFoundError(f"YOLO pose model not found: {args.pose_model}")

    # Check the class labels before loading weights to avoid mismatched predictions.
    checkpoint = torch.load(
        args.classifier, map_location="cpu", weights_only=True
    )
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError(
            f"Classifier classes do not match the expected classes {CLASS_NAMES}."
        )
    classifier = ExerciseClassifier(
        checkpoint["model_type"],
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        len(CLASS_NAMES),
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device)
    classifier.eval()

    pose_model = YOLO(str(args.pose_model))
    print(f"Using device: {device_name}")
    return classifier, pose_model


def selected_person(result, width: int, height: int) -> np.ndarray | None:
    """Return normalized pose points for the largest person in an image."""
    if result.keypoints is None or result.boxes is None:
        return None

    keypoints = result.keypoints.xy.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(keypoints) == 0 or len(boxes) == 0:
        return None

    # The largest box is treated as the person performing the exercise.
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    person_keypoints = keypoints[int(np.argmax(box_areas))].astype(np.float32)
    # Normalize coordinates to the same 0..1 scale used during data preparation.
    person_keypoints[:, 0] /= width
    person_keypoints[:, 1] /= height
    return person_keypoints


def classify_sequence(
    sequence: np.ndarray,
    classifier: ExerciseClassifier,
    device: torch.device,
) -> tuple[str, float]:
    """Return the predicted exercise name and its estimated confidence."""
    if sequence.shape != (SEQUENCE_LENGTH, 17, 2):
        raise ValueError(
            f"Expected sequence shape {(SEQUENCE_LENGTH, 17, 2)}, "
            f"got {sequence.shape}."
        )

    # Convert the 30-frame pose sequence into the classifier's expected tensor.
    inputs = torch.from_numpy(
        sequence.reshape(SEQUENCE_LENGTH, 34).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(classifier(inputs), dim=1)[0]
    confidence, predicted_index = probabilities.max(dim=0)
    return CLASS_NAMES[int(predicted_index.item())], float(confidence.item())


def add_prediction_text(
    frame: np.ndarray,
    message: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw one readable message banner over an image."""
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        output,
        message,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def console_quit_requested() -> bool:
    """Check for Q in a Windows terminal without pausing video processing."""
    if msvcrt is None or not msvcrt.kbhit():
        return False
    return msvcrt.getwch() in {"q", "Q"}


def process_video(
    video_path: Path,
    pose_model: YOLO,
    classifier: ExerciseClassifier,
    device: torch.device,
    device_name: str,
    save_preview: Path | None,
    display: bool,
) -> int:
    """Sample a saved video, classify its pose sequence, and show/save a preview."""
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise FileNotFoundError(f"Video file not found or unsupported: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < SEQUENCE_LENGTH:
            raise ValueError(
                f"Video has {frame_count} frames; at least {SEQUENCE_LENGTH} are needed."
            )

        # Use evenly spaced frames to represent the entire clip in fixed-size input.
        frame_indices = np.linspace(
            0, frame_count - 1, num=SEQUENCE_LENGTH, dtype=np.int64
        )
        sequence = []
        last_frame = None
        # Detect one pose in each sampled frame and display progress as it runs.
        for position, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

            result = pose_model.predict(
                source=frame, device=device_name, verbose=False
            )[0]
            height, width = frame.shape[:2]
            keypoints = selected_person(result, width, height)
            if keypoints is None:
                raise RuntimeError(
                    f"No person detected in sampled frame {frame_index}; "
                    "cannot classify this video."
                )
            sequence.append(keypoints)

            last_frame = result.plot()
            progress = add_prediction_text(
                last_frame,
                f"Analyzing {video_path.name}: frame {position}/{SEQUENCE_LENGTH}",
                (0, 255, 255),
            )
            if display:
                cv2.imshow("Fitness AI - Video Test", progress)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Video test cancelled.")
                    return 1

        # Classify only after every sampled frame has a usable person detection.
        prediction, confidence = classify_sequence(
            np.stack(sequence), classifier, device
        )
        print(f"Video: {video_path}")
        print(f"Prediction: {prediction} ({confidence:.1%} confidence)")
        if video_path.parent.name in CLASS_NAMES:
            expected = video_path.parent.name
            print(f"Folder label: {expected} ({'match' if prediction == expected else 'does not match'})")

        final_frame = add_prediction_text(
            last_frame,
            f"Prediction: {prediction} ({confidence:.1%}) - press any key to close",
            (0, 255, 0),
        )
        if save_preview is not None:
            save_preview.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(save_preview), final_frame):
                raise RuntimeError(f"Could not save preview image: {save_preview}")
            print(f"Saved annotated preview: {save_preview}")

        if display:
            cv2.imshow("Fitness AI - Video Test", final_frame)
            cv2.waitKey(0)
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def process_camera(
    pose_model: YOLO,
    classifier: ExerciseClassifier,
    device: torch.device,
    device_name: str,
    camera_index: int,
) -> int:
    """Record a short camera sample, then classify it when SPACE is pressed again."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check camera access and permissions."
        )

    print("Stop with Q in the camera window or this terminal, or press Ctrl+C.")
    recording = False
    pose_frames: list[np.ndarray] = []
    last_frame = None
    status = "Press SPACE to start recording an exercise"
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            result = pose_model.predict(
                source=frame, device=device_name, verbose=False
            )[0]
            height, width = frame.shape[:2]
            keypoints = selected_person(result, width, height)
            annotated = result.plot()
            # Store only frames where the detector can see a person.
            if recording and keypoints is not None:
                pose_frames.append(keypoints)

            if recording:
                overlay_text = (
                    f"Recording: {len(pose_frames)} detected frames | "
                    "SPACE stop and classify | R reset | Q quit"
                )
                color = (0, 255, 255)
            else:
                overlay_text = f"{status} | Q quit"
                color = (255, 255, 255)

            if last_frame is not None and not recording:
                cv2.imshow(
                    "Fitness AI - Camera Test",
                    add_prediction_text(last_frame, status, (0, 255, 0)),
                )
            else:
                cv2.imshow(
                    "Fitness AI - Camera Test",
                    add_prediction_text(annotated, overlay_text, color),
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or console_quit_requested():
                return 0
            if key == ord("r"):
                recording = False
                pose_frames.clear()
                last_frame = None
                status = "Reset. Press SPACE to start recording an exercise"
            elif key == ord(" ") and not recording:
                recording = True
                pose_frames.clear()
                last_frame = None
                status = "Recording"
            elif key == ord(" ") and recording:
                recording = False
                if len(pose_frames) < SEQUENCE_LENGTH:
                    status = (
                        f"Only {len(pose_frames)} frames had a detected person; "
                        f"need {SEQUENCE_LENGTH}. Press SPACE to try again"
                    )
                    pose_frames.clear()
                    continue

                # Sample the recorded poses down to the classifier's fixed length.
                indices = np.linspace(
                    0, len(pose_frames) - 1,
                    num=SEQUENCE_LENGTH,
                    dtype=np.int64,
                )
                sequence = np.stack([pose_frames[int(i)] for i in indices])
                prediction, confidence = classify_sequence(
                    sequence, classifier, device
                )
                status = f"Prediction: {prediction} ({confidence:.1%} confidence)"
                print(status)
                last_frame = annotated.copy()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    """Load models and route the chosen source to camera or saved-video handling."""
    args = parse_args()
    source = args.source
    if source is None:
        source = input(
            "Enter a video file path or 'camera' to use the webcam: "
        ).strip()
    if not source:
        raise ValueError("A video path or camera source is required.")
    is_camera = source.lower() in {"camera", "webcam"} or source.isdigit()
    if args.headless and is_camera:
        raise ValueError("--headless can only be used with a video file.")

    device_name, device = resolve_device(args.device)
    classifier, pose_model = load_models(args, device_name, device)

    if is_camera:
        camera_index = int(source) if source.isdigit() else 0
        return process_camera(
            pose_model, classifier, device, device_name, camera_index
        )

    video_path = Path(source)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    return process_video(
        video_path,
        pose_model,
        classifier,
        device,
        device_name,
        args.save_preview,
        display=not args.headless,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCamera test stopped.")
