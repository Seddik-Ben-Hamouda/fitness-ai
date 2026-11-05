import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
CLASS_NAMES = ("push-up", "squat")
SEQUENCE_LENGTH = 30


def extract_sequence(
    video_path: Path,
    model: YOLO,
    device: str,
    sequence_length: int,
) -> tuple[np.ndarray | None, str | None]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None, "could not open video"

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < sequence_length or width <= 0 or height <= 0:
            return None, f"video has invalid dimensions or fewer than {sequence_length} frames"

        frame_indices = np.linspace(
            0, frame_count - 1, num=sequence_length, dtype=np.int64
        )
        sequence = []

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                return None, f"could not read frame {frame_index}"

            result = model.predict(source=frame, device=device, verbose=False)[0]
            if result.keypoints is None or result.boxes is None:
                return None, f"no person detected at frame {frame_index}"

            keypoints = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            if len(keypoints) == 0 or len(boxes) == 0:
                return None, f"no person detected at frame {frame_index}"

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            person_keypoints = keypoints[int(np.argmax(areas))].astype(np.float32)
            person_keypoints[:, 0] /= width
            person_keypoints[:, 1] /= height
            sequence.append(person_keypoints)

        return np.stack(sequence), None
    finally:
        capture.release()


def save_sequence(destination: Path, sequence: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.stem}.tmp.npy")
    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, sequence)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed-length YOLO pose sequences from exercise videos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Folder containing one subfolder per exercise class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "pose_sequences",
        help="Folder where extracted .npy sequences will be saved.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "yolo26m-pose.pt",
        help="Path to the YOLO pose model weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or a CUDA device number such as 0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO model weights not found: {args.model}")

    device = (
        "0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model = YOLO(str(args.model))
    print(f"Using device: {device}")
    print(f"Saving sequences under: {args.output_dir}")

    failures = []
    for class_name in CLASS_NAMES:
        class_dir = args.input_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Exercise folder not found: {class_dir}")

        videos = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        for video_path in videos:
            relative_path = video_path.relative_to(class_dir).with_suffix(".npy")
            destination = args.output_dir / class_name / relative_path
            if destination.is_file():
                print(f"Already extracted; skipping: {video_path}")
                continue

            sequence, error = extract_sequence(
                video_path, model, device, SEQUENCE_LENGTH
            )
            if error is not None:
                print(f"Skipped {video_path}: {error}")
                failures.append(video_path)
                continue

            assert sequence is not None
            save_sequence(destination, sequence)
            print(f"Saved {destination} with shape {sequence.shape}")

    if failures:
        print(f"Extraction finished with {len(failures)} skipped video(s).")
        return 1

    print("Extraction finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
