"""Train an experimental model that distinguishes squat from push-up videos."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep the labels and expected sequence format consistent with extraction/testing.
CLASS_NAMES = ("push-up", "squat")
SEQUENCE_SHAPE = (30, 17, 2)
# Training examples teach the model; validation guides training; test checks it.
SPLIT_RATIOS = (0.70, 0.15, 0.15)


class ExerciseClassifier(nn.Module):
    """Read a sequence of pose points and predict its exercise class."""

    def __init__(self, model_type: str, input_size: int, hidden_size: int, class_count: int):
        super().__init__()
        # Choose the requested recurrent layer; both return a final sequence state.
        recurrent = nn.LSTM if model_type == "lstm" else nn.GRU
        self.recurrent = recurrent(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, class_count),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return one score per exercise class for each input video sequence."""
        _, hidden = self.recurrent(sequence)
        final_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return self.classifier(final_hidden[-1])


def parse_args() -> argparse.Namespace:
    """Read paths and training settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate an LSTM or GRU on extracted pose sequences."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "pose_sequences",
        help="Folder containing one .npy subfolder per exercise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Folder for the trained model and evaluation reports.",
    )
    parser.add_argument(
        "--model",
        choices=("lstm", "gru"),
        default="lstm",
        help="Recurrent model architecture.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, or CUDA device number such as 0.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed random number generators so a run is easier to reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_class_files(
    files: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle one class and split its examples into train, validation, and test."""
    if len(files) < 3:
        raise ValueError(
            f"Need at least 3 sequences per class to create train/validation/test "
            f"splits; found {len(files)}."
        )

    # Round down first, then assign leftover examples to the largest remainders.
    counts = [int(len(files) * ratio) for ratio in SPLIT_RATIOS]
    remainders = [
        len(files) * ratio - count
        for ratio, count in zip(SPLIT_RATIOS, counts)
    ]
    for index in sorted(range(3), key=lambda i: remainders[i], reverse=True)[
        : len(files) - sum(counts)
    ]:
        counts[index] += 1

    if min(counts) == 0:
        raise ValueError(
            f"Class with {len(files)} sequences would have an empty split; "
            "collect more examples."
        )

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    train_end = counts[0]
    validation_end = train_end + counts[1]
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )


def load_data(data_dir: Path, seed: int):
    """Load both classes, validate each sequence, and return tensors plus file lists."""
    partitions: dict[str, list[tuple[Path, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    # Split each class separately so every partition contains both exercise labels.
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Sequence folder not found: {class_dir}")

        files = sorted(class_dir.rglob("*.npy"))
        if not files:
            raise ValueError(f"No .npy sequences found in {class_dir}")

        class_splits = split_class_files(files, seed + label)
        for split_name, split_files in zip(
            ("train", "validation", "test"), class_splits
        ):
            partitions[split_name].extend(
                (path, label) for path in split_files
            )

    result = {}
    manifest = {}
    for split_name, examples in partitions.items():
        random.Random(seed + len(split_name)).shuffle(examples)
        sequences = []
        labels = []
        paths = []
        for path, label in examples:
            # Reject malformed or unsafe inputs before they reach model training.
            sequence = np.load(path, allow_pickle=False)
            if sequence.shape != SEQUENCE_SHAPE:
                raise ValueError(
                    f"{path} has shape {sequence.shape}; expected {SEQUENCE_SHAPE}."
                )
            if not np.issubdtype(sequence.dtype, np.number) or not np.isfinite(
                sequence
            ).all():
                raise ValueError(f"{path} contains non-numeric or non-finite values.")

            # Flatten each of the 17 (x, y) points into 34 values per frame.
            sequences.append(sequence.reshape(SEQUENCE_SHAPE[0], -1))
            labels.append(label)
            paths.append(path.relative_to(data_dir).as_posix())

        result[split_name] = (
            torch.from_numpy(np.stack(sequences).astype(np.float32)),
            torch.tensor(labels, dtype=torch.long),
        )
        manifest[split_name] = paths

    return result, manifest


def evaluate(
    model: ExerciseClassifier,
    loss_function: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Measure loss and predicted labels without updating model weights."""
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        loss = float(loss_function(logits, labels.to(device)).item())
        predictions = logits.argmax(dim=1).cpu().numpy()
    return loss, predictions


def classification_metrics(
    truth: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    """Summarize test predictions, including per-class recall and accuracy."""
    class_count = len(CLASS_NAMES)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(truth, predictions):
        confusion[int(actual), int(predicted)] += 1

    per_class = {}
    recalls = []
    for index, class_name in enumerate(CLASS_NAMES):
        actual_count = int(confusion[index].sum())
        recall = (
            float(confusion[index, index] / actual_count) if actual_count else 0.0
        )
        recalls.append(recall)
        per_class[class_name] = {
            "support": actual_count,
            "recall": recall,
        }

    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def main() -> int:
    """Train, keep the best validation checkpoint, test it, and save reports."""
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")

    set_seed(args.seed)
    data, manifest = load_data(args.data_dir, args.seed)
    device_name = (
        "0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(f"cuda:{device_name}" if device_name.isdigit() else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment has no CUDA.")

    train_features, train_labels = data["train"]
    validation_features, validation_labels = data["validation"]
    test_features, test_labels = data["test"]
    # Weight classes inversely by their training frequency to reduce imbalance.
    class_counts = Counter(train_labels.tolist())
    weights = torch.tensor(
        [
            len(train_labels) / (len(CLASS_NAMES) * class_counts[index])
            for index in range(len(CLASS_NAMES))
        ],
        dtype=torch.float32,
        device=device,
    )

    model = ExerciseClassifier(
        args.model, input_size=34, hidden_size=64, class_count=len(CLASS_NAMES)
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Early stopping keeps the checkpoint with the best validation performance.
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        # Shuffle the training examples each epoch and learn in small batches.
        model.train()
        permutation = torch.randperm(len(train_labels))
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch_features = train_features[indices].to(device)
            batch_labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

        validation_loss, validation_predictions = evaluate(
            model,
            loss_function,
            validation_features,
            validation_labels,
            device,
        )
        validation_accuracy = float(
            np.mean(validation_predictions == validation_labels.numpy())
        )
        print(
            f"Epoch {epoch:03d}: validation loss={validation_loss:.4f}, "
            f"accuracy={validation_accuracy:.3f}"
        )

        # Keep a copy of the best model; stop if validation no longer improves.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    # The test set is held aside until the model has finished training.
    _, test_predictions = evaluate(
        model, loss_function, test_features, test_labels, device
    )
    metrics = classification_metrics(test_labels.numpy(), test_predictions)
    report = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "best_validation_loss": best_validation_loss,
        "class_names": list(CLASS_NAMES),
        "split_counts": {
            split: {
                "total": len(split_paths),
                "by_class": {
                    class_name: sum(
                        path.startswith(f"{class_name}/")
                        for path in split_paths
                    )
                    for class_name in CLASS_NAMES
                },
            }
            for split, split_paths in manifest.items()
        },
        "test": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "exercise_classifier.pt"
    report_path = args.output_dir / "exercise_classifier_report.json"
    manifest_path = args.output_dir / "exercise_classifier_splits.json"
    # Save the model, a human-readable score report, and exact file split lists.
    torch.save(
        {
            "model_type": args.model,
            "input_size": 34,
            "hidden_size": 64,
            "class_names": list(CLASS_NAMES),
            "state_dict": best_state,
        },
        model_path,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved data split: {manifest_path}")
    print("Test confusion matrix (rows=true, columns=predicted):")
    print(f"              {CLASS_NAMES[0]:>8} {CLASS_NAMES[1]:>8}")
    for class_name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
        print(f"{class_name:>14} {row[0]:8d} {row[1]:8d}")
    print(
        f"Test accuracy: {metrics['accuracy']:.3f}; "
        f"balanced accuracy: {metrics['balanced_accuracy']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
