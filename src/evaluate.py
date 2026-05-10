import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.data import DataConfig, create_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model and create training plots.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset root directory.")
    parser.add_argument("--model_path", type=str, default="artifacts/model/best_model.keras")
    parser.add_argument("--history_path", type=str, default="artifacts/model/history.json")
    parser.add_argument("--plots_dir", type=str, default="artifacts/plots")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def plot_history(history_path: Path, plots_dir: Path):
    with history_path.open("r", encoding="utf-8") as f:
        hist = json.load(f)

    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(hist.get("accuracy", [])) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs, hist.get("val_accuracy", []), label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("loss", []), label="Train Loss")
    plt.plot(epochs, hist.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    out = plots_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved training curves to: {out}")


def evaluate(args):
    model = tf.keras.models.load_model(args.model_path)
    config = DataConfig(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    _, val_ds, test_ds, _ = create_datasets(config)

    val_metrics = model.evaluate(val_ds, verbose=0)
    print("Validation metrics:", dict(zip(model.metrics_names, val_metrics)))

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, verbose=0)
        print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))
    else:
        print("No test directory found; skipped test evaluation.")

    history_path = Path(args.history_path)
    if history_path.exists():
        plot_history(history_path, Path(args.plots_dir))
    else:
        print(f"History file not found at {history_path}. Skipping plots.")


if __name__ == "__main__":
    evaluate(parse_args())

