import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data import DataConfig, create_datasets, save_class_names
from src.model import build_model, unfreeze_top_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 transfer learning model.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--image_size", type=int, default=224, help="Image width/height.")
    parser.add_argument("--initial_lr", type=float, default=1e-3, help="Learning rate for phase 1.")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for phase 2.")
    parser.add_argument("--fine_tune_at", type=int, default=100, help="Freeze layers before this index.")
    parser.add_argument("--phase1_ratio", type=float, default=0.6, help="Fraction of epochs for phase 1.")
    parser.add_argument("--model_dir", type=str, default="artifacts/model", help="Where to save model artifacts.")
    return parser.parse_args()


def _callbacks(model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]


def _merge_histories(hist1, hist2):
    merged = {}
    for key in set(hist1.history.keys()).union(hist2.history.keys()):
        merged[key] = hist1.history.get(key, []) + hist2.history.get(key, [])
    return merged


def train_model(args):
    config = DataConfig(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    train_ds, val_ds, test_ds, class_names = create_datasets(config)
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, image_size=config.image_size, trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.initial_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase1_epochs = max(1, int(args.epochs * args.phase1_ratio))
    phase2_epochs = max(0, args.epochs - phase1_epochs)

    callbacks = _callbacks(Path(args.model_dir))
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=callbacks,
    )

    if phase2_epochs > 0:
        model = unfreeze_top_layers(model, fine_tune_at=args.fine_tune_at)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=phase1_epochs,
            epochs=args.epochs,
            callbacks=callbacks,
        )
        history_data = _merge_histories(history1, history2)
    else:
        history_data = history1.history

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "last_model.keras")
    save_class_names(class_names, str(model_dir / "class_names.json"))

    with (model_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, verbose=0)
        metric_names = model.metrics_names
        print("Test metrics:", dict(zip(metric_names, test_metrics)))

    print(f"Training complete. Best model at: {model_dir / 'best_model.keras'}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)

