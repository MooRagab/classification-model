import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DataConfig:
    data_dir: str
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    validation_split: float = 0.2


def _detect_dirs(data_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    train_candidates = [data_dir / "train", data_dir / "seg_train", data_dir / "Training"]
    test_candidates = [data_dir / "test", data_dir / "seg_test", data_dir / "Testing"]
    val_candidates = [data_dir / "val", data_dir / "valid", data_dir / "validation"]

    train_dir = next((p for p in train_candidates if p.exists()), None)
    test_dir = next((p for p in test_candidates if p.exists()), None)
    val_dir = next((p for p in val_candidates if p.exists()), None)

    if train_dir is None:
        if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()):
            train_dir = data_dir
        else:
            raise FileNotFoundError(f"Could not find train directory in: {data_dir}")
    return train_dir, val_dir, test_dir


def _resolve_nested_single_class_root(train_dir: Path) -> Path:
    """
    Handles accidental nesting such as:
    train/seg_train/<actual_classes>/...
    """
    subdirs = [p for p in train_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        return train_dir
    only = subdirs[0]
    nested = [p for p in only.iterdir() if p.is_dir()]
    if len(nested) >= 2:
        return only
    return train_dir


def _augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def _preprocess(ds: tf.data.Dataset, training: bool = False) -> tf.data.Dataset:
    aug = _augmenter()

    def _map(images, labels):
        images = tf.cast(images, tf.float32)
        if training:
            images = aug(images, training=True)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, labels

    return ds.map(_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def create_datasets(config: DataConfig):
    data_dir = Path(config.data_dir)
    train_dir, val_dir, test_dir = _detect_dirs(data_dir)
    train_dir = _resolve_nested_single_class_root(train_dir)

    if val_dir is not None:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=config.validation_split,
            subset="training",
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=config.validation_split,
            subset="validation",
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )

    class_names = train_ds.class_names
    train_ds = _preprocess(train_ds, training=True)
    val_ds = _preprocess(val_ds, training=False)

    test_ds = None
    if test_dir is not None and test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )
        test_ds = _preprocess(test_ds, training=False)

    return train_ds, val_ds, test_ds, class_names


def save_class_names(class_names, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
