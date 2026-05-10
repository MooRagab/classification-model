import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def load_class_names(class_names_path: str) -> List[str]:
    with open(class_names_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image: Image.Image, image_size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict_top_k(
    model: tf.keras.Model,
    image: Image.Image,
    class_names: List[str],
    top_k: int = 3,
    image_size=(224, 224),
) -> List[Tuple[str, float]]:
    input_tensor = preprocess_image(image, image_size=image_size)
    probs = model.predict(input_tensor, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Predict top-k classes from an input image.")
    parser.add_argument("--model_path", type=str, default="artifacts/model/best_model.keras")
    parser.add_argument("--class_names_path", type=str, default="artifacts/model/class_names.json")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = load_model(args.model_path)
    class_names = load_class_names(args.class_names_path)
    image = Image.open(image_path)
    results = predict_top_k(
        model=model,
        image=image,
        class_names=class_names,
        top_k=args.top_k,
        image_size=(args.image_size, args.image_size),
    )

    print("Top predictions:")
    for label, score in results:
        print(f"- {label}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()

