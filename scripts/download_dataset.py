import argparse
import shutil
import subprocess
import time
import zipfile
from pathlib import Path


DEFAULT_DATASET = "elhamafify/caltech101"
DEFAULT_CLASS_ROOT = "101_ObjectCategories"
BACKGROUND_CLASS = "BACKGROUND_Google"


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def ensure_kaggle_available():
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True, text=True)
    except Exception as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure kaggle.json."
        ) from exc


def extract_zip(zip_path: Path, target_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for class_dir in src.iterdir():
        if class_dir.is_dir():
            shutil.copytree(class_dir, dst / class_dir.name, dirs_exist_ok=True)


def _looks_like_class_dir(root: Path) -> bool:
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return False
    return all(any(f.is_file() for f in d.iterdir()) for d in subdirs)


def _find_class_root(start: Path) -> Path:
    if not start.exists():
        raise RuntimeError(f"Directory does not exist: {start}")
    if _looks_like_class_dir(start):
        return start
    for candidate in sorted([p for p in start.rglob("*") if p.is_dir()]):
        try:
            if _looks_like_class_dir(candidate):
                return candidate
        except PermissionError:
            continue
    raise RuntimeError(f"Could not find class root under: {start}")


def _iter_class_dirs(class_root: Path, exclude_background: bool):
    for class_dir in sorted(class_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if exclude_background and class_dir.name == BACKGROUND_CLASS:
            continue
        yield class_dir


def _find_preferred_class_root(extract_dir: Path, class_root_name: str) -> Path:
    direct = extract_dir / class_root_name
    if direct.exists() and direct.is_dir():
        return direct

    matches = [p for p in extract_dir.rglob(class_root_name) if p.is_dir()]
    if matches:
        return matches[0]

    return _find_class_root(extract_dir)


def normalize_dataset_structure(
    extract_dir: Path,
    output_dir: Path,
    class_root_name: str = DEFAULT_CLASS_ROOT,
    exclude_background: bool = True,
):
    class_root = _find_preferred_class_root(extract_dir, class_root_name)
    train_out = output_dir / "train"
    train_out.mkdir(parents=True, exist_ok=True)

    copied_classes = 0
    for class_dir in _iter_class_dirs(class_root, exclude_background=exclude_background):
        shutil.copytree(class_dir, train_out / class_dir.name, dirs_exist_ok=True)
        copied_classes += 1

    if copied_classes == 0:
        raise RuntimeError(f"No class directories were copied from: {class_root}")

    return copied_classes


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Kaggle image classification dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug.")
    parser.add_argument("--download_dir", default="data/downloads", help="Where to save downloaded zip.")
    parser.add_argument("--output_dir", default="data/raw", help="Prepared dataset directory.")
    parser.add_argument(
        "--class_root",
        default=DEFAULT_CLASS_ROOT,
        help="Preferred directory name that contains class folders.",
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        help=f"Include `{BACKGROUND_CLASS}` as a normal class if it exists.",
    )
    parser.add_argument("--force", action="store_true", help="Delete output directory before preparing.")
    args = parser.parse_args()

    ensure_kaggle_available()

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    extract_dir = download_dir / "extracted"
    download_dir.mkdir(parents=True, exist_ok=True)

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    run(["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(download_dir), "--force"])

    zip_files = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise RuntimeError("No zip file downloaded from Kaggle.")

    # Prefer zip file updated by the current download command.
    fresh_zip = next((p for p in zip_files if p.stat().st_mtime >= started_at - 2), None)
    zip_path = fresh_zip if fresh_zip is not None else zip_files[0]

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_path, extract_dir)
    copied_classes = normalize_dataset_structure(
        extract_dir=extract_dir,
        output_dir=output_dir,
        class_root_name=args.class_root,
        exclude_background=not args.include_background,
    )

    print(f"Dataset prepared successfully at: {output_dir}")
    print(f"Classes copied to train split: {copied_classes}")
    print("Prepared structure:")
    print(f"- {output_dir / 'train'}")
    print(f"- {output_dir / 'test'} (optional; not created unless dataset provides one)")


if __name__ == "__main__":
    main()
