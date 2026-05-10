import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


DEFAULT_DATASET = "puneet6060/intel-image-classification"


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


def normalize_intel_structure(extract_dir: Path, output_dir: Path):
    train_src = next((p for p in [extract_dir / "seg_train", extract_dir / "train"] if p.exists()), None)
    test_src = next((p for p in [extract_dir / "seg_test", extract_dir / "test"] if p.exists()), None)

    if train_src is None:
        candidates = [p for p in extract_dir.rglob("*") if p.is_dir() and any(x.is_dir() for x in p.iterdir())]
        if not candidates:
            raise RuntimeError(f"Could not infer train directory from extracted data: {extract_dir}")
        train_src = candidates[0]

    train_src = _find_class_root(train_src)
    if test_src is not None:
        test_src = _find_class_root(test_src)

    train_out = output_dir / "train"
    test_out = output_dir / "test"
    copy_tree(train_src, train_out)
    if test_src is not None:
        copy_tree(test_src, test_out)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Kaggle image classification dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug.")
    parser.add_argument("--download_dir", default="data/downloads", help="Where to save downloaded zip.")
    parser.add_argument("--output_dir", default="data/raw", help="Prepared dataset directory.")
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

    run(["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(download_dir), "--force"])

    zip_files = sorted(download_dir.glob("*.zip"))
    if not zip_files:
        raise RuntimeError("No zip file downloaded from Kaggle.")
    zip_path = zip_files[-1]

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_path, extract_dir)
    normalize_intel_structure(extract_dir, output_dir)

    print(f"Dataset prepared successfully at: {output_dir}")
    print("Expected structure:")
    print(f"- {output_dir / 'train'}")
    print(f"- {output_dir / 'test'}")


if __name__ == "__main__":
    main()
