"""Download and extract UIEB dataset into a deterministic local structure."""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests

try:
    from scripts.validate_dataset import validate_dataset
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.validate_dataset import validate_dataset


UIEB_URL = (
    "https://irvlab.cs.umn.edu/sites/default/files/"
    "Underwater%20Image%20Enhancement%20Benchmark.zip"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def _iter_image_files(folder):
    return [
        path_obj for path_obj in folder.iterdir()
        if path_obj.is_file() and path_obj.suffix in IMAGE_EXTENSIONS
    ]


def _find_candidate_dirs(extract_root):
    candidates = []
    for path_obj in extract_root.rglob("*"):
        if not path_obj.is_dir():
            continue
        image_count = len(_iter_image_files(path_obj))
        if image_count > 0:
            candidates.append((path_obj, image_count))
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def _pick_raw_and_reference_dirs(candidates):
    if not candidates:
        return None, None

    raw_keywords = ("raw", "input", "underwater")
    ref_keywords = ("reference", "ref", "gt", "ground", "target")

    raw_dir = None
    ref_dir = None

    for path_obj, _count in candidates:
        lower = path_obj.as_posix().lower()
        if raw_dir is None and any(keyword in lower for keyword in raw_keywords):
            raw_dir = path_obj
        if ref_dir is None and any(keyword in lower for keyword in ref_keywords):
            ref_dir = path_obj

    if raw_dir and ref_dir:
        return raw_dir, ref_dir

    # Fallback: use top two image-bearing directories.
    top = [item[0] for item in candidates[:2]]
    if len(top) < 2:
        return None, None
    return top[0], top[1]


def _copy_images(src_dir, dst_dir, clear=False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for existing in dst_dir.iterdir():
            if existing.is_file():
                existing.unlink()

    copied = 0
    for image_path in _iter_image_files(src_dir):
        shutil.copy2(image_path, dst_dir / image_path.name)
        copied += 1
    return copied


def download_uieb(data_dir="data", zip_name="uieb.zip", timeout=120):
    """Download UIEB zip archive into the data directory."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / zip_name

    print("Downloading UIEB dataset...")
    print(f"URL: {UIEB_URL}")

    response = requests.get(UIEB_URL, stream=True, timeout=timeout)
    response.raise_for_status()

    with open(zip_path, "wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)

    print(f"Download complete: {zip_path} ({zip_path.stat().st_size} bytes)")
    return zip_path


def extract_dataset(data_dir="data", zip_name="uieb.zip", overwrite=False):
    """Extract and normalize dataset into data/raw and data/reference."""
    data_path = Path(data_dir)
    zip_path = data_path / zip_name
    extract_root = data_path / "_uieb_extracted"

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    if overwrite and extract_root.exists():
        shutil.rmtree(extract_root)

    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting archive to: {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_root)

    candidates = _find_candidate_dirs(extract_root)
    raw_source, ref_source = _pick_raw_and_reference_dirs(candidates)
    if raw_source is None or ref_source is None:
        raise RuntimeError(
            "Could not identify raw/reference folders in extracted archive. "
            "Inspect data/_uieb_extracted manually."
        )

    raw_target = data_path / "raw"
    ref_target = data_path / "reference"

    copied_raw = _copy_images(raw_source, raw_target, clear=overwrite)
    copied_ref = _copy_images(ref_source, ref_target, clear=overwrite)
    print(f"Copied raw images: {copied_raw} from {raw_source}")
    print(f"Copied reference images: {copied_ref} from {ref_source}")

    is_valid, details = validate_dataset(data_path=str(data_path))
    if not is_valid:
        issues = "; ".join(details["issues"])
        raise RuntimeError(f"Post-extract validation failed: {issues}")

    print(
        f"Dataset ready with {details['raw_count']} raw and "
        f"{details['reference_count']} reference images"
    )
    return details


def main():
    parser = argparse.ArgumentParser(description="Download and extract UIEB dataset")
    parser.add_argument("--data-dir", default="data", help="Target data directory")
    parser.add_argument("--zip-name", default="uieb.zip", help="Downloaded zip filename")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip download and only extract existing archive",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing extracted/temp files before extraction",
    )
    args = parser.parse_args()

    if not args.extract_only:
        download_uieb(data_dir=args.data_dir, zip_name=args.zip_name)

    extract_dataset(
        data_dir=args.data_dir,
        zip_name=args.zip_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()