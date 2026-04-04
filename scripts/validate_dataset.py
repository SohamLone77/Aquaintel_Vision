"""Dataset integrity checks for paired underwater training data."""

import argparse
import os
from pathlib import Path

from PIL import Image


_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}



def _image_files(folder_path):
    if not folder_path.exists():
        return []
    files = []
    for item in folder_path.iterdir():
        if item.is_file() and item.suffix in _VALID_EXTS:
            files.append(item)
    return sorted(files)


def validate_dataset(data_path="data", strict_names=False, sample_check_count=25):
    """Validate paired dataset under data/raw and data/reference.

    Returns:
        tuple[bool, dict]: (is_valid, details)
    """
    root = Path(data_path)
    raw_dir = root / "raw"
    ref_dir = root / "reference"

    details = {
        "data_path": str(root),
        "raw_exists": raw_dir.exists(),
        "reference_exists": ref_dir.exists(),
        "raw_count": 0,
        "reference_count": 0,
        "paired_count": 0,
        "issues": [],
    }

    if not raw_dir.exists():
        details["issues"].append(f"Missing directory: {raw_dir}")
    if not ref_dir.exists():
        details["issues"].append(f"Missing directory: {ref_dir}")
    if details["issues"]:
        return False, details

    raw_files = _image_files(raw_dir)
    ref_files = _image_files(ref_dir)
    details["raw_count"] = len(raw_files)
    details["reference_count"] = len(ref_files)

    if not raw_files:
        details["issues"].append("No image files found in raw directory")
    if not ref_files:
        details["issues"].append("No image files found in reference directory")
    if details["issues"]:
        return False, details

    raw_map = {file_path.stem: file_path for file_path in raw_files}
    ref_map = {file_path.stem: file_path for file_path in ref_files}

    common_names = sorted(set(raw_map.keys()) & set(ref_map.keys()))
    details["paired_count"] = len(common_names)

    if strict_names:
        raw_only = sorted(set(raw_map.keys()) - set(ref_map.keys()))
        ref_only = sorted(set(ref_map.keys()) - set(raw_map.keys()))
        if raw_only:
            details["issues"].append(
                f"Raw images missing references: {len(raw_only)}"
            )
        if ref_only:
            details["issues"].append(
                f"Reference images missing raw counterparts: {len(ref_only)}"
            )
        if details["issues"]:
            return False, details
    else:
        if len(raw_files) != len(ref_files):
            details["issues"].append(
                f"Count mismatch: raw={len(raw_files)} reference={len(ref_files)}"
            )
            return False, details

    if details["paired_count"] == 0 and strict_names:
        details["issues"].append("No matching filename pairs found")
        return False, details

    # Readability check on a sample to fail fast on corrupted images.
    check_targets = []
    if common_names:
        for name in common_names[:sample_check_count]:
            check_targets.append(raw_map[name])
            check_targets.append(ref_map[name])
    else:
        max_pairs = min(len(raw_files), len(ref_files), sample_check_count)
        for index in range(max_pairs):
            check_targets.append(raw_files[index])
            check_targets.append(ref_files[index])

    for path_obj in check_targets:
        try:
            with Image.open(path_obj) as image:
                image.verify()
        except Exception:
            details["issues"].append(f"Unreadable image file: {path_obj}")
            return False, details

    return True, details


def _print_report(is_valid, details):
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Data path: {details['data_path']}")
    print(f"raw exists: {details['raw_exists']}")
    print(f"reference exists: {details['reference_exists']}")
    print(f"raw images: {details['raw_count']}")
    print(f"reference images: {details['reference_count']}")
    print(f"matched pairs: {details['paired_count']}")

    if is_valid:
        print("\nSTATUS: VALID")
        return

    print("\nSTATUS: INVALID")
    for issue in details["issues"]:
        print(f" - {issue}")


def main():
    parser = argparse.ArgumentParser(description="Validate training dataset integrity")
    parser.add_argument("--data-path", default="data", help="Dataset root path")
    parser.add_argument(
        "--strict-names",
        action="store_true",
        help="Require exact filename matching between raw and reference",
    )
    parser.add_argument(
        "--sample-check-count",
        type=int,
        default=25,
        help="Number of image pairs to read for corruption checks",
    )
    args = parser.parse_args()

    is_valid, details = validate_dataset(
        data_path=args.data_path,
        strict_names=args.strict_names,
        sample_check_count=args.sample_check_count,
    )
    _print_report(is_valid, details)

    if not is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
