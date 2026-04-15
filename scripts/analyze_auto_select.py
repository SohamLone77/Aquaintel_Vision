#!/usr/bin/env python3
"""Compare auto-select outputs from PSNR and SSIM runs.

Usage:
    python scripts/analyze_auto_select.py
    python scripts/analyze_auto_select.py --psnr-csv path --ssim-csv path
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

METHOD_SCORE_COLUMNS = {
    "model": "model_score",
    "unsharp": "unsharp_score",
    "kernel": "kernel_score",
    "adaptive": "adaptive_score",
}


def load_csv(path: Path) -> Dict[str, dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no rows: {path}")

    by_image = {}
    for row in rows:
        image = row.get("image", "").strip()
        if not image:
            continue
        by_image[image] = row

    if not by_image:
        raise ValueError(f"No valid image rows found: {path}")

    return by_image


def method_win_rates(rows_by_image: Dict[str, dict]) -> Dict[str, float]:
    wins = Counter(row["best_method"] for row in rows_by_image.values())
    total = len(rows_by_image)
    return {method: (wins[method] / total) * 100.0 for method in METHOD_SCORE_COLUMNS}


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_selectors(psnr_rows: Dict[str, dict], ssim_rows: Dict[str, dict]) -> dict:
    common_images = sorted(set(psnr_rows.keys()) & set(ssim_rows.keys()))
    if not common_images:
        raise ValueError("No overlapping images between PSNR and SSIM CSV files")

    method_match_count = 0

    psnr_advantage_gaps = []
    ssim_advantage_gaps = []
    disagreements = []

    for image in common_images:
        p_row = psnr_rows[image]
        s_row = ssim_rows[image]

        p_method = p_row["best_method"]
        s_method = s_row["best_method"]

        if p_method == s_method:
            method_match_count += 1

        p_method_col = METHOD_SCORE_COLUMNS[p_method]
        s_method_col = METHOD_SCORE_COLUMNS[s_method]

        p_score_for_p_method = float(p_row[p_method_col])
        p_score_for_s_method = float(p_row[s_method_col])

        s_score_for_s_method = float(s_row[s_method_col])
        s_score_for_p_method = float(s_row[p_method_col])

        # Positive means the metric prefers its own selected method.
        psnr_gap = p_score_for_p_method - p_score_for_s_method
        ssim_gap = s_score_for_s_method - s_score_for_p_method
        psnr_advantage_gaps.append(psnr_gap)
        ssim_advantage_gaps.append(ssim_gap)

        if p_method != s_method:
            disagreements.append(
                {
                    "image": image,
                    "psnr_best_method": p_method,
                    "ssim_best_method": s_method,
                    "psnr_best_score": p_row["best_score"],
                    "ssim_best_score": s_row["best_score"],
                    "psnr_score_for_psnr_method": p_row[p_method_col],
                    "psnr_score_for_ssim_method": p_row[s_method_col],
                    "ssim_score_for_psnr_method": s_row[p_method_col],
                    "ssim_score_for_ssim_method": s_row[s_method_col],
                    "psnr_advantage_gap": f"{psnr_gap:.6f}",
                    "ssim_advantage_gap": f"{ssim_gap:.6f}",
                }
            )

    return {
        "common_images": len(common_images),
        "selector_agreement_rate": (method_match_count / len(common_images)) * 100.0,
        "avg_psnr_advantage_gap": mean(psnr_advantage_gaps),
        "avg_ssim_advantage_gap": mean(ssim_advantage_gaps),
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
    }


def write_disagreements_csv(path: Path, disagreements: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "psnr_best_method",
        "ssim_best_method",
        "psnr_best_score",
        "ssim_best_score",
        "psnr_score_for_psnr_method",
        "psnr_score_for_ssim_method",
        "ssim_score_for_psnr_method",
        "ssim_score_for_ssim_method",
        "psnr_advantage_gap",
        "ssim_advantage_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)


def print_report(psnr_path: Path, ssim_path: Path, disagreements_csv: Path) -> None:
    psnr_rows = load_csv(psnr_path)
    ssim_rows = load_csv(ssim_path)

    psnr_win_rates = method_win_rates(psnr_rows)
    ssim_win_rates = method_win_rates(ssim_rows)
    comparison = compare_selectors(psnr_rows, ssim_rows)
    write_disagreements_csv(disagreements_csv, comparison["disagreements"])

    print("=== Auto-Select Comparison Report ===")
    print(f"PSNR CSV: {psnr_path}")
    print(f"SSIM CSV: {ssim_path}")
    print()

    print("Method win-rates (PSNR selector)")
    for method, rate in psnr_win_rates.items():
        print(f"  - {method:8s}: {rate:6.2f}%")
    print()

    print("Method win-rates (SSIM selector)")
    for method, rate in ssim_win_rates.items():
        print(f"  - {method:8s}: {rate:6.2f}%")
    print()

    print("Cross-metric selector comparison")
    print(f"  - Common images: {comparison['common_images']}")
    print(f"  - Selector agreement rate: {comparison['selector_agreement_rate']:.2f}%")
    print(f"  - Disagreement count: {comparison['disagreement_count']}")
    print(
        "  - Avg PSNR advantage gap "
        f"(PSNR-selected score - SSIM-selected score in PSNR CSV): "
        f"{comparison['avg_psnr_advantage_gap']:.6f}"
    )
    print(
        "  - Avg SSIM advantage gap "
        f"(SSIM-selected score - PSNR-selected score in SSIM CSV): "
        f"{comparison['avg_ssim_advantage_gap']:.6f}"
    )
    print(f"  - Disagreement CSV: {disagreements_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PSNR and SSIM auto-select CSVs")
    parser.add_argument(
        "--psnr-csv",
        type=Path,
        default=Path("results/sharp_outputs/auto_select_psnr.csv"),
        help="Path to PSNR auto-select CSV",
    )
    parser.add_argument(
        "--ssim-csv",
        type=Path,
        default=Path("results/sharp_outputs/auto_select_ssim.csv"),
        help="Path to SSIM auto-select CSV",
    )
    parser.add_argument(
        "--disagreements-csv",
        type=Path,
        default=Path("results/sharp_outputs/auto_select_disagreements.csv"),
        help="Path for disagreement-only output CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_report(args.psnr_csv, args.ssim_csv, args.disagreements_csv)


if __name__ == "__main__":
    main()
