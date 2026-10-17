#!/usr/bin/env python3
"""Compare FWHT block sizes and redundancy against PSNR, SSIM, and attacks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_watermark_robustness import (
    build_attacks,
    compute_psnr,
    list_images,
)
from fwht import embed_logo_watermark, extract_logo_watermark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/test/images"))
    parser.add_argument("--logo-name", default="Blason_univ_Yaoundé_1.png")
    parser.add_argument("--block-sizes", default="4,8")
    parser.add_argument("--redundancies", default="1,3")
    parser.add_argument(
        "--attacks",
        default="clean,crop_90,crop_75,crop_60,rotate_2,rotate_5,rotate_15,rotate_45,rotate_90,translate_10_05,resize_75,resize_50,zoom_125,zoom_150,brightness_080,brightness_120,contrast_080,contrast_120,saturation_050,saturation_150,grayscale,gamma_080,gamma_120,noise_sigma_3,noise_sigma_8,blur_1_0,blur_2_0,sharpen",
        help="Comma-separated attacks from the robustness benchmark.",
    )
    parser.add_argument("--target-psnr", type=float, default=65.0)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fwht_calibration.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_images(args.data_dir, args.max_images)
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans {args.data_dir}")
    attacks = build_attacks()
    attack_names = [name.strip() for name in args.attacks.split(",") if name.strip()]
    unknown_attacks = sorted(set(attack_names) - set(attacks))
    if unknown_attacks:
        raise ValueError(f"Attaques inconnues : {', '.join(unknown_attacks)}")
    selected_attacks = {name: attacks[name] for name in attack_names}
    combinations = []
    for block_size in map(int, args.block_sizes.split(",")):
        for redundancy in map(int, args.redundancies.split(",")):
            rows = []
            for image_path in images:
                original = image_path.read_bytes()
                try:
                    watermarked, embed = embed_logo_watermark(
                        original,
                        logo_filename=args.logo_name,
                        alpha=args.alpha,
                        block_size=block_size,
                        redundancy=redundancy,
                        target_psnr_db=args.target_psnr,
                    )
                    for attack_name, attack_fn in selected_attacks.items():
                        attacked = attack_fn(watermarked)
                        _, extracted = extract_logo_watermark(
                            attacked,
                            original,
                            logo_filename=args.logo_name,
                            alpha=embed["alpha"],
                            block_size=block_size,
                            redundancy=redundancy,
                        )
                        rows.append({
                            "image": image_path.relative_to(args.data_dir).as_posix(),
                            "attack": attack_name,
                            "embed_psnr_db": embed["psnr_db"],
                            "embed_ssim": embed["ssim"],
                            "embed_alpha": embed["alpha"],
                            "attack_psnr_db": round(compute_psnr(watermarked, attacked), 6),
                            "bit_accuracy": extracted["bit_accuracy"],
                            "correlation": extracted["correlation"],
                        })
                except Exception as exc:
                    rows.append({"image": image_path.relative_to(args.data_dir).as_posix(), "error": str(exc)})
            valid = [row for row in rows if "error" not in row]
            combinations.append({
                "block_size": block_size,
                "redundancy": redundancy,
                "valid_records": len(valid),
                "errors": len(rows) - len(valid),
                "mean_embed_psnr_db": round(sum(r["embed_psnr_db"] for r in valid) / len(valid), 4) if valid else None,
                "mean_embed_ssim": round(sum(r["embed_ssim"] for r in valid) / len(valid), 6) if valid else None,
                "mean_bit_accuracy_by_attack": {
                    name: round(sum(r["bit_accuracy"] for r in valid if r["attack"] == name) /
                                max(1, sum(1 for r in valid if r["attack"] == name)), 6)
                    for name in selected_attacks
                },
                "records": rows,
            })
    report = {
        "data_dir": str(args.data_dir),
        "images": [p.relative_to(args.data_dir).as_posix() for p in images],
        "target_psnr_db": args.target_psnr,
        "combinations": combinations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "results": [
            {k: combination[k] for k in (
                "block_size", "redundancy", "mean_embed_psnr_db", "mean_embed_ssim",
                "mean_bit_accuracy_by_attack", "errors",
            )}
            for combination in combinations
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
