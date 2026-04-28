import os
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from unet import (
    IMGSZ,
    MODEL_PATH,
    PIXEL_TO_CM,
    THRESHOLD,
    extract_features,
    load_unet_model,
    predict_mass,
)


INPUT_DIR = "testing gambar 1"
OUTPUT_DIR = "output_testing_gambar_1"
ML_MODEL_PATH = Path("machine_learning") / "random_forest_massa.pkl"
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
ML_MODEL_NAME = "Random Forest (1080x1440 retrained)"


def format_feature_space(img_size) -> str:
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        return f"{int(img_size[1])}x{int(img_size[0])}"
    side = int(img_size)
    return f"{side}x{side}"


def save_detail_txt(output_txt_path: Path, data: dict):
    lines = [
        f"Nama File: {data['nama_file']}",
        f"Skala Fitur: {data['feature_space']}",
        f"Luas Permukaan (px): {data['luas_permukaan_px']:.4f}",
        f"Diameter Diagonal (px): {data['diameter_diagonal_px']:.4f}",
        f"Diameter Tegak Lurus (px): {data['diameter_tegak_lurus_px']:.4f}",
        f"Luas Permukaan (cm2): {data['luas_permukaan_cm2']:.4f}",
        f"Diameter Diagonal (cm): {data['diameter_diagonal_cm']:.4f}",
        f"Diameter Tegak Lurus (cm): {data['diameter_tegak_lurus_cm']:.4f}",
        f"cm per pixel: {data['cm_per_pixel']:.6f}",
        f"Model ML: {data['model_ml']}",
        f"Estimasi Massa TBS (kg): {data['estimasi_massa_tbs_kg']:.4f}",
        f"Output Segmentasi: {data['output_segmentasi']}",
    ]
    output_txt_path.write_text("\n".join(lines), encoding="utf-8")


def create_final_image_with_info(segmented_image_path: Path, final_image_path: Path, data: dict):
    seg_img = cv2.imread(str(segmented_image_path))
    if seg_img is None:
        raise FileNotFoundError(f"Gagal baca gambar segmentasi: {segmented_image_path}")

    h, w = seg_img.shape[:2]
    info_lines = [
        f"Nama File: {data['nama_file']}",
        f"Skala Fitur: {data['feature_space']} px",
        f"Luas Permukaan (px): {data['luas_permukaan_px']:.2f}",
        f"Diameter Diagonal (px): {data['diameter_diagonal_px']:.2f}",
        f"Diameter Tegak Lurus (px): {data['diameter_tegak_lurus_px']:.2f}",
        f"Luas Permukaan (cm2): {data['luas_permukaan_cm2']:.2f}",
        f"Diameter Diagonal (cm): {data['diameter_diagonal_cm']:.2f}",
        f"Diameter Tegak Lurus (cm): {data['diameter_tegak_lurus_cm']:.2f}",
        f"CM Per Pixel: {data['cm_per_pixel']:.4f}",
        f"Model ML: {data['model_ml']}",
        f"Estimasi Massa TBS (kg): {data['estimasi_massa_tbs_kg']:.2f}",
    ]

    panel_h = 30 + (len(info_lines) * 32)
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, panel_h - 1), (255, 255, 255), 2)

    y = 34
    for idx, line in enumerate(info_lines):
        color = (240, 240, 240)
        if idx == len(info_lines) - 1:
            color = (0, 215, 255)
        cv2.putText(
            panel,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30

    combined = np.vstack([panel, seg_img])
    cv2.imwrite(str(final_image_path), combined)


def main():
    parser = argparse.ArgumentParser(description="Batch testing gambar + output final 1 gambar dengan panel info.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Folder input gambar.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Folder output hasil testing.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    seg_dir = output_dir / "segmentasi"
    final_dir = output_dir / "final"
    detail_dir = output_dir / "detail"
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")
    if not ML_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model ML tidak ditemukan: {ML_MODEL_PATH}")

    model_unet, device = load_unet_model(MODEL_PATH)
    model_ml = joblib.load(str(ML_MODEL_PATH))

    rows = []
    success = 0
    failed = 0

    for file_name in sorted(os.listdir(input_dir)):
        if not file_name.lower().endswith(ALLOWED_EXTS):
            continue

        image_path = input_dir / file_name
        seg_output_path = seg_dir / f"segmentasi_{Path(file_name).stem}.jpg"
        final_output_path = final_dir / f"final_{Path(file_name).stem}.jpg"
        txt_output_path = detail_dir / f"detail_{Path(file_name).stem}.txt"

        try:
            feats = extract_features(
                image_path=str(image_path),
                model=model_unet,
                device=device,
                source_file_name=file_name,
                threshold=THRESHOLD,
                img_size=IMGSZ,
                save_visualization=True,
                visualization_output_path=str(seg_output_path),
            )

            estimasi_kg = predict_mass(
                model_ml,
                luas_permukaan_px=feats["luas_permukaan_px"],
                diameter_diagonal_px=feats["diameter_diagonal_px"],
                diameter_tegak_lurus_px=feats["diameter_tegak_lurus_px"],
            )

            luas_cm2 = feats["luas_permukaan_px"] * (PIXEL_TO_CM ** 2)
            diagonal_cm = feats["diameter_diagonal_px"] * PIXEL_TO_CM
            tegak_cm = feats["diameter_tegak_lurus_px"] * PIXEL_TO_CM

            row = {
                "nama_file": file_name,
                "feature_space": format_feature_space(IMGSZ),
                "luas_permukaan_px": float(feats["luas_permukaan_px"]),
                "diameter_diagonal_px": float(feats["diameter_diagonal_px"]),
                "diameter_tegak_lurus_px": float(feats["diameter_tegak_lurus_px"]),
                "luas_permukaan_cm2": float(luas_cm2),
                "diameter_diagonal_cm": float(diagonal_cm),
                "diameter_tegak_lurus_cm": float(tegak_cm),
                "cm_per_pixel": float(PIXEL_TO_CM),
                "model_ml": ML_MODEL_NAME,
                "estimasi_massa_tbs_kg": float(estimasi_kg),
                "output_segmentasi": str(seg_output_path),
                "output_final": str(final_output_path),
            }
            create_final_image_with_info(seg_output_path, final_output_path, row)
            rows.append(row)
            save_detail_txt(txt_output_path, row)
            success += 1
            print(f"OK: {file_name} -> {final_output_path.name}")
        except Exception as exc:
            failed += 1
            print(f"SKIP: {file_name} ({exc})")

    if not rows:
        raise ValueError("Tidak ada gambar yang berhasil diproses.")

    summary_path = output_dir / "ringkasan_hasil.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("\nSelesai testing otomatis")
    print(f"Berhasil: {success}")
    print(f"Gagal   : {failed}")
    print(f"Folder output: {output_dir}")
    print(f"Ringkasan CSV: {summary_path}")


if __name__ == "__main__":
    main()

