import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from unet import (
    EXCEL_PATH,
    MODEL_PATH,
    THRESHOLD,
    FEATURE_COLUMNS,
    load_unet_model,
    extract_features,
)


SOURCE_IMAGE_DIR = "images_dataset"
OUTPUT_DIR = "machine_learning"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "hasil_pengukuran_unet.xlsx")
FEATURE_HEIGHT = 1440
FEATURE_WIDTH = 1080
SAVE_VISUALIZATION = True
VIS_OUTPUT_DIR = "hasil_batch"


def _read_original_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        img_fixed = ImageOps.exif_transpose(img)
        w, h = img_fixed.size
    return int(w), int(h)


def _normalize_target_excel(df_excel: pd.DataFrame) -> pd.DataFrame:
    # Samakan nama kolom agar merge stabil (mengikuti format file yang sudah ada di repo).
    rename_map = {}
    if "Nama File" in df_excel.columns:
        rename_map["Nama File"] = "nama_file"
    if "Massa" in df_excel.columns:
        rename_map["Massa"] = "massa"

    df = df_excel.rename(columns=rename_map).copy()
    if "nama_file" not in df.columns:
        raise ValueError("Kolom 'Nama File' tidak ditemukan di Excel target.")
    if "massa" not in df.columns:
        raise ValueError("Kolom 'Massa' tidak ditemukan di Excel target.")

    df["nama_file"] = df["nama_file"].astype(str)
    return df[["nama_file", "massa"]].dropna(subset=["nama_file"]).copy()


def main():
    model_unet, device = load_unet_model(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_VISUALIZATION:
        os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    image_dir = Path(SOURCE_IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {image_dir}")

    df_target_raw = pd.read_excel(EXCEL_PATH)
    df_target = _normalize_target_excel(df_target_raw)

    rows: list[dict] = []
    success = 0
    failed = 0

    for file in sorted(os.listdir(image_dir)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = str(image_dir / file)
        try:
            w, h = _read_original_size(image_path)
            feats = extract_features(
                image_path=image_path,
                model=model_unet,
                device=device,
                threshold=THRESHOLD,
                img_size=(FEATURE_HEIGHT, FEATURE_WIDTH),
                save_visualization=SAVE_VISUALIZATION,
                visualization_output_path=str(Path(VIS_OUTPUT_DIR) / f"overlay_{Path(file).stem}.jpg"),
                source_file_name=file,
            )

            rows.append(
                {
                    "nama_file": feats["nama_file"],
                    "luas_permukaan_px": feats["luas_permukaan_px"],
                    "diameter_diagonal_px": feats["diameter_diagonal_px"],
                    "diameter_tegak_lurus_px": feats["diameter_tegak_lurus_px"],
                    # Metadata opsional (tidak dipakai training, hanya audit/debug)
                    "Original Width": w,
                    "Original Height": h,
                    "Feature Space": f"{FEATURE_WIDTH}x{FEATURE_HEIGHT}",
                }
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"⚠️ Skip '{file}': {exc}")

    if success == 0:
        raise ValueError("Tidak ada gambar yang berhasil diekstrak.")

    df_features = pd.DataFrame(rows)
    df_merged = pd.merge(df_features, df_target, on="nama_file", how="left")

    # Output Excel dengan format yang tetap mudah dipakai ulang dan tidak mencampur fitur lama.
    # Kolom fitur inti harus tetap sama seperti FEATURE_COLUMNS (snake_case).
    core_cols = ["nama_file"] + FEATURE_COLUMNS + ["massa"]
    extra_cols = ["Original Width", "Original Height", "Feature Space"]
    out_cols = [c for c in core_cols if c in df_merged.columns] + [c for c in extra_cols if c in df_merged.columns]
    df_out = df_merged[out_cols].copy()

    df_out.to_excel(OUTPUT_EXCEL, index=False)
    print("\n✅ Batch processing selesai")
    print(f"   Berhasil: {success}")
    print(f"   Gagal   : {failed}")
    print(f"   Output  : {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()

