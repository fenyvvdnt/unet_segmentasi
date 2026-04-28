import os
import re
import sys
from pathlib import Path

import cv2
import gdown
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib  # ✅ TAMBAHAN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =========================
# KONFIGURASI
# =========================
IMAGE_PATH = "test_dawas/20260406_135444.jpg"
MODEL_PATH = "unet_best.pth"
THRESHOLD = 0.85
# Feature space default untuk seluruh pipeline (height, width) = 1440x1080 (rasio 3:4).
IMGSZ = (1440, 1080)
PIXEL_TO_CM = 0.05

FEATURES_CSV_PATH = "features.csv"
EXCEL_PATH = "hasil_pengukuran_unet.xlsx"
FEATURE_COLUMNS = ["luas_permukaan_px", "diameter_diagonal_px", "diameter_tegak_lurus_px"]
TARGET_COLUMN = "Massa"
RANDOM_STATE = 42
TEST_SIZE = 13 / 93
VALIDATION_OUTPUT_PATH = "validasi_prediksi_per_file.csv"


def _extract_google_drive_file_id(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", value):
        return value
    return None


def ensure_model_file(
    model_path: str,
    google_drive_url: str | None = None,
    google_drive_file_id: str | None = None,
):
    model_file = Path(model_path)
    if model_file.exists():
        return

    file_id = _extract_google_drive_file_id(google_drive_file_id) or _extract_google_drive_file_id(google_drive_url)
    if not file_id:
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{model_path}'. "
            "Sediakan GOOGLE_DRIVE_MODEL_URL atau GOOGLE_DRIVE_MODEL_FILE_ID."
        )

    model_file.parent.mkdir(parents=True, exist_ok=True)
    download_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Mengunduh model dari Google Drive ke '{model_path}'...")
    gdown.download(download_url, str(model_file), quiet=False, fuzzy=True)

    if not model_file.exists():
        raise FileNotFoundError(f"Gagal mengunduh model ke '{model_path}'.")


def load_unet_model(model_path: str):
    print("⏳ Memuat model U-Net (resnext50_32x4d)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name="resnext50_32x4d",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Bobot model '{model_path}' berhasil dimuat!")

    model.to(device)
    model.eval()
    return model, device


# =========================
# (extract_features tetap — tidak diubah)
# =========================


def extract_features(
    image_path: str,
    model,
    device: str,
    source_file_name: str = None,
    threshold: float = THRESHOLD,
    pixel_to_cm: float = PIXEL_TO_CM,
    img_size: int | tuple[int, int] = IMGSZ,
    save_visualization: bool = True,
    visualization_output_path: str = "step4_final_diameter_lurus_unet.jpg",
):
    try:
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = ImageOps.exif_transpose(pil_image)
        image_rgb = np.array(pil_image)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise FileNotFoundError(f"Gambar tidak dapat dibaca: '{image_path}' ({exc})")

    original_h, original_w = image_rgb.shape[:2]

    if isinstance(img_size, (tuple, list)):
        if len(img_size) != 2:
            raise ValueError("img_size tuple harus berisi (height, width).")
        feature_h, feature_w = int(img_size[0]), int(img_size[1])
    else:
        feature_h, feature_w = int(img_size), int(img_size)

    if feature_h <= 0 or feature_w <= 0:
        raise ValueError("img_size harus > 0.")

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((feature_h, feature_w)),
            T.ToTensor(),
        ]
    )

    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Fitur selalu dihitung pada cleaned_mask di feature space tetap
    # agar konsisten lintas resolusi kamera.
    pred_mask_feature = (pred > threshold).astype(np.uint8)
    mask_feature_u8 = (pred_mask_feature * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_feature_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Objek tidak terdeteksi pada mask segmentasi.")

    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_mask_u8 = np.zeros_like(mask_feature_u8)
    cv2.drawContours(cleaned_mask_u8, [largest_contour], -1, 255, thickness=cv2.FILLED)

    area_px = cv2.contourArea(largest_contour)
    if area_px <= 0:
        raise ValueError("Luas objek terdeteksi tidak valid.")

    ys, xs = np.where(cleaned_mask_u8 > 0)
    points = np.column_stack((xs, ys))

    if len(points) < 50:
        raise ValueError("Objek terlalu kecil.")

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sumbu utama (diagonal) = eigenvalue terbesar, sumbu tegak lurus = terkecil.
    major_idx = np.argmax(eigenvalues)
    minor_idx = np.argmin(eigenvalues)
    major_direction = eigenvectors[:, major_idx]
    minor_direction = eigenvectors[:, minor_idx]

    projections_major = centered @ major_direction
    projections_minor = centered @ minor_direction

    major_min = projections_major.min()
    major_max = projections_major.max()
    minor_min = projections_minor.min()
    minor_max = projections_minor.max()

    diameter_diagonal_px = major_max - major_min
    diameter_tegak_lurus_px = minor_max - minor_min
    luas_permukaan_px = area_px
    diameter_rata2_px = (diameter_diagonal_px + diameter_tegak_lurus_px) / 2.0
    volume_approx = (4 / 3) * np.pi * (diameter_rata2_px / 2) ** 3

    if save_visualization:
        visualization = image.copy()
        mask_original_u8 = cv2.resize(cleaned_mask_u8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # Overlay mask terisi agar area segmentasi terlihat jelas.
        overlay = visualization.copy()
        mask_bool = mask_original_u8 > 0
        overlay[mask_bool] = (0, 255, 0)
        alpha = 0.45
        visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)

        # Outline tetap dipertahankan sebagai batas objek.
        contours_original, _ = cv2.findContours(mask_original_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_original:
            largest_contour_original = max(contours_original, key=cv2.contourArea)
            cv2.drawContours(visualization, [largest_contour_original], -1, (0, 200, 0), 2)

        p_major_1f = centroid + (major_min * major_direction)
        p_major_2f = centroid + (major_max * major_direction)
        p_minor_1f = centroid + (minor_min * minor_direction)
        p_minor_2f = centroid + (minor_max * minor_direction)
        scale_x = original_w / float(feature_w)
        scale_y = original_h / float(feature_h)

        p_major_1 = (int(round(p_major_1f[0] * scale_x)), int(round(p_major_1f[1] * scale_y)))
        p_major_2 = (int(round(p_major_2f[0] * scale_x)), int(round(p_major_2f[1] * scale_y)))
        p_minor_1 = (int(round(p_minor_1f[0] * scale_x)), int(round(p_minor_1f[1] * scale_y)))
        p_minor_2 = (int(round(p_minor_2f[0] * scale_x)), int(round(p_minor_2f[1] * scale_y)))

        cv2.line(visualization, p_major_1, p_major_2, (0, 0, 255), 5)
        cv2.line(visualization, p_minor_1, p_minor_2, (255, 255, 0), 5)
        cv2.circle(visualization, p_major_1, 5, (255, 0, 0), -1)
        cv2.circle(visualization, p_major_2, 5, (255, 0, 0), -1)
        cv2.circle(visualization, p_minor_1, 5, (0, 165, 255), -1)
        cv2.circle(visualization, p_minor_2, 5, (0, 165, 255), -1)

        cv2.putText(
            visualization,
            f"diag: {diameter_diagonal_px:.2f}px | tegak lurus: {diameter_tegak_lurus_px:.2f}px",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(visualization_output_path, visualization)

    return {
        "nama_file": source_file_name if source_file_name else Path(image_path).name,
        "luas_permukaan_px": float(luas_permukaan_px),
        "diameter_diagonal_px": float(diameter_diagonal_px),
        "diameter_tegak_lurus_px": float(diameter_tegak_lurus_px),
        "volume_approx": float(volume_approx),
    }


def save_features(feature_dict: dict, csv_path: str = FEATURES_CSV_PATH):
    df = pd.DataFrame(
        [feature_dict],
        columns=[
            "nama_file",
            "luas_permukaan_px",
            "diameter_diagonal_px",
            "diameter_tegak_lurus_px",
            "volume_approx",
        ],
    )
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=not file_exists)
    print(f"💾 Feature disimpan ke '{csv_path}' (append mode).")


def load_target_data(data_path: str):
    suffix = Path(data_path).suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(data_path)
        except Exception as exc:
            print(f"⚠️ Gagal baca sebagai Excel ({exc}). Mencoba baca sebagai CSV...")

    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(data_path, encoding=enc)
        except Exception:
            continue

    raise ValueError(
        f"File target '{data_path}' tidak bisa dibaca sebagai Excel maupun CSV. "
        "Pastikan format file benar."
    )


def normalize_column_name(col_name: str):
    normalized = str(col_name).strip().lower()
    normalized = normalized.replace("\ufeff", "")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("(", "").replace(")", "")
    return normalized


def prepare_training_data(features_csv_path: str, excel_path: str):
    if not os.path.exists(features_csv_path):
        raise FileNotFoundError(f"File fitur CSV tidak ditemukan: '{features_csv_path}'")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"File Excel tidak ditemukan: '{excel_path}'")

    df_features = pd.read_csv(features_csv_path)
    df_excel = load_target_data(excel_path)

    # Normalisasi nama kolom agar aman dari spasi/BOM.
    df_features.columns = [normalize_column_name(col) for col in df_features.columns]
    df_excel.columns = [normalize_column_name(col) for col in df_excel.columns]

    excel_alias_map = {
        "nama_file": "nama_file",
        "nama_file_": "nama_file",
        "nama": "nama_file",
        "file_name": "nama_file",
        "mass": "massa",
        "massa": "massa",
        "luas_permukaan_px": "luas_permukaan_px",
        "luas": "luas_permukaan_px",
        "luas_px": "luas_permukaan_px",
        "diameter_diagonal_px": "diameter_diagonal_px",
        "diameter_diagonal": "diameter_diagonal_px",
        "diameter_tegak_lurus_px": "diameter_tegak_lurus_px",
        "diameter_tegak_lurus": "diameter_tegak_lurus_px",
    }
    for source_col, target_col in excel_alias_map.items():
        if source_col in df_excel.columns and target_col not in df_excel.columns:
            df_excel = df_excel.rename(columns={source_col: target_col})

    target_col_normalized = normalize_column_name(TARGET_COLUMN)

    if "nama_file" not in df_excel.columns or target_col_normalized not in df_excel.columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di file target. "
            f"Wajib ada: 'nama_file' dan '{TARGET_COLUMN}'. "
            f"Kolom tersedia: {list(df_excel.columns)}"
        )

    # Hindari bentrok nama kolom fitur saat merge (mis. _x/_y).
    df_excel = df_excel[["nama_file", target_col_normalized]].copy()

    merged = pd.merge(df_features, df_excel, on="nama_file", how="inner")
    feature_columns_normalized = [normalize_column_name(col) for col in FEATURE_COLUMNS]
    merged = merged.dropna(subset=feature_columns_normalized + [target_col_normalized]).reset_index(drop=True)
    if merged.empty:
        raise ValueError(
            "Data gabungan fitur dan target kosong. "
            "Pastikan nilai 'nama_file' di features.csv sama dengan di file target."
        )

    X = merged[feature_columns_normalized]
    y = merged[target_col_normalized]
    return X, y, merged, feature_columns_normalized, target_col_normalized


def evaluate_model(model_name: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 Evaluasi {model_name}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R2  : {r2:.4f}")
    return {"model": model_name, "mae": mae, "r2": r2}


def train_model(features_csv_path: str = FEATURES_CSV_PATH, excel_path: str = EXCEL_PATH):
    X, y, merged_df, feature_columns_normalized, target_col_normalized = prepare_training_data(
        features_csv_path, excel_path
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
    }

    eval_results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = evaluate_model(name, y_test, y_pred)
        eval_results.append(result)
        trained_models[name] = model

    best_result = max(eval_results, key=lambda item: item["r2"])
    best_model_name = best_result["model"]
    best_model = trained_models[best_model_name]

    print(f"\n🏆 Model terbaik: {best_model_name}")

    # ✅ SIMPAN MODEL
    joblib.dump(best_model, "model.pkl")
    print("💾 Model disimpan sebagai model.pkl")

    # ✅ VALIDASI PER-FILE (seluruh data gabungan)
    validation_df = merged_df[["nama_file"] + feature_columns_normalized + [target_col_normalized]].copy()
    validation_df["prediksi_massa"] = best_model.predict(merged_df[feature_columns_normalized])
    validation_df["selisih"] = validation_df["prediksi_massa"] - validation_df[target_col_normalized]
    validation_df["abs_error"] = validation_df["selisih"].abs()
    validation_df = validation_df.rename(columns={target_col_normalized: "massa_asli"})
    validation_df.to_csv(VALIDATION_OUTPUT_PATH, index=False)
    print(f"🧪 Validasi per-file disimpan ke '{VALIDATION_OUTPUT_PATH}'")

    return best_model, best_model_name, eval_results


def predict_mass(model, luas_permukaan_px: float, diameter_diagonal_px: float, diameter_tegak_lurus_px: float):
    input_data = pd.DataFrame(
        [
            {
                "luas_permukaan_px": luas_permukaan_px,
                "diameter_diagonal_px": diameter_diagonal_px,
                "diameter_tegak_lurus_px": diameter_tegak_lurus_px,
            }
        ]
    )
    return float(model.predict(input_data)[0])


def main():
    try:
        model_unet, device = load_unet_model(MODEL_PATH)

        folder = "images_dataset"

        # ✅ HAPUS CSV LAMA
        if os.path.exists(FEATURES_CSV_PATH):
            os.remove(FEATURES_CSV_PATH)
            print("🧹 features.csv lama dihapus")

        print("\n🚀 Extract semua gambar...")
        success_count = 0
        failed_count = 0
        last_extracted = None

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):  # ✅ LEBIH AMAN
                path = os.path.join(folder, file)
                try:
                    extracted = extract_features(
                        image_path=path,
                        model=model_unet,
                        device=device,
                        source_file_name=file,
                        pixel_to_cm=PIXEL_TO_CM,
                        save_visualization=False,
                    )

                    save_features(extracted, FEATURES_CSV_PATH)
                    last_extracted = extracted
                    success_count += 1
                except Exception as image_err:
                    failed_count += 1
                    print(f"⚠️ Skip '{file}': {image_err}")

        if success_count == 0:
            raise ValueError("Tidak ada gambar yang berhasil diekstrak. Cek threshold/model/dataset.")

        print(f"✅ Ekstraksi selesai. Berhasil: {success_count}, Gagal: {failed_count}")

        print("\n🧠 Training model...")
        best_model, best_model_name, _ = train_model(FEATURES_CSV_PATH, EXCEL_PATH)

        print("\n🎯 Contoh prediksi...")
        pred_mass = predict_mass(
            best_model,
            last_extracted["luas_permukaan_px"],
            last_extracted["diameter_diagonal_px"],
            last_extracted["diameter_tegak_lurus_px"],
        )

        print(f"Prediksi massa ({best_model_name}): {pred_mass:.4f} kg")

    except Exception as err:
        print(f"❌ Error: {err}")


if __name__ == "__main__":
    main()