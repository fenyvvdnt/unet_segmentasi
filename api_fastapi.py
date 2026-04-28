import io
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from unet import MODEL_PATH, THRESHOLD, load_unet_model, predict_mass


app = FastAPI(title="U-Net TBS API", version="1.0")

_model_unet = None
_device = None
_model_ml = None

FEATURE_HEIGHT = 1440
FEATURE_WIDTH = 1080
OUTPUT_DIR = Path("hasil_api")
ML_MODEL_PATH = Path("machine_learning") / "random_forest_massa.pkl"
# Rentang sementara dari data training normalisasi terbaru
# (`hasil_pengukuran_unet_regen_1280.xlsx`, 1080x1440).
TRAINING_FEATURE_RANGES = {
    "luas_permukaan_px": {"min": 354.5, "max": 320511.0},
    "diameter_diagonal_px": {"min": 42.0026, "max": 737.919676},
    "diameter_tegak_lurus_px": {"min": 12.570788, "max": 616.476755},
}


def _ensure_loaded():
    global _model_unet, _device, _model_ml
    if _model_unet is None or _device is None:
        _model_unet, _device = load_unet_model(MODEL_PATH)
    if _model_ml is None:
        if not ML_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model ML tidak ditemukan: {ML_MODEL_PATH}")
        _model_ml = joblib.load(str(ML_MODEL_PATH))


def _largest_contour_mask(mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Objek tidak terdeteksi pada mask segmentasi.")
    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask_u8)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    return cleaned, largest


def _extract_geometry_from_mask(cleaned_mask_u8: np.ndarray) -> dict:
    contours, _ = cv2.findContours(cleaned_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Kontur tidak ditemukan pada cleaned_mask.")

    largest_contour = max(contours, key=cv2.contourArea)
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

    return {
        "luas_permukaan_px": float(area_px),
        "diameter_diagonal_px": float(major_max - major_min),
        "diameter_tegak_lurus_px": float(minor_max - minor_min),
        "centroid": centroid,
        "major_direction": major_direction,
        "minor_direction": minor_direction,
        "major_min": float(major_min),
        "major_max": float(major_max),
        "minor_min": float(minor_min),
        "minor_max": float(minor_max),
    }


def _build_visualization(image_bgr: np.ndarray, cleaned_mask_feature_u8: np.ndarray, geo: dict) -> np.ndarray:
    original_h, original_w = image_bgr.shape[:2]
    mask_original = cv2.resize(cleaned_mask_feature_u8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    visualization = image_bgr.copy()
    overlay = visualization.copy()
    overlay[mask_original > 0] = (0, 255, 0)
    visualization = cv2.addWeighted(overlay, 0.45, visualization, 0.55, 0)

    contours_original, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_original:
        largest_contour_original = max(contours_original, key=cv2.contourArea)
        cv2.drawContours(visualization, [largest_contour_original], -1, (0, 200, 0), 2)

    centroid = geo["centroid"]
    major_direction = geo["major_direction"]
    minor_direction = geo["minor_direction"]

    p_major_1f = centroid + (geo["major_min"] * major_direction)
    p_major_2f = centroid + (geo["major_max"] * major_direction)
    p_minor_1f = centroid + (geo["minor_min"] * minor_direction)
    p_minor_2f = centroid + (geo["minor_max"] * minor_direction)

    scale_x = original_w / float(FEATURE_WIDTH)
    scale_y = original_h / float(FEATURE_HEIGHT)

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
        f"diag: {geo['diameter_diagonal_px']:.2f}px | tegak lurus: {geo['diameter_tegak_lurus_px']:.2f}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return visualization


def _check_feature_range(seg: dict) -> dict:
    warnings = []
    for feature_name, range_info in TRAINING_FEATURE_RANGES.items():
        value = float(seg[feature_name])
        min_v = float(range_info["min"])
        max_v = float(range_info["max"])
        if value < min_v or value > max_v:
            warnings.append(
                f"{feature_name}={value:.4f} di luar rentang training [{min_v:.4f}, {max_v:.4f}]"
            )
    return {
        "is_within_training_range": len(warnings) == 0,
        "warnings": warnings,
    }


def extract_features_from_image(image_path: str):
    _ensure_loaded()
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = ImageOps.exif_transpose(pil_image)
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((FEATURE_HEIGHT, FEATURE_WIDTH)),
            T.ToTensor(),
        ]
    )
    input_tensor = transform(image_rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        pred = _model_unet(input_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Inferensi segmentasi dan perhitungan fitur berada pada feature space yang sama: 1080x1440.
    pred_mask_feature = (pred > THRESHOLD).astype(np.uint8) * 255
    cleaned_mask_feature_u8, _ = _largest_contour_mask(pred_mask_feature)
    geo = _extract_geometry_from_mask(cleaned_mask_feature_u8)

    visualization = _build_visualization(image_bgr, cleaned_mask_feature_u8, geo)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"hasil_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), visualization)

    return {
        "segmentation_results": {
            "luas_permukaan_px": geo["luas_permukaan_px"],
            "diameter_diagonal_px": geo["diameter_diagonal_px"],
            "diameter_tegak_lurus_px": geo["diameter_tegak_lurus_px"],
            "feature_space": f"{FEATURE_WIDTH}x{FEATURE_HEIGHT}",
            "segment_inference_size": f"{FEATURE_WIDTH}x{FEATURE_HEIGHT}",
        },
        "output_image": str(out_path),
    }


@app.get("/health")
def health():
    _ensure_loaded()
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_loaded()
    try:
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        pil = ImageOps.exif_transpose(pil)
        tmp_path = Path("_upload_tmp.jpg")
        pil.save(tmp_path)

        result = extract_features_from_image(str(tmp_path))
        seg = result["segmentation_results"]
        range_check = _check_feature_range(seg)
        pred_kg = predict_mass(
            _model_ml,
            seg["luas_permukaan_px"],
            seg["diameter_diagonal_px"],
            seg["diameter_tegak_lurus_px"],
        )
        pred_gram = float(pred_kg * 1000.0)

        return JSONResponse(
            {
                "nama_file": file.filename,
                "segmentation_results": seg,
                "predicted_mass_gram": pred_gram,
                "output_image": result["output_image"],
                "range_check": range_check,
            }
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    finally:
        try:
            Path("_upload_tmp.jpg").unlink(missing_ok=True)
        except Exception:
            pass

