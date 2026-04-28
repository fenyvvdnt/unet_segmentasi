import streamlit as st
import cv2
import os
from pathlib import Path
import torch
import pandas as pd
from PIL import Image, ImageOps
import joblib  # ✅ TAMBAHAN

from unet import (
    ensure_model_file,
    load_unet_model, 
    extract_features, 
    predict_mass,
    IMGSZ,
    MODEL_PATH,
    FEATURES_CSV_PATH,
    EXCEL_PATH
)

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Estimasi Massa TBS", layout="wide")
st.title("🌴 Deteksi & Prediksi Massa Tandan Buah Segar (TBS)")

# ==========================================
# CACHE MODEL AGAR TIDAK LOAD BERULANG KALI
# ==========================================
ML_MODEL_PATH = Path("machine_learning") / "random_forest_massa.pkl"


@st.cache_resource
def init_models():
    drive_url = st.secrets.get("GOOGLE_DRIVE_MODEL_URL", None)
    drive_file_id = st.secrets.get("GOOGLE_DRIVE_MODEL_FILE_ID", None)
    ensure_model_file(MODEL_PATH, google_drive_url=drive_url, google_drive_file_id=drive_file_id)

    # Load U-Net
    model_unet, device = load_unet_model(MODEL_PATH)
    
    # ✅ LOAD MODEL ML HASIL RETRAIN TERBARU
    if not ML_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model ML tidak ditemukan: {ML_MODEL_PATH}")
    model_ml = joblib.load(str(ML_MODEL_PATH))
    model_name = "Random Forest (1080x1440 retrained)"
    
    return model_unet, device, model_ml, model_name

try:
    with st.spinner("Memuat AI Model... (U-Net & Regresi)"):
        model_unet, device, model_ml, model_name = init_models()
    st.sidebar.success(f"Model siap! Menggunakan ML: {model_name}")
except Exception as e:
    st.error(
        "Gagal memuat model/data. Pastikan file model ML tersedia dan "
        "set secret `GOOGLE_DRIVE_MODEL_URL` atau `GOOGLE_DRIVE_MODEL_FILE_ID` "
        "untuk mengunduh `unet_best.pth`. "
        f"Error: {e}"
    )
    st.stop()

# ==========================================
# SIDEBAR UNTUK KALIBRASI KONTROL
# ==========================================
st.sidebar.header("⚙️ Pengaturan Kalibrasi")
threshold = st.sidebar.slider("Threshold Segmentasi", 0.0, 1.0, 0.85, 0.05)

# ==========================================
# AREA UPLOAD GAMBAR
# ==========================================
uploaded_file = st.file_uploader("Upload Foto TBS (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Simpan sementara gambar yang diupload
    temp_path = f"temp_{uploaded_file.name}"
    with Image.open(uploaded_file) as img:
        img_fixed = ImageOps.exif_transpose(img).convert("RGB")
        img_fixed.save(temp_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Gambar Asli")
        st.image(temp_path, use_column_width=True)

    # 2. Proses Ekstraksi & Prediksi saat tombol ditekan
    if st.button("🚀 Prediksi Massa TBS", type="primary", use_container_width=True):
        with st.spinner("Menganalisis gambar dan menghitung geometri..."):
            try:
                hasil_ekstrak = extract_features(
                    image_path=temp_path,
                    model=model_unet,
                    device=device,
                    threshold=threshold,
                    img_size=IMGSZ,
                    save_visualization=True
                )
                
                # Tampilkan Gambar Hasil Segmentasi
                with col2:
                    st.subheader("🔍 Hasil Segmentasi")
                    if os.path.exists("step4_final_diameter_lurus_unet.jpg"):
                        st.image("step4_final_diameter_lurus_unet.jpg", use_column_width=True)
                
                # Prediksi ML
                estimasi_kg = predict_mass(
                    model=model_ml,
                    luas_permukaan_px=hasil_ekstrak["luas_permukaan_px"],
                    diameter_diagonal_px=hasil_ekstrak["diameter_diagonal_px"],
                    diameter_tegak_lurus_px=hasil_ekstrak["diameter_tegak_lurus_px"],
                )
                
                # Dashboard hasil
                st.divider()
                st.subheader("📊 Hasil Kalkulasi Akhir")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Diameter Diagonal", f"{hasil_ekstrak['diameter_diagonal_px']:.2f} px")
                m2.metric("Diameter Tegak Lurus", f"{hasil_ekstrak['diameter_tegak_lurus_px']:.2f} px")
                m3.metric("Luas Permukaan", f"{hasil_ekstrak['luas_permukaan_px']:.2f} px²")
                m4.metric("⚖️ ESTIMASI MASSA", f"{estimasi_kg:.2f} Kg", delta_color="off")
                st.caption(f"Feature space: {IMGSZ[1]}x{IMGSZ[0]} (rasio 3:4)")
                
                st.success("Analisis selesai!")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)