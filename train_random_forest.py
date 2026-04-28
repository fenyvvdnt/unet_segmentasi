import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


EXCEL_PATH = Path(__file__).resolve().parent / "hasil_pengukuran_unet.xlsx"
MODEL_OUT = Path(__file__).resolve().parent / "random_forest_massa.pkl"

FEATURE_COLUMNS = ["luas_permukaan_px", "diameter_diagonal_px", "diameter_tegak_lurus_px"]
TARGET_COLUMN = "massa"
RANDOM_STATE = 42
TEST_SIZE = 13 / 93


def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel training tidak ditemukan: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    for col in ["nama_file"] + FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ada di {EXCEL_PATH}. Kolom tersedia: {list(df.columns)}")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Data training kosong setelah dropna.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nEvaluasi Random Forest (feature space 1080x1440)")
    print(f"   MAE : {mae:.4f}")
    print(f"   R2  : {r2:.4f}")

    # Refit ke seluruh data untuk model inferensi akhir (data relatif kecil).
    model.fit(X, y)
    os.makedirs(MODEL_OUT.parent, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel inferensi (refit full data) disimpan: {MODEL_OUT}")


if __name__ == "__main__":
    main()

