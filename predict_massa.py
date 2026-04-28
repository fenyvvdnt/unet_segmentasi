from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "random_forest_massa.pkl"
FEATURE_COLUMNS = ["luas_permukaan_px", "diameter_diagonal_px", "diameter_tegak_lurus_px"]


def predict(model, luas_permukaan_px: float, diameter_diagonal_px: float, diameter_tegak_lurus_px: float) -> float:
    X = pd.DataFrame(
        [
            {
                "luas_permukaan_px": float(luas_permukaan_px),
                "diameter_diagonal_px": float(diameter_diagonal_px),
                "diameter_tegak_lurus_px": float(diameter_tegak_lurus_px),
            }
        ]
    )
    return float(model.predict(X)[0])


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    print("Gunakan nilai fitur dari hasil segmentasi pada feature space 1080x1440 (bukan piksel asli foto).")
    print(f"Model yang dipakai: {MODEL_PATH}")
    luas = float(input("luas_permukaan_px: ").strip())
    diag = float(input("diameter_diagonal_px: ").strip())
    tegak = float(input("diameter_tegak_lurus_px: ").strip())

    model = joblib.load(MODEL_PATH)
    y = predict(model, luas, diag, tegak)
    print(f"\nPrediksi massa: {y:.4f}")


if __name__ == "__main__":
    main()

