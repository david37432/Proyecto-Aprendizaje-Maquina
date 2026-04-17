# src/predict.py

import joblib
import pandas as pd
import json
import os

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
def run_prediction(output_dir: str = "../reports") -> None:
    """
    Carga los modelos entrenados, evalúa sobre los datos preparados
    y guarda las métricas de predicción en output_dir.

    Args:
        output_dir: Directorio donde se guardará predict_metrics.json.
    """

    models_dir = "../models"
    data_path  = "../data/prepared/datos_df.pkl"
    # -------------------------
    # 1. Cargar datos
    # -------------------------
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH  = os.path.join(BASE_DIR, "../data/prepared", "datos_df.pkl")
    MODEL_DIR  = os.path.join(BASE_DIR, "../models")
    REPORT_DIR = os.path.join(BASE_DIR, "../reports")
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Carga de datos ────────────────────────────────────────────────────────────
    print("📂 Cargando datos...")
    df = pd.read_pickle(DATA_PATH)

    # -------------------------
    # 2. Cargar features usadas
    # -------------------------
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        features = json.load(f)

    X = df[features]
    y = df["stress_category"]

    # -------------------------
    # 3. Cargar modelos
    # -------------------------
    models = {
        "logistic_regression": joblib.load("./models/logistic_regression.joblib"),
        "gaussian_nb": joblib.load("./models/gaussian_nb.joblib"),
        "sgd_classifier": joblib.load("./models/sgd_classifier.joblib")
    }

    scaler = joblib.load("./models/scaler.joblib")

    # -------------------------
    # 4. Evaluación
    # -------------------------
    results = {}

    for name, model in models.items():

        if name in ["logistic_regression", "sgd_classifier"]:
            X_input = scaler.transform(X)
        else:
            X_input = X

        preds = model.predict(X_input)

        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")

        results[name] = {
            "accuracy": acc,
            "f1_score": f1
        }

        print(f"\n🔹 Modelo: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y, preds))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y, preds))

    # -------------------------
    # 5. Guardar métricas
    # -------------------------
    os.makedirs("../reports", exist_ok=True)

    with open("../reports/predict_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Métricas guardadas en ../reports/predict_metrics.json")