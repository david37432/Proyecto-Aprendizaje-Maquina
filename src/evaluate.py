import os
import json
import joblib
import pandas as pd
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.model_selection import learning_curve

def run_evaluation(output_dir: str = "../reports") -> None:
    """
    Carga modelos entrenados, genera métricas y gráficas de evaluación
    (Confusion Matrix, ROC Curve, Learning Curve, comparación de modelos)
    y las guarda en output_dir.

    Args:
        output_dir: Directorio donde se guardarán imágenes y metrics.json.
    """

    models_dir = "../models"
    data_path  = "../data/prepared/datos_df.pkl"

    os.makedirs(output_dir, exist_ok=True)
    # -------------------------
    # 1. Configuración
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

    with open(f"{MODEL_DIR}/features.json", "r") as f:
        features = json.load(f)

    X = df[features]
    y = df["stress_category"]

    # -------------------------
    # 3. Cargar modelos
    # -------------------------
    models = {
        "logistic_regression": joblib.load(f"{MODEL_DIR}/logistic_regression.joblib"),
        "gaussian_nb": joblib.load(f"{MODEL_DIR}/gaussian_nb.joblib"),
        "sgd_classifier": joblib.load(f"{MODEL_DIR}/sgd_classifier.joblib")
    }

    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")

    # -------------------------
    # 4. Evaluación
    # -------------------------
    results = {}

    for name, model in models.items():

        # decidir input
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

        # -------------------------
        # 📊 1. Confusion Matrix
        # -------------------------
        cm = confusion_matrix(y, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(f"{REPORT_DIR}/cm_{name}.png")
        plt.close()

        # -------------------------
        # 📈 2. ROC Curve
        # -------------------------
        if hasattr(model, "predict_proba"):

            if name in ["logistic_regression"]:
                probs = model.predict_proba(X_input)[:, 1]

                fpr, tpr, _ = roc_curve(y, probs, pos_label="Alto")
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.title(f"ROC Curve - {name}")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.legend()

                plt.savefig(f"{REPORT_DIR}/roc_{name}.png")
                plt.close()

        # -------------------------
        # 📉 3. Learning Curve
        # -------------------------
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_input,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        plt.figure()
        plt.plot(train_sizes, train_mean, label="Train")
        plt.plot(train_sizes, test_mean, label="Test")
        plt.title(f"Learning Curve - {name}")
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig(f"{REPORT_DIR}/lc_{name}.png")
        plt.close()

    # -------------------------
    # 📊 4. Comparación de modelos
    # -------------------------
    df_results = pd.DataFrame(results).T

    plt.figure()
    df_results["accuracy"].plot(kind="bar")
    plt.title("Model Comparison - Accuracy")
    plt.xticks(rotation=45)

    plt.savefig(f"{REPORT_DIR}/model_comparison.png")
    plt.close()

    # -------------------------
    # 💾 5. Guardar métricas
    # -------------------------
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Evaluación completa. Resultados en /reports/")