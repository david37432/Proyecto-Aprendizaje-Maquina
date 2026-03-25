import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.preprocessing import ProcesadorDatos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class DiabetesModel:

    def entrenar_modelo(self, df):

        procesador_datos = ProcesadorDatos()

        X, y = procesador_datos.preparar_datos(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        escala = StandardScaler()
        X_train = escala.fit_transform(X_train)
        X_test = escala.transform(X_test)

        RGL = LogisticRegression(class_weight = 'balanced', random_state=42)
        RGL.fit(X_train, y_train)

        y_pred = RGL.predict(X_test)
        y_proba = RGL.predict_proba(X_test)[:, 1]

        #Calcular metricas
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred), 3),
            "recall": round(recall_score(y_test, y_pred), 3),
            "f1": round(f1_score(y_test, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 3)
        }

        # Guardar modelo
        joblib.dump(RGL, "models/modelo_diabetes.joblib")
        print("Modelo guardado en 'models/modelo_diabetes.joblib'")

        joblib.dump(escala, "models/scaler_diabetes.joblib")
        print("Escalador guardado en 'models/scaler_diabetes.joblib'")

        #Guardar métricas
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("Métricas guardadas en 'reports/metrics.json'")

        return metrics