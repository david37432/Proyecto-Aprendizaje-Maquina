import joblib
import pandas as pd

class DiabetesPredictor:
    """Usa el modelo guardado para hacer predicciones."""

    def __init__(self):
        self.modelo = joblib.load("models/modelo_diabetes.joblib")

    def predecir(self, datos_usuario):
        df = pd.DataFrame([datos_usuario])
        prob = self.modelo.predict_proba(df)[0][1]
        pred = self.modelo.predict(df)[0]

        resultado = "Diabetes" if pred == 1 else "No diabetes"
        print(f"Resultado: {resultado} (Probabilidad: {prob:.2%})")

        return resultado, prob