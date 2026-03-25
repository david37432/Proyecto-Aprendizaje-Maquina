import pandas as pd
import numpy as np

class ProcesadorDatos:

    def preparar_datos(self, datos):
        df = datos.copy()

        eliminar = ['SkinThickness', 'Insulin']
        df.drop(eliminar, axis=1, inplace=True)

        mediana_BMI = df["BMI"].median()
        df["BMI"].fillna(mediana_BMI, inplace=True)

        media_glucose = df["Glucose"].mean()
        df["Glucose"].fillna(media_glucose, inplace=True)

        X=df.drop("Outcome", axis=1)
        y=df["Outcome"]

        return X,y