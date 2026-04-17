# src/train.py

import os
import json
import joblib
from lazypredict import LazyClassifier, LazyRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Modelos
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif , mutual_info_regression


# -------------------------
# 1. Cargar datos
# -------------------------
df = pd.read_pickle('../data/prepared/datos_df.pkl')
# -------------------------
# 2. Decision entre regresión o clasificación
# -------------------------

    ## -------------------------
    ## 2.1 Regresión 
    ## -------------------------
X = df.drop(columns=["stress_level", "stress_category"]) # Eliminar variables objetivo
y = df["stress_level"] # Variable objetivo para regresión

X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

importances = mutual_info_regression(
    X_train_regression, 
    y_train_regression,
    random_state=42
)
ranking = pd.Series(importances, index=X_train_regression.columns).sort_values(ascending=False) # Mostrar ranking de importancia de features
print("Ranking de importancia de features (regresión):")
print(ranking)
# Automático: TOP 8 (o lo que quieras)
n_top = 8
features_selected_regression = ranking.head(n_top).index.tolist()
print("Top", n_top, "features:", features_selected_regression)

reg = LazyRegressor(verbose=0, ignore_warnings=True) # Evaluar múltiples modelos de regresión
models, predictions = reg.fit(X_train_regression, X_test_regression, y_train_regression, y_test_regression)
print("Resultados de modelos de regresión:")
print(models.sort_values("R-Squared", ascending=False))
## El dataset no contiene variables suficientemente predictivas del nivel de estrés en terminos de regresion. 
# Se evaluaron múltiples modelos sin lograr mejoras sobre el baseline, lo que indica baja capacidad predictiva de las variables disponibles.
with open("../models/features.json", "w") as f:
    json.dump(features_selected_regression, f)

    ## -------------------------
    ## 2.2 Clasificación
    ## -------------------------
X = df.drop(columns=['stress_level', 'stress_category'], errors='ignore')
y = df['stress_category']
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
importances = mutual_info_classif(
    X_train_classification, 
    y_train_classification,
    random_state=42
)
ranking = pd.Series(importances, index=X_train_classification.columns).sort_values(ascending=False)
print("Ranking de importancia de features para clasificación:")
print(ranking)
# Automático: TOP 8 (o lo que quieras)
n_top = 8
features_selected_classification = ranking.head(n_top).index.tolist()
print("Top", n_top, "features:", features_selected_classification)

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train_classification, X_test_classification, y_train_classification, y_test_classification)

print("Resultados de modelos de clasificación:")
print(models.sort_values("Accuracy", ascending=False))
##La reducción de variables permitió mejorar ligeramente el desempeño, evidenciando que el ruido en los datos afectaba el aprendizaje. 
# Sin embargo, incluso con las variables más relevantes, la capacidad predictiva sigue siendo limitada.
# Guardar features seleccionadas
import json

with open("../models/features.json", "w") as f:
    json.dump(features_selected_classification, f)
# -------------------------
# 4. Escalado (solo para modelos que lo necesitan)
# -------------------------
X_train_selected = X_train_classification[features_selected_classification]
X_test_selected = X_test_classification[features_selected_classification]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_selected) 
X_test_scaled = scaler.transform(X_test_selected)     

# -------------------------
# 5. Modelos
# -------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "gaussian_nb": GaussianNB(),
    "sgd_classifier": SGDClassifier()
}

results = {}

# -------------------------
# 6. Entrenamiento y evaluación
# -------------------------
for name, model in models.items():

    # decidir si usar escalado
    if name in ["logistic_regression", "sgd_classifier"]:
        model.fit(X_train_scaled, y_train_classification)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train_selected, y_train_classification)
        preds = model.predict(X_test_selected)

    acc = accuracy_score(y_test_classification, preds)
    f1 = f1_score(y_test_classification, preds, average="weighted")

    results[name] = {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": classification_report(y_test_classification, preds, output_dict=True)
    }

    # -------------------------
    # 7. Guardar modelo
    # -------------------------
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, f"../models/{name}.joblib")

# -------------------------
# 8. Guardar scaler
# -------------------------
joblib.dump(scaler, "../models/scaler.joblib")

# -------------------------
# 9. Guardar métricas
# -------------------------
os.makedirs("../reports", exist_ok=True)

with open("../reports/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Entrenamiento completo")

# El modelo con mejor desempeño fue el SGDClassifier, aunque la capacidad predictiva sigue siendo limitada,
#  lo que sugiere que las variables disponibles no capturan suficientemente la complejidad del estrés académico.
# Se entrenaron múltiples modelos de clasificación con diferentes enfoques (lineales, probabilísticos y basados en optimización).
# A pesar de aplicar selección de variables, el desempeño se mantuvo cercano al baseline, lo que indica una limitada capacidad predictiva de las variables disponibles