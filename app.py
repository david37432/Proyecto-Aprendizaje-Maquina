"""
app.py  –  Despliegue Streamlit: Predicción de Nivel de Estrés Estudiantil
Ejecutar: streamlit run app.py
"""

import os, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
matplotlib.use("Agg")

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Predictor de Estrés Estudiantil",
    page_icon="🧠",
    layout="wide",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] hr { border-color: #333; }

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    border: none; color: white; font-weight: 700;
    font-size: 1.05rem; border-radius: 10px;
    padding: 0.6rem 1.2rem; transition: opacity .2s;
}
div.stButton > button[kind="primary"]:hover { opacity: .85; }

.result-card {
    border-radius: 12px; padding: 22px 26px;
    margin-bottom: 18px; box-shadow: 0 4px 14px rgba(0,0,0,.15);
}
[data-testid="metric-container"] {
    background: #1e2130; border-radius: 10px; padding: 14px 16px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR   = os.path.join(BASE_DIR, "data", "prepared")

COLORES = {"Bajo": "#2ECC71", "Alto": "#E74C3C"}
ICONOS  = {"Bajo": "😌",     "Alto": "😰"}

NOMBRES_MODELO = {
    "logistic_regression": "Regresión Logística",
    "gaussian_nb":         "Gaussian Naive Bayes",
    "sgd_classifier":      "SGD Classifier",
}

# ── Carga de artefactos ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_artefactos():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
        cols = json.load(f)
    modelos = {}
    for nombre in NOMBRES_MODELO:
        path = os.path.join(MODEL_DIR, f"{nombre}.joblib")
        if os.path.exists(path):
            modelos[nombre] = joblib.load(path)
    return scaler, cols, modelos

@st.cache_data(show_spinner=False)
def cargar_metricas():
    path = os.path.join(REPORT_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def cargar_datos():
    path = os.path.join(DATA_DIR, "datos_df.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    return None

scaler, feature_cols, modelos = cargar_artefactos()
metricas = cargar_metricas()
df_full  = cargar_datos()

# ── Ingeniería de características ─────────────────────────────────────────────
def construir_features(inp: dict) -> pd.DataFrame:
    d = inp.copy()
    d["total_screen_hours"]             = d["phone_usage_hours"] + d["social_media_hours"] + d["youtube_hours"] + d["gaming_hours"]
    d["study_sleep_ratio"]              = d["study_hours_per_day"] / max(d["sleep_hours"], 0.1)
    d["academic_efficiency_tasks"]      = d["assignments_completed"] * d["final_grade"]
    d["academic_efficiency_attendance"] = d["final_grade"] * d["attendance_percentage"] / 100
    d["caffeine_per_study_hour"]        = d["coffee_intake_mg"] / max(d["study_hours_per_day"], 0.1)
    d["gaming_sleep_interaction"]       = d["gaming_hours"] * d["sleep_hours"]
    d["study_phone_interaction"]        = d["study_hours_per_day"] * d["phone_usage_hours"]
    d["screen_exercise_interaction"]    = d["total_screen_hours"] * d["exercise_minutes"]
    d["coffee_sleep_interaction"]       = d["coffee_intake_mg"] * d["sleep_hours"]
    d["academic_efficiency"]            = d["assignments_completed"] * d["final_grade"]
    d["sleep_deficit"]                  = 8 - d["sleep_hours"]
    d["recovery_ratio"]                 = (d["sleep_hours"] + d["exercise_minutes"] / 60) / (d["total_screen_hours"] + 1e-9)
    d["task_overwhelm_index"]           = (d["study_hours_per_day"] + d["total_screen_hours"]) / (d["breaks_per_day"] + 1e-9)
    df = pd.DataFrame([d])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Predictor de Estrés")
    st.markdown("*Proyecto CRISP-DM · ML Aplicado*")
    st.divider()

    pagina = st.radio(
        "Navegación",
        ["🔮 Predicción", "📊 Evaluación de Modelos", "ℹ️ Acerca del Proyecto"],
        label_visibility="collapsed",
    )
    st.divider()

    if metricas:
        mejor_k = max(metricas, key=lambda k: metricas[k]["f1_score"])
        mejor_m = metricas[mejor_k]
        st.markdown("**🏆 Mejor modelo**")
        st.markdown(f"`{NOMBRES_MODELO.get(mejor_k, mejor_k)}`")
        st.markdown(f"F1 **{mejor_m['f1_score']:.3f}** · Acc **{mejor_m['accuracy']:.3f}**")
        st.divider()

    st.caption("Dataset: Student Productivity & Distraction · Kaggle CC0")

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 – PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "🔮 Predicción":
    st.title("🔮 Predicción de Nivel de Estrés")
    st.markdown(
        "Ajusta los controles con los hábitos del estudiante y pulsa **Predecir** "
        "para obtener la estimación de los tres modelos."
    )

    top_col1, top_col2 = st.columns([2, 1])
    with top_col1:
        modelo_sel = st.selectbox(
            "Modelo principal",
            list(NOMBRES_MODELO.keys()),
            format_func=lambda k: NOMBRES_MODELO[k],
        )
    with top_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**{len(modelos)}** modelos cargados ✓", icon="✅")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📚 Académico")
        study_hours = st.slider("Horas de estudio/día", 0.0, 12.0, 5.0, 0.1)
        assignments = st.slider("Tareas completadas",   0.0, 18.0, 8.0, 0.5)
        attendance  = st.slider("Asistencia (%)",       40.0, 100.0, 80.0, 1.0)
        final_grade = st.slider("Nota final",           40.0, 100.0, 70.0, 1.0)
        focus_score = st.slider("Puntaje de enfoque",   30, 100, 65)

    with col2:
        st.markdown("### 😴 Hábitos")
        sleep_hours  = st.slider("Horas de sueño/día",  3.0, 10.0, 7.0, 0.1)
        exercise_min = st.slider("Ejercicio (min/día)",  0, 120, 30)
        coffee_mg    = st.slider("Cafeína (mg/día)",     0, 500, 150)
        breaks       = st.slider("Descansos/día",        1, 15, 5)

    with col3:
        st.markdown("### 📱 Pantallas")
        phone_hours  = st.slider("Teléfono (h/día)",        0.0, 12.0, 3.0, 0.1)
        social_hours = st.slider("Redes sociales (h/día)",  0.0,  8.0, 2.0, 0.1)
        youtube_h    = st.slider("YouTube (h/día)",         0.0,  6.0, 1.0, 0.1)
        gaming_h     = st.slider("Videojuegos (h/día)",     0.0,  6.0, 1.0, 0.1)

    # ── Indicadores en tiempo real ────────────────────────────────────────────
    total_screen = phone_hours + social_hours + youtube_h + gaming_h
    sleep_def    = max(0.0, 8.0 - sleep_hours)

    ind1, ind2, ind3, ind4 = st.columns(4)
    ind1.metric("📺 Pantallas totales",    f"{total_screen:.1f} h/día",
                delta=f"{total_screen - 6:.1f}h vs límite 6h", delta_color="inverse")
    ind2.metric("😴 Déficit de sueño",    f"{sleep_def:.1f} h",       delta_color="inverse")
    ind3.metric("📚 Ratio estudio/sueño", f"{study_hours/max(sleep_hours,0.1):.2f}")
    ind4.metric("☕ Cafeína/h estudio",   f"{coffee_mg/max(study_hours,0.1):.0f} mg",
                delta_color="off")

    st.divider()

    if st.button("🔍 Predecir nivel de estrés", type="primary", use_container_width=True):
        entrada = {
            "study_hours_per_day":   study_hours,
            "sleep_hours":           sleep_hours,
            "phone_usage_hours":     phone_hours,
            "social_media_hours":    social_hours,
            "youtube_hours":         youtube_h,
            "gaming_hours":          gaming_h,
            "breaks_per_day":        breaks,
            "coffee_intake_mg":      coffee_mg,
            "exercise_minutes":      exercise_min,
            "assignments_completed": assignments,
            "attendance_percentage": attendance,
            "focus_score":           focus_score,
            "final_grade":           final_grade,
        }

        X_new  = construir_features(entrada)
        X_sc   = scaler.transform(X_new)
        modelo = modelos[modelo_sel]
        pred   = modelo.predict(X_sc)[0]
        proba  = modelo.predict_proba(X_sc)[0] if hasattr(modelo, "predict_proba") else None

        color = COLORES.get(pred, "#888")
        icono = ICONOS.get(pred, "🤔")

        # Banner resultado
        st.markdown(
            f"""<div class="result-card"
                style="background:{color}18; border-left:6px solid {color};">
                <h2 style="color:{color}; margin:0 0 6px 0;">{icono} Nivel: {pred}</h2>
                <p style="margin:0; color:#aaa; font-size:.95rem;">
                    Modelo: <b>{NOMBRES_MODELO[modelo_sel]}</b>
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Barras de probabilidad
        if proba is not None:
            st.markdown("#### Confianza por clase")
            prob_df = (
                pd.DataFrame({"Clase": modelo.classes_, "Prob": proba})
                .sort_values("Prob", ascending=False)
            )
            for _, row in prob_df.iterrows():
                cl  = row["Clase"]
                pct = row["Prob"]
                c   = COLORES.get(cl, "#888")
                ico = ICONOS.get(cl, "")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'  <span style="width:80px;font-weight:600;">{ico} {cl}</span>'
                    f'  <div style="flex:1;background:#2a2a3a;border-radius:6px;height:18px;">'
                    f'    <div style="width:{pct*100:.1f}%;background:{c};border-radius:6px;height:18px;"></div>'
                    f'  </div>'
                    f'  <span style="width:48px;text-align:right;font-weight:700;color:{c};">{pct*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Comparación entre modelos
        st.markdown("#### 🔄 Todos los modelos")
        res_cols = st.columns(len(modelos))
        votos = []
        for i, (nm, mod) in enumerate(modelos.items()):
            pcls = mod.predict(X_sc)[0]
            votos.append(pcls)
            c       = COLORES.get(pcls, "#888")
            acuerdo = "✓ Coincide" if pcls == pred else "≠ Difiere"
            c_ac    = "#2ECC71" if pcls == pred else "#F39C12"
            with res_cols[i]:
                st.markdown(
                    f"""<div style="background:{c}18;border:1px solid {c};
                            padding:14px;border-radius:10px;text-align:center;">
                        <div style="font-size:.8rem;color:#aaa;margin-bottom:4px;">
                            {NOMBRES_MODELO[nm]}</div>
                        <div style="font-size:2rem;">{ICONOS.get(pcls,'🤔')}</div>
                        <div style="color:{c};font-weight:700;">{pcls}</div>
                        <div style="font-size:.75rem;color:{c_ac};margin-top:4px;">{acuerdo}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Consenso
        mayoria   = max(set(votos), key=votos.count)
        pct_ac    = votos.count(mayoria) / len(votos) * 100
        c_mayoria = COLORES.get(mayoria, "#888")
        st.markdown(
            f"""<div style="background:#1e2130;border-radius:10px;padding:14px 20px;
                    margin-top:16px;border-left:4px solid {c_mayoria};">
                🗳️ <b>Consenso:</b> {ICONOS.get(mayoria,'')} <b>{mayoria}</b>
                &nbsp;·&nbsp; {votos.count(mayoria)}/{len(votos)} modelos ({pct_ac:.0f}%)
            </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 – EVALUACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Evaluación de Modelos":
    st.title("📊 Evaluación de Modelos")

    if not metricas:
        st.error("No se encontró reports/metrics.json.")
        st.stop()

    mejor_k = max(metricas, key=lambda k: metricas[k]["f1_score"])
    mejor_m = metricas[mejor_k]
    st.success(
        f"🏆 Mejor modelo: **{NOMBRES_MODELO.get(mejor_k, mejor_k)}** "
        f"— Accuracy **{mejor_m['accuracy']:.4f}** · F1 **{mejor_m['f1_score']:.4f}**"
    )

    # Tabla con highlight
    st.subheader("Resumen comparativo")
    filas = []
    for nm, m in metricas.items():
        filas.append({
            "Modelo":   ("🏆 " if nm == mejor_k else "") + NOMBRES_MODELO.get(nm, nm),
            "Accuracy": round(m["accuracy"],  4),
            "F1 Score": round(m["f1_score"],  4),
        })
    df_tab = pd.DataFrame(filas).set_index("Modelo")
    st.dataframe(
        df_tab.style.highlight_max(axis=0, color="#2d4a2d"),
        use_container_width=True,
    )

    # Gráfico
    st.subheader("Comparativa visual")
    nombres  = [NOMBRES_MODELO.get(k, k) for k in metricas]
    acc_vals = [v["accuracy"]  for v in metricas.values()]
    f1_vals  = [v["f1_score"]  for v in metricas.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    x  = np.arange(len(nombres))
    b1 = ax.bar(x - 0.2, acc_vals, 0.35, label="Accuracy", color="#3498DB")
    b2 = ax.bar(x + 0.2, f1_vals,  0.35, label="F1 Score", color="#9B59B6")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=9, color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Métrica", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1e2130", labelcolor="white")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8, color="white")
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8, color="white")
    ax.set_title("Accuracy vs F1 Score por modelo", color="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Detalle
    st.subheader("Detalle por modelo")
    modelo_det = st.selectbox(
        "Selecciona modelo",
        list(metricas.keys()),
        format_func=lambda k: NOMBRES_MODELO.get(k, k),
        index=list(metricas.keys()).index(mejor_k),
    )
    m = metricas[modelo_det]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['accuracy']:.4f}")
    c2.metric("F1 Score", f"{m['f1_score']:.4f}")

   
# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 – ACERCA DEL PROYECTO
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "ℹ️ Acerca del Proyecto":
    st.title("ℹ️ Acerca del Proyecto")

    st.markdown("""
## 🎯 Objetivo del proyecto

Predecir el **nivel de estrés** de estudiantes universitarios a partir de sus hábitos académicos, sueño, uso de pantallas y actividad física, utilizando la metodología **CRISP-DM**.

---

## 🎯 Variable objetivo

`stress_category` — derivada de `stress_level` (escala 1–10):

| Rango  | Categoría |
|--------|-----------|
| 1 – 5  | 😌 Bajo   |
| 6 – 10 | 😰 Alto   |

---

## 🤖 Modelos entrenados

| Modelo |
|--------|
| Regresión Logística |
| Gaussian Naive Bayes |
| SGD Classifier |

---

## ⚙️ Variables derivadas (Feature Engineering)

- `total_screen_hours`: suma de todas las horas de pantalla
- `study_sleep_ratio`: relación estudio/sueño
- `sleep_deficit`: diferencia respecto a 8 horas ideales
- `recovery_ratio`: balance entre descanso activo y pantallas
- `task_overwhelm_index`: carga de tareas vs descansos
- `caffeine_per_study_hour`: intensidad de cafeína según horas de estudio
- Variables de interacción entre hábitos (pantalla, sueño, ejercicio, etc.)

---

## 📁 Estructura del proyecto

```
proyecto-aprendizaje-maquina/
│
├── app.py                          # Aplicación web con Streamlit
├── README.md
│
├── config/                         # Configuración del proyecto
│   └── requirements.txt
│
├── data/
│   ├── raw/                        # Datos originales
│   │   └── student_productivity_distraction_dataset.csv
│   └── prepared/                   # Datos procesados
│       └── datos_df.pkl
│
├── models/                         # Modelos entrenados y artefactos
│   ├── logistic_regression.joblib
│   ├── gaussian_nb.joblib
│   ├── sgd_classifier.joblib
│   ├── scaler.joblib
│   └── features.json
│
├── reports/                        # Resultados y visualizaciones
│   ├── metrics.json
│   ├── predict_metrics.json
│   ├── model_comparison.png
│   ├── cm_.png
│   ├── lc_.png
│   ├── roc_*.png
│   └── (otras gráficas exploratorias)
│
├── src/                            # Scripts de procesamiento y modelado
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
└── notebooks/                      # Análisis exploratorio y desarrollo
    ├── 1.ComprensionNegocio.ipynb
    └── 2.ComprensionDatos.ipynb
```

---

## ▶️ Flujo del proyecto

```
1. Carga y comprensión de datos   →  data_loader.py  +  notebooks
2. Preprocesamiento               →  preprocessing.py
3. Entrenamiento de modelos       →  train_model.py
4. Predicción                     →  predict.py
5. Evaluación y métricas          →  evaluate.py
6. Despliegue                     →  app.py  (streamlit run app.py)
```
""")

    if metricas:
        st.subheader("Métricas actuales")
        mejor_k = max(metricas, key=lambda k: metricas[k]["f1_score"])
        mejor_m = metricas[mejor_k]
        st.success(
            f"🏆 Mejor modelo: **{NOMBRES_MODELO.get(mejor_k, mejor_k)}** "
            f"— F1: {mejor_m['f1_score']:.4f} · Accuracy: {mejor_m['accuracy']:.4f}"
        )