"""
Análisis exploratorio de datos - Student Productivity & Distraction Dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(
    data_path: str = '../data/raw/student_productivity_distraction_dataset_20000.csv',
    output_dir: str = './reports'
) -> None:
    """
    Ejecuta el análisis exploratorio completo del dataset de productividad estudiantil.

    Parámetros
    ----------
    data_path  : ruta al archivo CSV de entrada.
    output_dir : directorio donde se guardarán los gráficos generados.
    """

    # --- Configuración general -------------------------------------------
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Carga de datos -----------------------------------------------
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH  = os.path.join(BASE_DIR, "../data/raw", "student_productivity_distraction_dataset_20000.csv")
    MODEL_DIR  = os.path.join(BASE_DIR, "../models")
    REPORT_DIR = os.path.join(BASE_DIR, "../reports")
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("📂 Cargando datos...")
    df = pd.read_csv(DATA_PATH)

    print("Primeras 10 columnas del dataset:")
    print(df.iloc[:, :10].head())
    print("\nColumnas:", df.columns.tolist())
    print("\nTipos de datos:\n", df.dtypes)
    print(f"\nDimensiones — Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    # --- 2. Valores nulos ------------------------------------------------
    print("\n=== [2] Valores nulos ===")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols     = df.select_dtypes(include=['object']).columns

    for label, cols in [("numéricas", numeric_cols), ("categóricas", cat_cols)]:
        counts = df[cols].isnull().sum()
        null_df = pd.DataFrame({
            'Variable': counts.index,
            'Nulos':    counts.values,
            '% Nulos': (counts.values / len(df) * 100).round(2)
        })
        print(f"\nVariables {label}:\n{null_df}")

    # --- 3. Duplicados ---------------------------------------------------
    print(f"\n=== [3] Filas duplicadas: {df.duplicated().sum()} ===")

    # --- 4. Histogramas + KDE de variables numéricas --------------------
    print("\n=== [4] Histogramas de variables numéricas ===")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')

    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    _save(fig, REPORT_DIR, 'distribuciones_variables_numericas.png')

    # --- 5. Boxplots de variables numéricas -----------------------------
    print("\n=== [5] Boxplots de variables numéricas ===")
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot de {col}')
        axes[i].set_ylabel(col)

    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    _save(fig, REPORT_DIR, 'boxplots_variables_numericas.png')

    # --- 6. Countplots de variables categóricas -------------------------
    print("\n=== [6] Countplots de variables categóricas ===")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if cat_cols:
        fig, axes = plt.subplots(nrows=len(cat_cols), ncols=1,
                                 figsize=(10, 5 * len(cat_cols)))
        if len(cat_cols) == 1:
            axes = [axes]

        for i, col in enumerate(cat_cols):
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f'Distribución de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        _save(fig, REPORT_DIR, 'distribuciones_variables_categoricas.png')

        for col in cat_cols:
            print(f"\nFrecuencias de '{col}':\n{df[col].value_counts()}")
    else:
        print("No hay variables categóricas de tipo object.")

    # --- 7. Distribución de stress_level --------------------------------
    print("\n=== [7] Distribución de stress_level ===")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df['stress_level'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribución de Stress Level')
    plt.tight_layout()
    _save(fig, REPORT_DIR, 'distribucion_stress_level.png')

    print("Frecuencias:\n", df['stress_level'].value_counts().sort_index())

    # --- 8. Outliers (IQR) ----------------------------------------------
    print("\n=== [8] Análisis de outliers (IQR) ===")
    outliers_summary = []
    for col in numeric_cols:
        n_out, lb, ub = _count_outliers_iqr(df, col)
        outliers_summary.append({
            'Variable':        col,
            'N° Outliers':     n_out,
            '% Outliers':      round(100 * n_out / len(df), 2),
            'Límite inferior': round(lb, 2),
            'Límite superior': round(ub, 2)
        })
    print(pd.DataFrame(outliers_summary))

    # --- 9. Correlaciones con stress_level ------------------------------
    print("\n=== [9] Correlación con stress_level ===")
    numeric_for_corr = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'student_id' in numeric_for_corr:
        numeric_for_corr.remove('student_id')

    pearson_corr  = df[numeric_for_corr].corrwith(df['stress_level']).sort_values(ascending=False)
    spearman_corr = df[numeric_for_corr].corrwith(df['stress_level'], method='spearman').sort_values(ascending=False)

    print(pd.DataFrame({'Pearson': pearson_corr, 'Spearman': spearman_corr}))

    corr_with_stress = pearson_corr.drop('stress_level', errors='ignore').sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(corr_with_stress.index, corr_with_stress.values,
            color=['steelblue' if x > 0 else 'salmon' for x in corr_with_stress.values])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coeficiente de correlación de Pearson')
    ax.set_title('Correlación de cada variable con Stress Level')
    ax.grid(axis='x', alpha=0.3)

    for i, val in enumerate(corr_with_stress.values):
        ax.text(val + 0.01 * (1 if val >= 0 else -1), i, f'{val:.2f}',
                va='center', fontsize=9)

    plt.tight_layout()
    _save(fig, REPORT_DIR, 'correlacion_con_stress_level.png')

    # --- 10. Matriz de correlación completa -----------------------------
    print("\n=== [10] Matriz de correlación completa ===")
    fig, ax = plt.subplots(figsize=(14, 10))
    corr_matrix = df[numeric_for_corr].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Matriz de correlaciones (Pearson)')
    plt.tight_layout()
    _save(fig, REPORT_DIR, 'matriz_correlaciones.png')

    print(f"\n=== Análisis completado. Gráficos guardados en: {REPORT_DIR} ===")


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, REPORT_DIR: str, filename: str) -> None:
    """Guarda y cierra una figura."""
    path = os.path.join(REPORT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Guardado: {path}")


def _count_outliers_iqr(df: pd.DataFrame, col: str) -> tuple:
    """Devuelve (n_outliers, límite_inferior, límite_superior) usando el método IQR."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = int(((df[col] < lb) | (df[col] > ub)).sum())
    return n_out, lb, ub