"""
Análisis exploratorio de datos - Student Productivity & Distraction Dataset
Script unificado a partir de notebook .ipynb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración general de gráficos
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Directorio de salida para reportes/gráficos
output_dir = '../reports'
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Carga de datos
# ----------------------------------------------------------------------
df = pd.read_csv('../data/raw/student_productivity_distraction_dataset_20000.csv')

print("=== Primeras 10 columnas del dataset ===")
print(df.iloc[:, :10].head())

print("\n=== Columnas del dataset ===")
print(df.columns.tolist())

print("\n=== Tipos de datos ===")
print(df.dtypes)

print("\n=== Dimensiones del dataset ===")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# ----------------------------------------------------------------------
# 2. Valores nulos
# ----------------------------------------------------------------------
print("\n=== Valores nulos en variables numéricas ===")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
null_counts_num = df[numeric_cols].isnull().sum()
null_num_df = pd.DataFrame({
    'Variable': null_counts_num.index,
    'Nulos': null_counts_num.values,
    '% Nulos': (null_counts_num.values / len(df) * 100).round(2)
})
print(null_num_df)

print("\n=== Valores nulos en variables categóricas (object) ===")
cat_cols = df.select_dtypes(include=['object']).columns
null_counts_cat = df[cat_cols].isnull().sum()
null_cat_df = pd.DataFrame({
    'Variable': null_counts_cat.index,
    'Nulos': null_counts_cat.values,
    '% Nulos': (null_counts_cat.values / len(df) * 100).round(2)
})
print(null_cat_df)

# ----------------------------------------------------------------------
# 3. Duplicados
# ----------------------------------------------------------------------
duplicados = df.duplicated().sum()
print(f"\n=== Filas duplicadas: {duplicados} ===")

# ----------------------------------------------------------------------
# 4. Distribución de variables numéricas (histogramas + KDE)
# ----------------------------------------------------------------------
print("\n=== Generando histogramas de variables numéricas ===")
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

# Eliminar subplots vacíos
for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
save_path = os.path.join(output_dir, 'distribuciones_variables_numericas.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"Guardado: {save_path}")

# ----------------------------------------------------------------------
# 5. Boxplots de variables numéricas
# ----------------------------------------------------------------------
print("\n=== Generando boxplots de variables numéricas ===")
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_ylabel(col)

for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
save_path = os.path.join(output_dir, 'boxplots_variables_numericas.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"Guardado: {save_path}")

# ----------------------------------------------------------------------
# 6. Distribución de variables categóricas (countplots)
# ----------------------------------------------------------------------
print("\n=== Generando countplots de variables categóricas ===")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(cat_cols) > 0:
    fig, axes = plt.subplots(nrows=len(cat_cols), ncols=1, figsize=(10, 5 * len(cat_cols)))
    if len(cat_cols) == 1:
        axes = [axes]

    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'distribuciones_variables_categoricas.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Guardado: {save_path}")
else:
    print("No hay variables categóricas de tipo object en el dataset.")

# Mostrar frecuencias de cada variable categórica
for col in cat_cols:
    print(f"\nFrecuencias de '{col}':")
    print(df[col].value_counts())

# ----------------------------------------------------------------------
# 7. Distribución específica de stress_level (variable objetivo)
# ----------------------------------------------------------------------
print("\n=== Distribución de stress_level ===")
plt.figure(figsize=(8, 4))
sns.histplot(df['stress_level'], bins=20, kde=True)
plt.title('Distribución de Stress Level')
plt.tight_layout()
save_path = os.path.join(output_dir, 'distribucion_stress_level.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"Guardado: {save_path}")

print("Tabla de frecuencias de stress_level:")
print(df['stress_level'].value_counts().sort_index())

# ----------------------------------------------------------------------
# 8. Detección de outliers (método IQR)
# ----------------------------------------------------------------------
print("\n=== Análisis de outliers (IQR) ===")
def count_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

outliers_summary = []
for col in numeric_cols:
    n_outliers, lb, ub = count_outliers_iqr(df, col)
    outliers_summary.append({
        'Variable': col,
        'N° Outliers': n_outliers,
        '% Outliers': round(100 * n_outliers / len(df), 2),
        'Límite inferior': round(lb, 2),
        'Límite superior': round(ub, 2)
    })

outliers_df = pd.DataFrame(outliers_summary)
print(outliers_df)

# ----------------------------------------------------------------------
# 9. Correlaciones con stress_level
# ----------------------------------------------------------------------
print("\n=== Correlación de variables numéricas con stress_level ===")
# Excluir 'student_id' si existe
numeric_for_corr = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'student_id' in numeric_for_corr:
    numeric_for_corr.remove('student_id')

# Pearson y Spearman
pearson_corr = df[numeric_for_corr].corrwith(df['stress_level']).sort_values(ascending=False)
spearman_corr = df[numeric_for_corr].corrwith(df['stress_level'], method='spearman').sort_values(ascending=False)

corr_df = pd.DataFrame({
    'Pearson': pearson_corr,
    'Spearman': spearman_corr
})
print(corr_df)

# Gráfico de barras horizontales (Pearson)
corr_with_stress = pearson_corr.drop('stress_level', errors='ignore').sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(corr_with_stress.index, corr_with_stress.values,
               color=['steelblue' if x > 0 else 'salmon' for x in corr_with_stress.values])
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coeficiente de correlación de Pearson')
ax.set_title('Correlación de cada variable con Stress Level')
ax.grid(axis='x', alpha=0.3)

for i, (val, name) in enumerate(zip(corr_with_stress.values, corr_with_stress.index)):
    ax.text(val + 0.01 * (1 if val >= 0 else -1), i, f'{val:.2f}',
            va='center', fontsize=9)

plt.tight_layout()
save_path = os.path.join(output_dir, 'correlacion_con_stress_level.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"Guardado: {save_path}")

# ----------------------------------------------------------------------
# 10. Matriz de correlación completa (Pearson)
# ----------------------------------------------------------------------
print("\n=== Generando matriz de correlación completa ===")
plt.figure(figsize=(14, 10))
corr_matrix = df[numeric_for_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Matriz de correlaciones (Pearson)')
plt.tight_layout()

save_path = os.path.join(output_dir, 'matriz_correlaciones.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"Guardado: {save_path}")

print("\n=== Análisis completado. Gráficos guardados en:", output_dir, "===")