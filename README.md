# Comprensión de Negocio

## Integrantes
- Harold Steven Vargas Heano
- David Eduardo Lopez Jimenez
## ¿Que problema quiero resolver? 
En el entorno escolar de hoy día, los alumnos lidian con muchas presiones provenientes de la cantidad de tareas, su uso de tecnología, dormir poco y no moverse mucho, además de otras cosas. El estar estresado a largo plazo perjudica no solo el bienestar mental y corporal, sino que también pega fuerte al desempeño escolar, a cuán productivos son y a si abandonan los estudios. No obstante, las escuelas normalmente notan el estrés una vez que ya se ve en malas notas o problemas de salud, y carecen de formas de anticipar que esto pase para poder ayudar a tiempo.
Esta iniciativa busca cubrir esa falta usando métodos de aprendizaje automático. Con un grupo de datos que tiene 20,000 registros de alumnos con detalles sobre sus rutinas (tiempo dedicado a estudiar, dormir, usar el móvil, redes sociales, juegos, hacer deporte, etcétera) y qué tan estresados dicen estar, el objetivo es poder adivinar el nivel de tensión de un alumno basándose en cómo vive y cómo se comporta en lo académico. Solucionar esto ayudaría a señalar a los estudiantes en peligro antes de que el estrés dañe mucho sus resultados, abriendo el camino para crear ayudas hechas a la medida (como sugerencias de costumbres sanas, ayuda psicológica temprana o cambios en sus deberes escolares).
## ¿Que objetivo tiene el grupo? 
La meta central de esta iniciativa es desarrollar un sistema de aprendizaje automático supervisado diseñado para calcular o categorizar el grado de tensión de un alumno, empleando como datos de predicción el resto de los campos presentes en la información recolectada (excluyendo, si aplica, claves o datos irrelevantes para el sistema). Este sistema debe ser formado, probado y medido siguiendo los lineamientos correctos del método CRISP-DM, asegurando que sea sólido, aplicable a nuevos casos y fácil de entender.

Los propósitos concretos que marcarán la labor son:

- Ejecutar un estudio inicial profundo de la información para entender cómo se reparten los valores, cómo se vinculan los campos, qué valores son extremos y qué tan buena es la calidad de los datos.
- Preparar los datos de forma apropiada (tratando valores faltantes, transformando campos, ajustando escalas, etcétera) para tener lista la información para crear el modelo.
- Elegir y capacitar varias estrategias de predicción o clasificación, dependiendo si el indicador de tensión es un número continuo o una categoría ordenada, comparando qué tan bien funcionan usando validación cruzada.
- Determinar cuáles elementos tienen mayor peso en la tensión mediante métodos de explicación (relevancia de atributos, valores SHaP, entre otros), ofreciendo no solo una herramienta para predecir sino también entendimientos útiles.
- Dejar constancia de todo el procedimiento y exponer los hallazgos de forma transparente, señalando las restricciones y las posibles formas de uso en ambientes de enseñanza.
## ¿Como medir el exito?
La valía del proyecto se medirá considerando dos ángulos que se complementan: el provecho para el negocio (el efecto en el ámbito de la enseñanza) y la calidad técnica (la solidez del modelo y su estudio). a continuación, se explican las pautas específicas:

### Beneficio empresarial (educativo)

- Identificación precoz: Se considerará un acierto si el modelo logra señalar correctamente al menos al 80 % de los alumnos con mucha tensión (sensibilidad de ≥ 0. 80 en el grupo más vulnerable), permitiendo a las escuelas enfocar sus apoyos.
- Utilidad práctica: aparte de pronosticar, el proyecto triunfará si se consiguen señalar factores de riesgo clave (como pasar demasiado tiempo en redes, dormir poco, o tener mucha tarea) que puedan explicarse de manera clara a consejeros y personal docente para guiarlos.
- Aplicabilidad general: El modelo debe ser lo suficientemente firme para usarse en grupos parecidos (estudiantes de universidad o preparatoria) sin perder mucha exactitud de repente.

### Logro técnico

La excelencia técnica se definirá según el tipo de modelo final, puesto que la magnitud de estrés podría manejarse como regresión (si es un valor numérico, por ejemplo del 1 al 10) o como clasificación (si está agrupada, como bajo/medio/alto). En cualquier escenario, se fijarán estos límites mínimos:

- Si es clasificación:
  - Exactitud general de ≥ 75 %
  - Precisión y Exhaustividad para el grupo de “mucho estrés” de ≥ 0. 80
  - Puntuación F1 (macro o ponderada) de ≥ 0. 75
  - Área bajo la curva ROC de ≥ 0. 85 para diferenciar entre los grupos.

- Si es regresión:
  - R² (coeficiente de determinación) de ≥ 0. 70
  - RMSE (raíz del error promedio al cuadrado) más bajo que un modelo base (como estimar el promedio).
  - MaE (error promedio absoluto) de ≤ 0. 8 si la escala va del 1 al 10.

además, se dará por bueno el proyecto si se cumplen estos aspectos de calidad en el proceso:

- Repetibilidad: Toda la secuencia (desde cargar los datos hasta la valoración) debe poder repetirse usando código bien anotado.
- Explicabilidad: Se requiere un estudio de qué tan importantes son las variables para poder justificar las predicciones.
- Contraste de modelos: Se evaluarán al menos tres formas algorítmicas distintas (por ejemplo, regresión lineal/logística, árboles de decisión/bosques aleatorios, XGBoost, y una red neuronal simple) eligiendo la mejor según las pautas marcadas.
## Metodología 
Metodología CRISP-DM (Cross-Industry Standard Process for Data Mining)
### Comprensión del negocio
Se аclаrа lа cuestión que se buscа solucionаr enfocándose en el ámbito del negocio o lа empresа. Se fijаn lаs metаs del proyecto, los pаrámetros pаrа considerаr un éxito (medidаs de negocio y tecnológicаs) y se creа un plаn preliminаr.

### Comprensión de los datos
Se juntаn los dаtos que hаcen fаltа, se revisаn sus pаrticulаridаdes (clаses, repаrtos, аusenciаs, vаlores extrаños), se observаn los vínculos entre lаs distintаs pаrtes y se compruebа que todo esté en orden. El propósito de este pаso es conocer bien lа informаción y hаllаr cuаlquier inconveniente de entrаdа.

### Preparación de los datos
Los dаtos iniciаles se prepаrаn pаrа poder usаrlos en modelos. Esto аbаrcа: depurаción (trаtаmiento de vаlores fаltаntes, extremos), аsignаción de números а lаs cаtegoríаs, аjuste de escаlаs, generаción de vаriаbles nuevаs (diseño de dаtos), elección de los rаsgos importаntes y sepаrаción en grupos pаrа entrenаr y verificаr.
### Modelado
Se escogen y empleаn métodos de modelаdo como regresión, clаsificаción o аgrupаmiento. Luego, se аjustаn los pаrámetros, se entrenаn los modelos con lа informаción disponible y se verificаn usаndo métodos como lа vаlidаción cruzаdа. Este pаso se repite, probаndo diferentes аlgoritmos y аjustes.

### Evaluación
Se evаlúаn los desempeños de los modelos con informаción de vаlidаción а trаvés de indicаdores pertinentes (exаctitud, precisión, exhаustividаd, F1, error cuаdrático medio, coeficiente de determinаción, etc. ). Se contrаstа lа аctuаción frente а los objetivos de éxito estаblecidos en lа etаpа comerciаl. De no cumplirse lаs metаs, se repаsаn lаs etаpаs previаs.

### Despliegue
El modelo se implementа pаrа su uso o se creаn informes, pаneles o sistemаs que аprovechаn los resultаdos. Esto аbаrcа lа documentаción, el seguimiento del modelo а lo lаrgo del tiempo y lаs estrаtegiаs de mаntenimiento.


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
## Instrucciones de ejecución 
```
1. Instalar dependencias "pip install -r ./config/requirements.txt
2. Ejecutar piplene main.py para procesar datos, entrenar modelos y generar reportes
3. Despliegue                     →  app.py  (streamlit run app.py)
```
## Evaluación de Modelos y Capacidad Predictiva

Durante la fase de modelado se evaluó la capacidad del dataset para predecir la variable objetivo **nivel de estrés (Stress Level)** utilizando distintos algoritmos de clasificación, entre ellos Regresión Logística, Gaussian Naive Bayes y SGD Classifier.

### Resultados obtenidos

| Modelo                 | Accuracy | F1 Score |
|----------------------|----------|----------|
| 🏆 Regresión Logística | 0.51     | 0.5016   |
| Gaussian Naive Bayes | 0.5068   | 0.4976   |
| SGD Classifier       | 0.5009   | 0.4036   |

### Análisis de resultados

A partir de estas métricas, se observa que todos los modelos presentan desempeños cercanos al azar (accuracy ≈ 0.5), lo cual indica una **baja capacidad predictiva del dataset respecto a la variable Stress Level**. Adicionalmente, los valores de F1 Score refuerzan esta conclusión, evidenciando un equilibrio deficiente entre precisión y recall.

Este comportamiento sugiere que las variables disponibles no contienen suficiente información relevante o patrones discriminativos que permitan modelar adecuadamente el nivel de estrés.

En contraste, durante el análisis exploratorio inicial se identificó que la variable **Productivity Score** presentaba correlaciones más significativas con varias características del dataset, lo que indicaba un mayor potencial predictivo.

### Conclusiones

- El dataset **no es adecuado para predecir el nivel de estrés** con los atributos actuales.
- Existe una **limitación en la representatividad o calidad de las variables explicativas** respecto a esta variable objetivo.
- La selección de la variable objetivo es un factor crítico en el desempeño del modelo, incluso más determinante que el algoritmo utilizado.

### Recomendaciones

- Reorientar el problema hacia variables con mayor capacidad predictiva, como **Productivity Score**.
- Enriquecer el dataset con nuevas variables que capturen de manera más directa los factores asociados al estrés.