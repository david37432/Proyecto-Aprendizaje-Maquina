"""
main.py — Orquestador del pipeline de análisis de productividad estudiantil.

Ejecuta en orden todos los scripts del proyecto.
Uso:
    python main.py
    python main.py --data ruta/al/archivo.csv --output ruta/reportes
"""

import argparse
import sys
import traceback

from src.data_loader import run_eda
from src.preprocessing import run_feature_engineering
from src.train_model import run_training
from src.predict import run_prediction
from src.evaluate import run_evaluation
# Cuando agregues más scripts, impórtalos aquí, por ejemplo:
# from feature_engineering import run_feature_engineering
# from model_training      import run_training
# from model_evaluation    import run_evaluation


# ---------------------------------------------------------------------------
# Definición del pipeline
# ---------------------------------------------------------------------------

def build_pipeline(data_path: str, output_dir: str) -> list:
    """
    Retorna la lista ordenada de pasos del pipeline.
    Cada paso es un dict con:
        - name     : nombre descriptivo del paso
        - fn       : función a ejecutar (callable)
        - kwargs   : argumentos que recibe la función
    """
    return [
        {
            "name":   "Análisis Exploratorio de Datos (EDA)",
            "fn":     run_eda,
            "kwargs": {"data_path": data_path, "output_dir": output_dir},
        },
        {
            "name":   "Ingeniería de Características",
            "fn":     run_feature_engineering,
            "kwargs": {"data_path": data_path, "output_dir": output_dir},
        },
        {
            "name":   "Entrenamiento del Modelo",          
            "fn":     run_training,
            "kwargs": {"output_dir": output_dir},
        },
        {
            "name":   "Predicción",          
            "fn":     run_prediction,
            "kwargs": {"output_dir": output_dir},
        },  
        {
            "name":   "Evaluación del Modelo",
            "fn":     run_evaluation,
            "kwargs": {"output_dir": output_dir},
        }
        ]


# ---------------------------------------------------------------------------
# Ejecución del pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, output_dir: str) -> None:
    pipeline = build_pipeline(data_path, output_dir)
    total    = len(pipeline)

    print("=" * 60)
    print("  INICIO DEL PIPELINE")
    print("=" * 60)

    for idx, step in enumerate(pipeline, start=1):
        print(f"\n[{idx}/{total}] {step['name']}")
        print("-" * 60)
        try:
            step["fn"](**step["kwargs"])
            print(f"[{idx}/{total}] ✓ Completado: {step['name']}")
        except Exception:
            print(f"\n[{idx}/{total}] ✗ ERROR en: {step['name']}")
            traceback.print_exc()
            sys.exit(1)           # Detiene el pipeline ante cualquier fallo

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis de productividad estudiantil"
    )
    parser.add_argument(
        "--data",
        default="../data/raw/student_productivity_distraction_dataset_20000.csv",
        help="Ruta al CSV de entrada (default: ../data/raw/...csv)",
    )
    parser.add_argument(
        "--output",
        default="../reports",
        help="Directorio de salida para reportes y gráficos (default: ../reports)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(data_path=args.data, output_dir=args.output)