import pandas as pd

class DataLoader:
    """Clase encargada de cargar los datos en formato DataFrame."""

    def __init__(self, ruta):
        self.ruta = ruta

    def cargar_datos(self):
        datos = pd.read_csv(self.ruta)
        print(f"Datos cargados correctamente: {datos.shape[0]} filas y {datos.shape[1]} columnas.")
        return datos