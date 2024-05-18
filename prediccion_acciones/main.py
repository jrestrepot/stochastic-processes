import warnings

warnings.filterwarnings("ignore")
from analisis_acciones import (
    escenarios_posibles,
    estimar_parametros,
    pronosticar_accion,
)
from punto_1 import iteracion_sobre_parametros_homogenea, obtener_y_analizar_accion

if __name__ == "__main__":

    # Itera sobre los parametros mu y sigma y simula la EDE (punto 1)
    iteracion_sobre_parametros_homogenea(x0=1)

    # Obtiene y analiza los datos de una accion que cumple con los requisitos
    obtener_y_analizar_accion(año=2021)
