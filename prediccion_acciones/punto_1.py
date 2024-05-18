from concurrent.futures import ProcessPoolExecutor

import numpy as np

from analisis_acciones import escenarios_posibles, estimar_parametros, obtener_accion
from analisis_trayectorias import (
    analizar_log_normalidad,
    analizar_normalidad,
    autocorrelacion_parcial,
    distribucion_dimension_fractal,
    graficar,
    graficar_distribucion,
    simular_trayectorias,
    varianza_en_el_tiempo,
)
from simulacion_euler_maruyama import euler_maruyama_homogenea


def iteracion_sobre_parametros_homogenea(
    num_trayectorias: int = 10000, x0: float = 0.0, n: int = 500
) -> None:
    """
    Itera sobre diferentes combinaciones de mu y sigma y simula la EDE.

    Parametros
    ----------
    funcion : Callable
        Funcion que simula la EDE.
    num_trayectorias : int
        Numero de trayectorias a simular.
    x0 : float
        Valor inicial de x.
    n : int
        Numero de pasos.
    """

    mu_sigma_combinaciones = [
        (mu, sigma)
        for mu in np.linspace(0.0, 1, 3)
        for sigma in np.linspace(0.1, 0.4, 3)
    ]

    with ProcessPoolExecutor() as executor:
        results = {}
        # Simular en paralelo (mucho más rápido que en serie)
        for mu, sigma in mu_sigma_combinaciones:
            future = executor.submit(
                simular_trayectorias,
                euler_maruyama_homogenea,
                num_trayectorias,
                mu,
                sigma,
                x0,
                n,
            )
            results[(mu, sigma)] = future

        # Analizar y graficar resultados
        print("------------------------------------------------------")
        for (mu, sigma), future in results.items():
            trayectorias = future.result()
            trayectorias = np.array(trayectorias)
            graficar(trayectorias, mu, sigma, n)
            graficar_distribucion(trayectorias, mu, sigma, n)
            analizar_log_normalidad(trayectorias, mu, sigma, n, sample_size=5000)
            rendimientos = np.log(trayectorias[:, 1:] / trayectorias[:, :-1])
            analizar_normalidad(rendimientos, mu, sigma, len(rendimientos))
            varianza_en_el_tiempo(trayectorias, x0, mu, sigma, n)
            autocorrelacion_parcial(trayectorias, mu, sigma)
            distribucion_dimension_fractal(trayectorias, mu, sigma)
        print("------------------------------------------------------")
        print()


def obtener_y_analizar_accion(año: int):
    """
    Obtiene y analiza los datos de una accion que cumple con los requisitos.

    Parametros
    ----------
    año : int
        Año de la accion.
    """
    precios, accion = obtener_accion(año)

    mu, sigma = estimar_parametros(precios, 1 / len(precios))
    escenarios_posibles(precios, mu, sigma, accion, año, 1000)
