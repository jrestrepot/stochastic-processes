from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

from mbe import (
    movimiento_browniano_bridge,
    movimiento_browniano_estandar,
    movimiento_browniano_geometrico,
    movimiento_segundo_punto,
)
from utils import simular_movimiento_y_computar_propiedades


# Solución al punto 1
def primer_punto(n, T, n_iter=10000):
    """Solución al primer punto.

    Parámetros
    ----------
    n : int
        Número de pasos.
    T : float
        Tiempo final.
    n_iter : int
        Número de iteraciones.
    """

    vector_de_resultados = []
    for _ in tqdm(range(n_iter), desc="Simulando primer punto..."):
        # Simula un movimiento browniano
        t, B = movimiento_browniano_estandar(n, T)
        # Calcula expresion e^[2Bt^2]
        Wt = np.exp(2 * B**2)
        vector_de_resultados.append(Wt)

    # Calcula el vector de medias
    vector_de_medias = np.mean(vector_de_resultados, axis=0)
    # Calcula funcion a la que converge cuando 0 <= t <= 1/4
    t_acotado = t[np.where(t <= 1 / 4)]
    funcion_a_converger = (1 - 4 * t_acotado) ** (-1 / 2)

    # Grafica
    plt.plot(t, vector_de_medias, label="Media de e^(2Bt^2)")
    plt.plot(t_acotado, funcion_a_converger, label="(1-4t)^(-1/2)")
    plt.xlabel("Tiempo")
    plt.title("Comparación de media de e^(2Bt^2) y (1-4t)^(-1/2)")
    plt.legend()
    # Guarda la gráfica
    plt.savefig("graficas/media_e_2Bt2_vs_1-4t.png")
    plt.close()


# Solución al punto 2
def segundo_punto(T, alpha, _lambda, sigma, n_iter=10000):
    """Simular un movimiento browniano geométrico, bridge y Wt, y obtener sus
    medias, varianzas, covarianzas y correlaciones.

    Parámetros
    ----------
    T : float
        Tiempo final.
    alpha : float
        Parámetro alpha (asociado a tendencia).
    _lambda : float
        Parámetro lambda (asociado a volatilidad).
    sigma : float
        Parámetro sigma para W.
    """

    n = 500
    t, geos = simular_movimiento_y_computar_propiedades(
        n=n,
        T=T,
        n_iter=n_iter,
        funcion=movimiento_browniano_geometrico,
        alpha=alpha,
        _lambda=_lambda,
    )
    _, bridges = simular_movimiento_y_computar_propiedades(
        n=n, T=T, n_iter=n_iter, funcion=movimiento_browniano_bridge
    )
    _, Ws = simular_movimiento_y_computar_propiedades(
        n=n, T=T, n_iter=n_iter, funcion=movimiento_segundo_punto, sigma=sigma
    )

    plt.plot(t, geos[0], label="GB(t)")
    plt.plot(t, bridges[0], label="B(t)")
    plt.plot(t, Ws[0], label="W(t)")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.title("Movimientos brownianos")
    plt.legend()
    plt.savefig("graficas/movimientos_brownianos_comp.png")
    plt.close()


# Solución al punto 3
def tercer_punto(n, T, n_iter=1000):
    """Demostrar que Bs-(s/t)*Bt (que en el código se llama Wts) y Bt son independientes.

    Parámetros
    ----------
    n : int
        Número de pasos.
    T : float
        Tiempo final.
    n_iter : int
        Número de iteraciones.
    """

    vector_Bt = []
    vector_Wts = []
    vector_Bs = []
    for _ in tqdm(range(n_iter), desc="Simulando tercer punto..."):
        # Simula un movimiento browniano
        t, B = movimiento_browniano_estandar(n, T)
        par_aleatorio = [np.random.randint(0, n - 1), np.random.randint(1, n)]
        s = min(par_aleatorio)
        t = max(par_aleatorio)
        # Calcula Bt
        Bt = B[t]
        # Calcula Wts
        Wts = B[s] - (s / t) * Bt
        vector_Bt.append(Bt)
        vector_Wts.append(Wts)
        vector_Bs.append(B[s])

    # Grafica
    plt.scatter(vector_Bt, vector_Wts)
    plt.xlabel("Bt")
    plt.ylabel("Wts")
    plt.title("Bt vs Wts")
    plt.savefig("graficas/Bt_vs_Wts.png")
    plt.close()

    # Calcula correlación
    correlacion = np.corrcoef(vector_Bt, vector_Wts)
    print("Correlación entre Bt y Wts:")
    pprint(correlacion)
    print("-------------------")

    mutual_information = mutual_info_regression(
        np.array(vector_Bt).reshape(-1, 1), np.array(vector_Wts).reshape(-1)
    )
    print("Información mutua entre Bt y Wts:")
    pprint(mutual_information[0])
    print("-------------------")
    print("")

    # Grafica Bt y Bs para comprobar correlacion entre Bt y Bs
    plt.scatter(vector_Bt, vector_Bs)
    plt.xlabel("Bt")
    plt.ylabel("Bs")
    plt.title("Bt vs Bs")
    plt.savefig("graficas/Bt_vs_Bs.png")
    plt.close()

    # Calcula correlación
    correlacion = np.corrcoef(vector_Bt, vector_Bs)
    print("Correlación entre Bt y Bs:")
    pprint(correlacion)
    print("-------------------")

    mutual_information = mutual_info_regression(
        np.array(vector_Bt).reshape(-1, 1), np.array(vector_Bs).reshape(-1)
    )
    print("Información mutua entre Bt y Wts:")
    pprint(mutual_information[0])
    print("-------------------")
    print("")
