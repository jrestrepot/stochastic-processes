"""Un modulo para simular distintas EDES por el metodo de Euler-Maruyama."""

from typing import Callable

import numpy as np


def euler_maruyama_homogenea(mu: float, sigma: float, x0: float, n: int) -> None:
    """Aproxima la solucion de una ecuacion diferencial estocastica homogenea
    mediante el metodo de Euler-Maruyama.

    Parametros
    ----------
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    x0 : float
        Valor inicial de x.
    n : int
        Numero de pasos.
    """

    dt = 1 / n
    x = np.zeros(n)
    # El delta de W puede ser simulado como una normal con media 0 y
    # desviacion estandar dt
    dw = np.random.randn(n) * np.sqrt(dt)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i - 1] + mu * x[i - 1] * dt + sigma * x[i - 1] * dw[i]
    return x


def euler_maruyama_sentido_estricto(
    mu: Callable, sigma: Callable, x0: float, n: int
) -> None:
    """Aproxima la solucion de una ecuacion diferencial estocastica homogenea
    mediante el metodo de Euler-Maruyama.

    Parametros
    ----------
    mu : Callable
        Funcion de tendencia (mu(x,t))
    sigma : Callable
        Funcion de difusion (sigma(t))
    x0 : float
        Valor inicial de x.
    n : int
        Numero de pasos.
    """

    dt = 1 / n
    x = np.zeros(n)
    # El delta de W puede ser simulado como una normal con media 0 y
    # desviacion estandar dt
    dw = np.random.randn(n) * np.sqrt(dt)
    t = np.linspace(0, 1, n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i - 1] + mu(x[i - 1], t[i - 1]) * dt + sigma(t[i - 1]) * dw[i]
    return x
