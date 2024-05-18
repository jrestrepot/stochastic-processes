"""A modulo que contiene las funciones relacionadas al MBE."""

from typing import Callable

import numpy as np


def movimiento_browniano_estandar(n, T):
    """Simular un movimiento browniano.

    Parámetros
    ----------
    n : int
        Número de pasos.
    T : float
        Tiempo final.

    Retorna
    -------
    t : NumPy array
        Vector de tiempo.
    B : NumPy array
        Movimiento browniano B(t).
    """

    # Calcula el delta t
    dt = T / n
    # Crea un vector de ruido blanco
    dB = np.sqrt(dt) * np.random.randn(n)
    # Construye el movimiento browniano
    B = np.cumsum(dB)
    # Crea el vector de tiempo
    t = np.linspace(0, T, n)
    return t, B


def movimiento_browniano_geometrico(n, T, alpha, _lambda):
    """Simular un movimiento browniano geométrico.

    Parámetros
    ----------
    n : int
        Número de pasos.
    T : float
        Tiempo final.
    alpha : float
        Parámetro alpha (asociado a tendencia).
    _lambda : float
        Parámetro lambda (asociado a volatilidad).

    Retorna
    -------
    t : NumPy array
        Vector de tiempo.
    Wt : NumPy array
        Movimiento browniano geométrico GB(t).
    """

    assert _lambda > 0
    t, B = movimiento_browniano_estandar(n, T)
    Wt = np.exp(alpha * t + _lambda * B)
    return t, Wt


def movimiento_browniano_bridge(
    n,
    _,
):
    """Simular un movimiento browniano bridge.

    Parámetros
    ----------
    n : int
        Número de pasos.

    Retorna
    -------
    t : NumPy array
        Vector de tiempo.
    W : NumPy array
        Movimiento browniano bridge W(t).
    """

    t, B = movimiento_browniano_estandar(n, 1)
    W = B - t * B[-1]
    return t, W


def movimiento_segundo_punto(n, T, sigma):
    """Simular el movimiento browniano que se pide en el segundo punto.

    Parámetros
    ----------
    n : int
        Número de pasos.
    T : float
        Tiempo final.
    sigma : float
        Parámetro sigma para W.
    """

    t, B = movimiento_browniano_estandar(n, T)
    W = B**3 - np.exp(sigma * B)
    return t, W


def obtener_varianza_teorica(funcion, t, **kwargs):

    # TODO: Implementar
    if funcion.__name__ == "movimiento_segundo_punto":
        sigma = kwargs["sigma"]
        return (
            15 * t**3
            - 2 * sigma * t**2 * np.exp(1 / 2 * sigma**2 * t) * (sigma**2 * t + 3)
            + np.exp(2 * sigma**2 * t)
            - np.exp(sigma**2 * t)
        )

    if funcion.__name__ == "movimiento_browniano_estandar":
        return t

    if funcion.__name__ == "movimiento_browniano_geometrico":
        return np.exp((2 * kwargs["alpha"] + kwargs["_lambda"] ** 2) * t) * (
            np.exp(kwargs["_lambda"] ** 2 * t) - 1
        )

    if funcion.__name__ == "movimiento_browniano_bridge":
        return t - t**2


def obtener_media_teorica(funcion, t, **kwargs):

    if funcion.__name__ == "movimiento_segundo_punto":
        return -np.exp(1 / 2 * kwargs["sigma"] ** 2 * t)

    if funcion.__name__ == "movimiento_browniano_estandar":
        return np.zeros(len(t))

    if funcion.__name__ == "movimiento_browniano_geometrico":
        return np.exp((kwargs["alpha"] + 1 / 2 * kwargs["_lambda"] ** 2) * t)

    if funcion.__name__ == "movimiento_browniano_bridge":
        return np.zeros(len(t))


def obtener_covarianza_teorica(funcion: Callable, t: np.ndarray, **kwargs):

    vf_covarianza_teorica = np.vectorize(_covarianza_teorica)
    covarianza_teorica = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        covarianza_teorica[i, :] = vf_covarianza_teorica(funcion, t[i], t, **kwargs)
    return covarianza_teorica


def _covarianza_teorica(funcion: Callable, t1: float, t2: float, **kwargs):
    s = min(t1, t2)
    t = max(t1, t2)
    if funcion.__name__ == "movimiento_segundo_punto":
        sigma = kwargs["sigma"]
        return (
            6 * s**3
            + 9 * s**2 * t
            - np.exp(1 / 2 * sigma**2 * t) * s**2 * sigma * (3 + s * sigma**2)
            - 3 * t * sigma * s * np.exp(1 / 2 * sigma**2 * s)
            + s**3 * sigma**3 * np.exp(1 / 2 * sigma**2 * s)
            + np.exp(1 / 2 * sigma**2 * (t - s)) * np.exp(2 * sigma**2 * s)
            - np.exp(1 / 2 * sigma**2 * (t + s))
        )
    if funcion.__name__ == "movimiento_browniano_estandar":
        return s

    if funcion.__name__ == "movimiento_browniano_geometrico":
        return np.exp((kwargs["alpha"] + 1 / 2 * kwargs["_lambda"] ** 2) * (t + s)) * (
            np.exp(kwargs["_lambda"] ** 2 * s) - 1
        )

    if funcion.__name__ == "movimiento_browniano_bridge":
        return s - s * t
