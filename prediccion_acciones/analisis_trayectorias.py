from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import pacf


def graficar(trayectorias: np.ndarray, mu: float, sigma: float, n: int) -> None:
    """
    Grafica las trayectorias simuladas por el metodo de Euler Maruyama.

    Parametros
    ----------
    trayectorias : np.ndarray
        Lista de trayectorias simuladas.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    n : int
        Numero de pasos.
    """

    for x in trayectorias:
        plt.plot(np.linspace(0, 1, n), x, alpha=0.5)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Aproximación por Euler Maruyama de mu={mu}, sigma={sigma}")
    plt.savefig(f"figures/euler_maruyama_{mu}_{sigma}.png")
    plt.clf()


def graficar_distribucion(
    trayectorias: list[list], mu: float, sigma: float, n: int
) -> None:
    """
    Grafica la distribucion de las trayectorias simuladas
    por el metodo de Euler Maruyama.

    Parametros
    ----------
    trayectorias : list
        Lista de trayectorias simuladas.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    n : int
        Numero de pasos.
    """

    min_y = np.min(trayectorias)
    max_y = np.max(trayectorias)
    y = np.linspace(min_y, max_y, n // 20)
    # Contar ocurrencias de cada valor en el grid (usandolos como bins)
    z = np.zeros((n // 20, n))
    for trayectoria in trayectorias:
        # Digitize retorna el índice del bin en el que cayó cada valor
        inds = np.digitize(trayectoria, y) - 1
        for idx, val in enumerate(inds):
            try:
                z[n // 20 - 1 - val, idx] += 1
            except:
                print(val, idx)
    # Escalar z
    z = z / len(trayectorias)

    plt.imshow(z, aspect="auto", extent=[0, 1, min_y, max_y], cmap="hot")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Distribucion de trayectorias de mu={mu}, sigma={sigma}")
    plt.savefig(f"figures/distribucion_{mu}_{sigma}.png")
    plt.clf()


def analizar_log_normalidad(
    trayectorias: np.ndarray, mu: float, sigma: float, n: int, sample_size: int
) -> None:
    """
    Analiza la distribucion en cada punto de las trayectorias simuladas
    por el metodo de Euler Maruyama.

    Parametros
    ----------
    trayectorias : np.ndarray
        Lista de trayectorias simuladas.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    n : int
        Numero de pasos.
    """

    if len(trayectorias) > sample_size:
        indices = np.random.choice(trayectorias.shape[0], sample_size, replace=False)
        trayectorias = trayectorias[indices]

    trayectorias_t = trayectorias.T

    # Test Shapiro-Wilk log normal
    resultados_log = np.zeros(n)
    for i, punto in enumerate(trayectorias_t):
        if np.any(punto <= 0):
            continue
        _, p = stats.shapiro(np.log(punto))
        if p < 0.05:
            resultados_log[i] += 1
    perc_rechazo = sum(resultados_log) * 100 / n
    if len(resultados_log) == 0:
        msg = "No aplica test de Shapiro-Wilk"
    else:
        msg = f"El test de Shapiro-Wilk rechazó la hipótesis el {perc_rechazo}% de las veces"
    label = f" para la lognormalidad cuando mu={mu}, sigma={sigma}"

    print(msg + label)


def analizar_normalidad(
    rendimientos: np.ndarray, mu: float, sigma: float, n: int
) -> float:
    """
    Analiza la distribucion de los rendimientos de las trayectorias simuladas
    por el metodo de Euler Maruyama.
    NOTA: No analiza la distributción en cada punto sino a lo largo de la trayectoria.

    Parametros
    ----------
    trayectorias : np.ndarray
        Lista de trayectorias simuladas.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    n : int
        Numero de trayectorias a analizar.

    Retorna:
    --------
    perc_rechazo : float
        Porcentaje de rechazo del test de Shapiro-Wilk.
    """

    resultados_shapiro = np.zeros(n)
    for i, trayectoria in enumerate(rendimientos):
        _, p = stats.shapiro(trayectoria)
        if p < 0.05:
            resultados_shapiro[i] += 1
    perc_rechazo = sum(resultados_shapiro) * 100 / n
    msg = f"El test de Shapiro-Wilk rechazó la hipótesis para los rendimientos el {perc_rechazo}% de las veces"
    label = f" para mu={mu}, sigma={sigma}"

    print(msg + label)

    return perc_rechazo


def varianza_en_el_tiempo(
    trayectorias: np.ndarray, x0: float, mu: float, sigma: float, n: int
) -> np.ndarray:
    """
    Calcula la varianza en el tiempo de las trayectorias simuladas.

    Parametros
    ----------
    trayectorias : np.ndarray
        Lista de trayectorias simuladas.
    x0 : float
        Valor inicial de x.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    n : int
        El número de pasos.

    Retorna
    -------
    varianzas : np.ndarray
        Varianza en el tiempo de las trayectorias simuladas.
    """

    varianzas = np.var(trayectorias, axis=0)
    t = np.linspace(0, 1, n)
    varianzas_teoricas = x0 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)
    plt.plot(varianzas)
    plt.plot(varianzas_teoricas)
    plt.legend(["Varianza simulada", "Varianza teorica"])
    plt.xlabel("t")
    plt.ylabel("Varianza")
    plt.title(f"Varianza en el tiempo mu = {mu}, sigma = {sigma}")
    plt.savefig(f"figures/varianza_tiempo_{mu}_{sigma}.png")
    plt.clf()


def autocorrelacion_parcial(trayectorias, mu, sigma) -> None:
    """
    Grafica la autocorrelacion parcial de las trayectorias simuladas.

    Parametros
    ----------
    trayectorias : np.ndarray
        Lista de trayectorias simuladas.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    """

    pacf_values = [pacf(trayectoria, nlags=20) for trayectoria in trayectorias]
    pacf_values = np.array(pacf_values)
    mean_pacf = np.mean(pacf_values, axis=0)
    meadian_pacf = np.median(pacf_values, axis=0)
    plt.plot(mean_pacf, "o", alpha=0.5, color="blue")
    plt.plot(meadian_pacf, "o", alpha=0.5, color="green")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelacion parcial")
    plt.legend(["Media", "Mediana"])
    plt.title(f"Autocorrelaciones parciales de mu={mu}, sigma={sigma}")
    plt.savefig(f"figures/autocorrelacion_parcial_{mu}_{sigma}.png")
    plt.clf()


def distribucion_dimension_fractal(
    trayectorias: np.ndarray, mu: float, sigma: float
) -> None:
    """
    Grafica la distribucion de la dimensión fractal de las trayectorias simuladas.

    Parametros
    ----------
    trayectorias : np.ndarray
        Lista de trayectorias simuladas.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    """

    dimensiones = []
    for trayectoria in trayectorias:
        dimension = fractal_vol(trayectoria)
        dimensiones.append(dimension)

    plt.hist(dimensiones, bins=20)
    plt.xlabel("Dimension fractal")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribucion de la dimension fractal de mu={mu}, sigma={sigma}")
    plt.savefig(f"figures/distribucion_dimension_fractal_{mu}_{sigma}.png")
    plt.clf()


def fractal_vol(data: np.ndarray) -> Tuple[float, float]:
    """
    Función que calcula la dimensión fractal para una trayectoria dada.

    Parametros
    ----------
    data : np.ndarray
        Un arreglo de numpy con 2 dimensiones. La primera columna es el tiempo y
        la segunda es el valor de la trayectoria.
    """

    # Asegurarse de que los datos estén en la forma correcta
    if len(data.shape) == 1:
        data = np.vstack((np.arange(len(data)), data)).T
    max1, min1 = np.max(data[:, 0]), np.min(data[:, 0])
    max2, min2 = np.max(data[:, 1]), np.min(data[:, 1])
    # Normalizar todos a la unidad cuadrada
    normdata = data.copy()
    normdata[:, 0] = (normdata[:, 0] - min1) / (max1 - min1)
    normdata[:, 1] = (normdata[:, 1] - min2) / (max2 - min2)

    # Asegurarse de que nada caiga a través del eje x
    minwidth = np.log2(np.min(np.diff(normdata[:, 0])))
    minwidth = np.abs(np.ceil(minwidth)) - 1
    n = np.zeros(int(minwidth), dtype=int)

    for j in range(int(minwidth)):
        width = 2**-j
        xaxis_pos = 0
        boxcount = 0

        while xaxis_pos < 1:
            indx = (xaxis_pos <= normdata[:, 0]) & (normdata[:, 0] < xaxis_pos + width)
            if 1 - xaxis_pos == width:
                indx[-1] = True

            vertical_column = normdata[indx, 1]

            if len(vertical_column) == 1:
                boxcount += 1
            else:
                rawcount = (np.max(vertical_column) - np.min(vertical_column)) / width
                rawcount += np.remainder(np.min(vertical_column), width)
                count = np.ceil(rawcount)
                boxcount += count

            xaxis_pos += width  # Avanzar en el eje x

        n[j] = boxcount

    r = np.power(2, -(np.arange(1, minwidth + 1)))
    s = -np.gradient(np.log(n)) / np.gradient(np.log(r))
    IQR = (
        np.percentile(s, [25, 75], interpolation="midpoint")[1]
        - np.percentile(s, [25, 75], interpolation="midpoint")[0]
    )
    indx2 = np.abs(s - np.median(s)) > IQR / 2
    x2 = np.log(r)
    y2 = np.log(n)
    s = s[~indx2]
    x2 = x2[~indx2]
    y2 = y2[~indx2]

    # std es la varianza de la pendiente según la teoría de OLS
    X = np.column_stack((np.ones_like(x2), x2))
    beta = np.linalg.pinv(X) @ y2
    C = np.linalg.inv(X.T @ X)
    e = y2 - X @ np.linalg.pinv(X) @ y2
    dimension = -beta[1]

    return dimension


def simular_trayectorias(
    funcion, num_trayectorias: int, mu: float, sigma: float, x0: float, n: int
) -> list[list]:
    """
    Simula un numero de trayectorias de una EDE dada por mu y sigma.

    Parametros
    ----------
    funcion : Callable
        Funcion que simula la EDE.
    num_trayectorias : int
        Numero de trayectorias a simular.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    x0 : float
        Valor inicial de x.
    n : int
        Numero de pasos.

    Retorna
    -------
    trayectorias : list
        Lista de trayectorias simuladas.
    """

    trayectorias = [funcion(mu, sigma, x0, n) for _ in range(num_trayectorias)]
    return trayectorias
