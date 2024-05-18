from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import shapiro
from statsmodels.graphics.tsaplots import plot_pacf

from analisis_trayectorias import fractal_vol
from simulacion_euler_maruyama import euler_maruyama_homogenea


def datos_historicos(stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene los datos historicos de una acción.

    Parameters
    ----------
    stock_symbol : str
        Simbolo de la acción.
    start_date : str
        Fecha de inicio.
    end_date : str
        Fecha de fin.

    Returns
    -------
    pd.DataFrame
        Datos historicos de la acción.
    """

    stock = yf.Ticker(stock_symbol)
    return stock.history(start=start_date, end=end_date)


def obtener_sp500_tickers() -> list[str]:
    """
    Obtiene los tickers de las acciones del S&P 500 leyendo la tabla de la
    pagina de Wikipedia.

    Returns
    -------
    list[str]
        Lista de tickers del S&P 500.
    """

    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0]
    tickers = df["Symbol"].tolist()
    return tickers


def obtener_accion(año: int) -> Tuple[np.ndarray, str]:
    """
    Obtiene una acción del S&P 500 que pueda ser modelada por una EDE lineal
    homogenea

    Parameters
    ----------
    año : int
        Año de los datos.

    Returns
    -------
    np.ndarray
        Precios de la acción.
    str
        Nombre de la acción.
    """

    # Lista del S&P 500
    acciones = obtener_sp500_tickers()

    for accion in acciones:
        datos = datos_historicos(accion, f"{año-1}-12-31", f"{año}-12-31")[
            "Close"
        ].values
        rendimientos = datos[1:] / datos[:-1]
        # Prueba de normalidad
        _, p_value = shapiro(rendimientos)
        if p_value > 0.05:
            print("------------------------------------------------------")
            print(f"Los rendimientos de la acción {accion} son normales")
            print(f"p-value: {p_value}")
            graficar_precios(datos, accion, "2021")
            graficar_histograma(rendimientos, accion, "2021")
            autocorrelacion_precios(datos, accion, "2021")
            dimesion_fractal = fractal_vol(datos)
            print(f"Dimension fractal: {dimesion_fractal}")
            print("------------------------------------------------------")
            print()
            return datos, accion


def graficar_histograma(rendimientos: np.ndarray, accion: str, año: str):
    """
    Grafica un histograma de los rendimientos de una acción.

    Parameters
    ----------
    rendimientos : np.ndarray
        Rendimientos de la acción.
    accion : str
        Nombre de la acción.
    año : str
        Año de los datos.
    """

    plt.hist(rendimientos, bins=50, density=True)
    plt.title(f"Histograma de rendimientos de {accion} en {año}")
    plt.xlabel("Rendimientos")
    plt.ylabel("Frecuencia")
    plt.savefig(f"figures/rendimientos_{accion}_{año}.png")
    plt.clf()


def autocorrelacion_precios(precios: np.ndarray, accion: str, año: str):
    """
    Grafica la autocorrelacion parcial de los precios de una acción.

    Parameters
    ----------
    precios : np.ndarray
        Precios de la acción.
    accion : str
        Nombre de la acción.
    año : str
        Año de los datos.
    """

    plt.figure(figsize=(10, 5))
    plot_pacf(precios, lags=20)
    plt.title(f"Autocorrelación parcial de los precios de {accion} en {año}")
    plt.savefig(f"figures/autocorrelacion_parcial_precios_{accion}_{año}.png")
    plt.clf()


def graficar_precios(precios: np.ndarray, accion: str, año: str):
    """
    Grafica los precios de una acción.

    Parameters
    ----------
    precios : np.ndarray
        Precios de la acción.
    accion : str
        Nombre de la acción.
    año : str
        Año de los datos.
    """

    plt.plot(precios)
    plt.title(f"Precios de {accion} en {año}")
    plt.xlabel("Dias")
    plt.ylabel("Precio")
    plt.savefig(f"figures/precios_{accion}_{año}.png")
    plt.clf()


def estimar_parametros(precios: np.ndarray, dt: float) -> tuple[float, float]:
    """
    Estima los parametros mu y sigma de una EDE lineal homogenea.

    Parameters
    ----------
    precios : np.ndarray
        Precios de la acción.

    Returns
    -------
    float
        Estimacion de mu.
    float
        Estimacion de sigma.
    """

    rendimientos = np.log(precios[1:] / precios[:-1])
    mu = np.mean(rendimientos) / dt
    sigma = np.sqrt(np.var(rendimientos) / dt)
    print("------------------------------------------------------")
    print(f"Estimacion de mu: {mu}")
    print(f"Estimacion de sigma: {sigma}")
    print("------------------------------------------------------")
    print()
    return mu, sigma


def pronosticar_accion(
    precios: np.ndarray,
    mu: float,
    sigma: float,
    n: int,
    iter: int,
    accion: str,
    año: int = 2021,
    escenario: str = "total",
) -> np.ndarray:
    """
    Pronostica los precios futuros de una acción mediante una EDE lineal homogenea.

    Parameters
    ----------
    precios : np.ndarray
        Precios de la acción.
    mu : float
        Coeficiente de tendencia.
    sigma : float
        Coeficiente de difusion.
    n : int
        Numero de pasos.
    iter : int
        Numero de iteraciones.
    accion : str
        Nombre de la acción.
    año : str
        Año de los datos.

    Returns
    -------
    np.ndarray
        Precios pronosticados.
    """

    pronosticos = np.zeros((iter, n))
    for i in range(iter):
        trayectoria = euler_maruyama_homogenea(mu, sigma, precios[-1], n)
        pronosticos[i] = trayectoria
    x_pronostico = np.arange(len(precios), len(precios) + n)
    x_precios = np.arange(len(precios))
    plt.plot(x_precios, precios, color="blue")
    plt.plot(x_pronostico, pronosticos.T, alpha=0.5)
    plt.title(f"Pronostico de precios de {accion} en {año} (Escenario {escenario})")
    plt.savefig(f"figures/pronosticos_{accion}_{año}_{escenario}.png")
    plt.clf()
    bandas = np.percentile(pronosticos, [10, 90], axis=0)
    plt.fill_between(x_pronostico, bandas[0], bandas[1], color="blue", alpha=0.5)
    plt.plot(x_pronostico, np.mean(pronosticos, axis=0), color="red")
    plt.plot(x_precios, precios, color="blue")
    plt.title(
        f"Pronostico de precios de {accion} en {año} con bandas de confianza al 80% (Escenario {escenario})",
        wrap=True,
    )
    plt.savefig(f"figures/bandas_{accion}_{año}_{escenario}.png")
    plt.clf()
    return pronosticos, bandas


def escenarios_posibles(
    precios: np.ndarray,
    mu: float,
    sigma: float,
    accion: str,
    año: str,
    iter: int,
):
    """
    Grafica las bandas de confianza de los precios pronosticados para los
    escenarios bajista, intermedio y alcista.

    Parameters
    ----------
    precios : np.ndarray
        Precios de la acción.
    pronosticos : np.ndarray
        Precios pronosticados.
    mu : float
        Coeficiente de tendencia.
    accion : str
        Nombre de la acción.
    año : str
        Año de los datos.
    """

    # Escenario bajista (bajamos el coeficiente de tendencia un 60%)
    pronosticos_bajista, bandas_bajistas = pronosticar_accion(
        precios,
        mu - 0.2 * mu,
        sigma,
        len(precios),
        iter,
        accion,
        año,
        escenario="bajista",
    )

    # Escenario intermedio
    pronosticos_intermedio, bandas_intermedias = pronosticar_accion(
        precios,
        mu,
        sigma,
        len(precios),
        iter,
        accion,
        año,
        escenario="intermedio",
    )

    # Escenario alcista (subimos el coeficiente de tendencia un 60%)
    pronosticos_alcista, bandas_alcistas = pronosticar_accion(
        precios,
        mu + 0.2 * mu,
        sigma,
        len(precios),
        iter,
        accion,
        año,
        escenario="alcista",
    )

    datos_out_of_sample = datos_historicos(accion, "2022-01-01", "2023-01-4")[
        "Close"
    ].values
    datos_totales = np.concatenate((precios, datos_out_of_sample))
    plt.plot(datos_totales, color="blue")
    plt.title(f"Precios reales de {accion} en {año} y {año+1}")
    plt.savefig(f"figures/precio_{accion}_{año}_{año+1}.png")

    # Comparar datos reales con los escenarios
    x_pronostico = np.arange(len(precios), len(precios) + len(datos_out_of_sample))
    x_precios = np.arange(len(precios))
    plt.plot(x_precios, precios, color="blue")
    plt.plot(x_pronostico, datos_out_of_sample, color="black")
    plt.fill_between(
        x_pronostico,
        bandas_bajistas[0],
        bandas_bajistas[1],
        color="red",
        alpha=0.5,
        label="Banda de confianza bajista",
    )
    plt.plot(x_pronostico, np.mean(pronosticos_bajista, axis=0), color="red")
    plt.fill_between(
        x_pronostico,
        bandas_intermedias[0],
        bandas_intermedias[1],
        color="blue",
        alpha=0.5,
        label="Banda de confianza intermedia",
    )
    plt.plot(x_pronostico, np.mean(pronosticos_intermedio, axis=0), color="blue")
    plt.fill_between(
        x_pronostico,
        bandas_alcistas[0],
        bandas_alcistas[1],
        color="green",
        alpha=0.5,
        label="Banda de confianza alcista",
    )
    plt.plot(x_pronostico, np.mean(pronosticos_alcista, axis=0), color="green")
    plt.legend()
    plt.title(
        f"Pronostico de precios de {accion} en {año+1} con bandas de confianza al 80%",
        wrap=True,
    )
    plt.savefig(f"figures/escenarios_{accion}_{año}_{año+1}.png")
    plt.clf()

    perc_bajista = np.mean(
        (datos_out_of_sample < bandas_bajistas[1]) * 1
        & (datos_out_of_sample > bandas_bajistas[0]) * 1
    )
    perc_alcista = np.mean(
        (datos_out_of_sample < bandas_alcistas[1]) * 1
        & (datos_out_of_sample > bandas_alcistas[0]) * 1
    )
    perc_intermedio = np.mean(
        (datos_out_of_sample < bandas_intermedias[1]) * 1
        & (datos_out_of_sample > bandas_intermedias[0]) * 1
    )
    print("-----------------------------------------------------------------")
    print(f"Porcentaje de datos reales en banda bajista: {perc_bajista}")
    print(f"Porcentaje de datos reales en banda intermedia: {perc_intermedio}")
    print(f"Porcentaje de datos reales en banda alcista: {perc_alcista}")
    print("-----------------------------------------------------------------")
