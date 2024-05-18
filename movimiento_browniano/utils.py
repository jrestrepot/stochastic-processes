import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm

from mbe import (
    obtener_covarianza_teorica,
    obtener_media_teorica,
    obtener_varianza_teorica,
)


def _obtener_propiedades_distr(W):
    """Obtener las propiedades de la distribución del proceso estocástico W.

    Parámetros
    ----------
    W : NumPy array
        El vector de W.

    Retorna
    -------
    media : float
        Media de Wt.
    varianza : float
        Varianza de Wt.
    """

    W = np.array(W)
    medias = np.mean(W, axis=0)
    varianzas = np.var(W, axis=0)
    matrix_covarianza = np.cov(W.T)
    matrix_correlacion = np.corrcoef(W.T)
    return medias, varianzas, matrix_covarianza, matrix_correlacion


def _analizar_distribucion(W, funcion):
    """Analiza la distribución de W.

    Parámetros
    ----------
    W : NumPy array
        El vector de W.
    t : NumPy array
        Vector de tiempo.
    funcion : callable
        Función que simula un movimiento browniano.
    """

    _, ax = plt.subplots(1, 1)
    W = np.array(W).T
    sns.histplot(W[0], kde=True, stat="density", ax=ax)
    ax.set_title(f"Distribución de {funcion.__name__} en t=0")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    plt.savefig(f"graficas/distribucion_{funcion.__name__}.png")
    plt.close()

    # Test de normalidad de Shapiro-Wilk
    valor_cutoff = 0.05
    resultados_normal = []
    for i in range(len(W)):
        _, p = stats.shapiro(W[i])
        binario = 1 if p > 0.05 else 0
        resultados_normal.append(binario)

    prop = np.mean(resultados_normal)
    print("--------------------------------------------------------------------")
    print(
        f"La distribución de {funcion.__name__} es normal en {prop * 100}% de los casos."
    )
    if prop < 1 - valor_cutoff:
        print("La distribución no es normal.")
    else:
        print("La distribución es normal.")
    print("")

    resultados_beta = []
    # Solo la aplicamos al 3r movimiento porque se demora mucho
    if funcion.__name__ == "movimiento_segundo_punto":
        for i in range(len(W)):
            args = stats.beta.fit(W[i])
            _, p = stats.kstest(W[i], "beta", args=args)
            binario = 1 if p > 0.05 else 0
            resultados_beta.append(binario)
        prop = np.mean(resultados_beta)
        print(
            f"La distribución de {funcion.__name__} es beta en {prop * 100}% de los casos."
        )
        if prop < 1 - valor_cutoff:
            print("La distribución no es beta.")
        else:
            print("La distribución es beta.")
    print("")

    resultados_log_normal = []
    for i in range(len(W)):
        if np.any(W[i] <= 0):
            continue
        _, p = stats.shapiro(np.log(W[i]))
        binario = 1 if p > 0.05 else 0
        resultados_log_normal.append(binario)
    if len(resultados_log_normal) == 0:
        print(
            f"No se puede aplicar la prueba de log-normalidad para función {funcion.__name__}."
        )
        return
    prop = np.mean(resultados_log_normal)
    print(
        f"La distribución de {funcion.__name__} es log-normal en {prop * 100}% de los casos."
    )
    if prop < 1 - valor_cutoff:
        print("La distribución no es log-normal.")
    else:
        print("La distribución es log-normal.")
    print("--------------------------------------------------------------------")
    print("")


def _converge(funcion, t, trayectorias, **kwargs):
    """Determina si la simulación converge.

    Parámetros
    ----------
    funcion : callable
        Función que simula un movimiento browniano.
    t : np.ndarray
        Vector de tiempos.
    trayectorias : list
        Lista de trayectorias.

    Retorna
    -------
    converge : bool
        True si converge, False en otro caso.
    """

    medias, varianzas, _, _ = _obtener_propiedades_distr(trayectorias)

    if not np.mean(abs(medias - obtener_media_teorica(funcion, t, **kwargs))) < 0.005:
        return False
    if (
        not np.mean(abs(varianzas - obtener_varianza_teorica(funcion, t, **kwargs)))
        < 0.005
    ):
        return False
    return True


def simular_movimiento_y_computar_propiedades(n, T, n_iter, funcion, **kwargs):
    """Simula un movimiento browniano y computa sus propiedades.

    Parámetros
    ----------
    n : int
        Número de pasos.
    T : float
        Tiempo final.
    n_iter : int
        Número de iteraciones.
    funcion : callable
        Función que simula un movimiento browniano.
    kwargs : dict
        Parámetros adicionales para la función.

    Retorna
    -------
    t : np.ndarray
        Vector de tiempos.
    resultados : list
        Lista de resultados.
    """

    trayectorias = []
    for i in tqdm(range(n_iter), desc=f"Simulando {funcion.__name__}..."):
        t, B = funcion(n, T, **kwargs)
        trayectorias.append(B)
        plt.plot(t, B, alpha=0.5)
        if i % 100 == 0:
            if _converge(funcion, t, trayectorias, **kwargs):
                print(f"Converge en la iteración {i}.")
                break
    plt.title(f"Simulación de {funcion.__name__}")
    plt.xlabel("Tiempo")
    plt.savefig(f"graficas/simulacion_{funcion.__name__}.png")
    plt.close()

    # Calcula propiedades
    medias, varianzas, covarianzas, correlaciones = _obtener_propiedades_distr(
        trayectorias
    )

    _analizar_distribucion(trayectorias, funcion)

    # Graficar media
    plt.plot(t, medias, label="Media simulada")
    plt.plot(t, obtener_media_teorica(funcion, t, **kwargs), label="Media teórica")
    plt.title(f"Media de {funcion.__name__}")
    plt.xlabel("Tiempo")
    plt.legend()
    plt.savefig(f"graficas/media_vs{funcion.__name__}.png")
    plt.close()

    # Graficar varianza
    var_teorica = obtener_varianza_teorica(funcion, t, **kwargs)
    plt.plot(t, varianzas, label="Varianza simulada")
    plt.plot(t, var_teorica, label="Varianza teórica")
    plt.title(f"Varianza de {funcion.__name__}")
    plt.xlabel("Tiempo")
    plt.legend()
    plt.savefig(f"graficas/varianza_vs{funcion.__name__}.png")
    plt.close()

    # Graficar plano de la covarianza y el tiempo
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ta, tb = np.meshgrid(t, t)
    ax.plot_surface(ta, tb, covarianzas, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Tiempo A")
    ax.set_ylabel("Tiempo B")
    ax.set_zlabel("Covarianza")
    plt.savefig(f"graficas/covarianza_{funcion.__name__}.png")
    plt.close()

    # Graficar plano de la covarianza teórica y el tiempo
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ta, tb = np.meshgrid(t, t)
    cov_teorica = obtener_covarianza_teorica(funcion, t, **kwargs)
    ax.plot_surface(
        ta,
        tb,
        cov_teorica,
        cmap="viridis",
        edgecolor="none",
    )
    ax.set_xlabel("Tiempo A")
    ax.set_ylabel("Tiempo B")
    ax.set_zlabel("Covarianza")
    plt.savefig(f"graficas/covarianza_teo_{funcion.__name__}.png")
    plt.close()

    # Graficar heatmap de la correlación y el tiempo
    plt.imshow(correlaciones, cmap="viridis")
    plt.colorbar()
    plt.title(f"Correlaciones de {funcion.__name__}")
    plt.xlabel("Tiempo")
    plt.ylabel("Tiempo")
    plt.savefig(f"graficas/correlaciones_{funcion.__name__}.png")
    plt.close()

    return t, trayectorias
