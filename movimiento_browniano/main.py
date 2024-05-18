import warnings

from solucion import primer_punto, segundo_punto, tercer_punto

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    n = 500
    primer_punto(n, 1 / 4 - 10e-5, 100000)  # Le restamos 10e-5 porque 1/4 es asíntota
    segundo_punto(
        10,
        0.5,
        0.2,
        0.5,
        n_iter=10000,
    )
    tercer_punto(n, 1, 10000)
