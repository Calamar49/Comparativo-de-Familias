"""Un Pipeline por modelo: el escalado va DENTRO del pipeline para no filtrar datos
y para que, al exportar a ONNX, el escalado quede incluido en el grafo."""
from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from familia3 import SEED


def pipeline_naive_bayes(var_smoothing: float = 1e-9) -> Pipeline:
    """NB Gaussiano no necesita escalado para predecir bien (es invariante por atributo),
    pero se deja un pipeline explícito para mantener la misma interfaz que los demás."""
    return Pipeline(
        [
            ("clf", GaussianNB(var_smoothing=var_smoothing)),
        ]
    )


def pipeline_mlp(
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    solver: str = "adam",
    batch_size: int | str = "auto",
    early_stopping: bool = True,
    seed: int = SEED,
) -> Pipeline:
    """MLP sensible a la escala: el StandardScaler va dentro del pipeline, así el
    .onnx exportado incluye el escalado y la web no tiene que reimplementarlo."""
    return Pipeline(
        [
            ("escalado", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    solver=solver,
                    batch_size=batch_size,
                    early_stopping=early_stopping,
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def kernel_rbf(longitud_escala: float = 1.0, ruido: float = 1.0):
    return RBF(length_scale=longitud_escala) + WhiteKernel(noise_level=ruido)


def kernel_matern(longitud_escala: float = 1.0, nu: float = 1.5, ruido: float = 1.0):
    return Matern(length_scale=longitud_escala, nu=nu) + WhiteKernel(noise_level=ruido)


def pipeline_gpr(kernel=None, alpha: float = 1e-10, seed: int = SEED) -> Pipeline:
    """GPR: el escalado de entradas se deja fuera a propósito para poder leer las
    longitudes de escala del núcleo en unidades originales (relevancia automática).

    normalize_y=False a propósito: skl2onnx 1.20 tiene un error conocido al exportar
    GaussianProcessRegressor con normalize_y=True + return_std=True (falla la conversión
    con un TypeError interno en el operador Mul). Se documenta como limitación técnica
    en el informe; no afecta la validez matemática, solo evita centrar/escalar y antes
    de ajustar (el GP sigue siendo correcto con un prior de media cero)."""
    if kernel is None:
        kernel = kernel_rbf()
    return Pipeline(
        [
            (
                "clf",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=False,
                    n_restarts_optimizer=5,
                    random_state=seed,
                ),
            )
        ]
    )
