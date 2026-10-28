"""Carga de datasets, particiones estratificadas y datos sintéticos, todo con semilla fija."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.model_selection import train_test_split

from familia3 import SEED


@dataclass
class Particion:
    """Contenedor simple para un split train/test ya hecho."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    nombres_atributos: list[str]
    nombres_clases: list[str] | None = None


def cargar_breast_cancer(seed: int = SEED) -> Particion:
    """Breast Cancer Wisconsin (569x30, binario). Caso principal de clasificación."""
    datos = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        datos.data,
        datos.target,
        test_size=0.2,
        random_state=seed,
        stratify=datos.target,
    )
    return Particion(X_train, X_test, y_train, y_test, list(datos.feature_names), list(datos.target_names))


def cargar_wine(seed: int = SEED) -> Particion:
    """Wine (178x13, 3 clases). Segundo dataset de clasificación, multiclase."""
    datos = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        datos.data,
        datos.target,
        test_size=0.2,
        random_state=seed,
        stratify=datos.target,
    )
    return Particion(X_train, X_test, y_train, y_test, list(datos.feature_names), list(datos.target_names))


def cargar_diabetes(seed: int = SEED) -> Particion:
    """Diabetes (442x10, regresión)."""
    datos = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        datos.data, datos.target, test_size=0.2, random_state=seed
    )
    return Particion(X_train, X_test, y_train, y_test, list(datos.feature_names))


def sintetico_moons(n_samples: int = 400, ruido: float = 0.25, seed: int = SEED) -> Particion:
    """make_moons 2D: fronteras no lineales donde Naive Bayes falla y el MLP brilla."""
    X, y = make_moons(n_samples=n_samples, noise=ruido, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return Particion(X_train, X_test, y_train, y_test, ["x1", "x2"], ["0", "1"])


def sintetico_circles(n_samples: int = 400, ruido: float = 0.1, seed: int = SEED) -> Particion:
    X, y = make_circles(n_samples=n_samples, noise=ruido, factor=0.5, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return Particion(X_train, X_test, y_train, y_test, ["x1", "x2"], ["0", "1"])


def sintetico_blobs(n_samples: int = 400, centros: int = 3, seed: int = SEED) -> Particion:
    X, y = make_blobs(n_samples=n_samples, centers=centros, random_state=seed, cluster_std=1.8)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return Particion(X_train, X_test, y_train, y_test, ["x1", "x2"])


def funcion_objetivo_seno(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def sintetico_seno1d(
    n_samples: int = 60, ruido: float = 0.15, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f(x) = sin(x) + ruido, con un hueco deliberado en [3, 5] para ver crecer la incertidumbre del GPR."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6, 6, size=n_samples)
    x = x[(x < 3) | (x > 5)]
    y = funcion_objetivo_seno(x) + rng.normal(0, ruido, size=x.shape)
    x_malla = np.linspace(-7, 7, 400)
    return x.reshape(-1, 1), y, x_malla.reshape(-1, 1)


def contaminar_outliers(
    X: np.ndarray,
    y: np.ndarray,
    fraccion: float,
    tipo: str = "atributos",
    es_clasificacion: bool = True,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Contamina una fracción de filas de ENTRENAMIENTO con atípicos, para el experimento de robustez.

    tipo="atributos": multiplica atributos por 5-10 desviaciones estándar.
    tipo="etiqueta": invierte etiquetas (clasificación) o pone valores extremos (regresión).
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    y = y.copy()
    n = X.shape[0]
    n_contaminados = int(round(fraccion * n))
    if n_contaminados == 0:
        return X, y
    idx = rng.choice(n, size=n_contaminados, replace=False)

    if tipo == "atributos":
        sigma = X.std(axis=0, keepdims=True)
        factor = rng.uniform(5, 10, size=(n_contaminados, X.shape[1]))
        signo = rng.choice([-1, 1], size=(n_contaminados, X.shape[1]))
        X[idx] = X[idx] + signo * factor * sigma
    elif tipo == "etiqueta":
        if es_clasificacion:
            clases = np.unique(y)
            for i in idx:
                otras = clases[clases != y[i]]
                y[i] = rng.choice(otras) if len(otras) else y[i]
        else:
            sigma_y = y.std()
            y[idx] = y[idx] + rng.choice([-1, 1], size=n_contaminados) * rng.uniform(5, 10, size=n_contaminados) * sigma_y
    else:
        raise ValueError(f"tipo de contaminación desconocido: {tipo}")

    return X, y
