"""Métricas de clasificación, regresión, calibración y bootstrap de intervalos de confianza."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)

from familia3 import SEED


def metricas_clasificacion(y_true, y_pred, y_proba=None) -> dict:
    binaria = len(np.unique(y_true)) == 2
    salida = {
        "precision": precision_score(y_true, y_pred, average="binary" if binaria else "macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary" if binaria else "macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary" if binaria else "macro", zero_division=0),
    }
    if y_proba is not None:
        try:
            if binaria:
                salida["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                salida["log_loss"] = log_loss(y_true, y_proba)
                salida["brier"] = brier_score_loss(y_true, y_proba[:, 1])
            else:
                salida["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                salida["log_loss"] = log_loss(y_true, y_proba)
        except ValueError:
            pass
    return salida


def metricas_regresion(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def nlpd_gaussiano(y_true, media, sigma) -> float:
    """Log-verosimilitud predictiva negativa promedio, bajo una posterior gaussiana (GPR)."""
    sigma = np.clip(sigma, 1e-9, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - media) ** 2) / sigma**2))


def cobertura_intervalo(y_true, media, sigma, z: float = 1.96) -> float:
    """Fracción de puntos de prueba dentro de media ± z·sigma (z=1.96 -> 95%)."""
    dentro = np.abs(y_true - media) <= z * sigma
    return float(np.mean(dentro))


def bootstrap_ic(
    y_true,
    y_pred,
    funcion_metrica,
    n_bootstrap: int = 1000,
    alfa: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Intervalo de confianza por bootstrap no paramétrico sobre el conjunto de prueba."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    valores = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        valores[b] = funcion_metrica(y_true[idx], y_pred[idx])
    punto = funcion_metrica(y_true, y_pred)
    bajo = np.percentile(valores, 100 * alfa / 2)
    alto = np.percentile(valores, 100 * (1 - alfa / 2))
    return float(punto), float(bajo), float(alto)
