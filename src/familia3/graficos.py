"""Gráficos reutilizables: fronteras de decisión, curvas de aprendizaje, bandas GPR, etc.
Todas las funciones guardan a PNG en reports/figuras/ y también devuelven la figura."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

CARPETA_FIGURAS = Path(__file__).resolve().parents[2] / "reports" / "figuras"


def _guardar(fig, nombre: str):
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CARPETA_FIGURAS / nombre, dpi=150, bbox_inches="tight")


def frontera_decision(modelo, X, y, nombre_archivo: str, titulo: str = ""):
    """Dibuja la frontera de decisión de un clasificador binario/multiclase en 2D."""
    fig, ax = plt.subplots(figsize=(5, 4))
    margen = 0.5
    x_min, x_max = X[:, 0].min() - margen, X[:, 0].max() + margen
    y_min, y_max = X[:, 1].min() - margen, X[:, 1].max() + margen
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_title(titulo)
    _guardar(fig, nombre_archivo)
    return fig


def superficie_probabilidad(modelo, X, y, nombre_archivo: str, titulo: str = ""):
    fig, ax = plt.subplots(figsize=(5, 4))
    margen = 0.5
    x_min, x_max = X[:, 0].min() - margen, X[:, 0].max() + margen
    y_min, y_max = X[:, 1].min() - margen, X[:, 1].max() + margen
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    proba = modelo.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    proba = proba.reshape(xx.shape)
    cs = ax.contourf(xx, yy, proba, levels=20, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax, label="P(clase=1)")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_title(titulo)
    _guardar(fig, nombre_archivo)
    return fig


def curva_aprendizaje(modelo, X, y, nombre_archivo: str, titulo: str = "", scoring="roc_auc_ovr"):
    tamanos, train_scores, test_scores = learning_curve(
        modelo, X, y, cv=5, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(tamanos, train_scores.mean(axis=1), "o-", label="entrenamiento")
    ax.plot(tamanos, test_scores.mean(axis=1), "o-", label="validación")
    ax.fill_between(
        tamanos,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.2,
    )
    ax.set_xlabel("tamaño de entrenamiento")
    ax.set_ylabel(scoring)
    ax.set_title(titulo)
    ax.legend()
    _guardar(fig, nombre_archivo)
    return fig


def banda_gpr_1d(x_train, y_train, x_malla, media, sigma, nombre_archivo: str, titulo: str = "", y_real=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    x_malla_flat = x_malla.ravel()
    ax.scatter(x_train, y_train, c="k", s=15, label="datos de entrenamiento", zorder=3)
    ax.plot(x_malla_flat, media, color="tab:blue", label="media predictiva")
    ax.fill_between(
        x_malla_flat,
        media - 2 * sigma,
        media + 2 * sigma,
        alpha=0.25,
        color="tab:blue",
        label="±2σ",
    )
    if y_real is not None:
        ax.plot(x_malla_flat, y_real, "--", color="gray", label="f(x) real")
    ax.set_title(titulo)
    ax.legend()
    _guardar(fig, nombre_archivo)
    return fig


def importancia_permutacion(importancias, nombres_atributos, nombre_archivo: str, titulo: str = ""):
    orden = np.argsort(importancias)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(nombres_atributos))))
    ax.barh(np.array(nombres_atributos)[orden], importancias[orden])
    ax.set_title(titulo)
    _guardar(fig, nombre_archivo)
    return fig


def curva_calibracion_manual(y_true, y_proba, nombre_archivo: str, titulo: str = "", n_bins: int = 10):
    from sklearn.calibration import calibration_curve

    frac_pos, media_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectamente calibrado")
    ax.plot(media_pred, frac_pos, "o-", label="modelo")
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
    ax.set_title(titulo)
    ax.legend()
    _guardar(fig, nombre_archivo)
    return fig
