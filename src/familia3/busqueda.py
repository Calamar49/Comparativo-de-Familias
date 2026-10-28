"""Rejillas de búsqueda de hiperparámetros: GridSearchCV / RandomizedSearchCV,
siempre con validación cruzada SOLO dentro del conjunto de entrenamiento."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from familia3 import SEED
from familia3.pipelines import kernel_matern, kernel_rbf, pipeline_gpr, pipeline_mlp, pipeline_naive_bayes


def buscar_naive_bayes(X_train, y_train, seed: int = SEED) -> GridSearchCV:
    rejilla = {"clf__var_smoothing": np.logspace(-12, -1, 23)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    busqueda = GridSearchCV(
        pipeline_naive_bayes(), rejilla, scoring="roc_auc_ovr", cv=cv, n_jobs=-1, refit=True
    )
    busqueda.fit(X_train, y_train)
    return busqueda


def buscar_mlp(X_train, y_train, n_iter: int = 30, seed: int = SEED) -> RandomizedSearchCV:
    rejilla = {
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (32, 32, 16)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": np.logspace(-5, -1, 10),
        "clf__learning_rate_init": np.logspace(-4, -1, 10),
        "clf__solver": ["adam", "sgd"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    busqueda = RandomizedSearchCV(
        pipeline_mlp(seed=seed),
        rejilla,
        n_iter=n_iter,
        scoring="roc_auc_ovr",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        refit=True,
    )
    busqueda.fit(X_train, y_train)
    return busqueda


def buscar_gpr(X_train, y_train, seed: int = SEED) -> GridSearchCV:
    """Los parámetros continuos del núcleo (longitud de escala, ruido) se optimizan
    solos dentro de .fit() maximizando la log-verosimilitud marginal (L-BFGS con
    reinicios); aquí solo buscamos la FAMILIA de núcleo y el alpha de regularización."""
    rejilla = {
        "clf__kernel": [kernel_rbf(), kernel_matern(nu=1.5), kernel_matern(nu=2.5)],
        "clf__alpha": [1e-10, 1e-8, 1e-6, 1e-4],
    }
    busqueda = GridSearchCV(pipeline_gpr(seed=seed), rejilla, scoring="r2", cv=5, n_jobs=-1, refit=True)
    busqueda.fit(X_train, y_train)
    return busqueda
