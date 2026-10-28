import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB

from familia3 import datos as D
from familia3.desde_cero import (
    MLPNumPy,
    NaiveBayesGaussianoNumPy,
    ProcesoGaussianoNumPy,
    comprobar_gradientes,
)


def test_naive_bayes_numpy_coincide_con_sklearn():
    p = D.cargar_breast_cancer()
    propio = NaiveBayesGaussianoNumPy().fit(p.X_train, p.y_train)
    referencia = GaussianNB().fit(p.X_train, p.y_train)

    proba_propio = propio.predict_proba(p.X_test)
    proba_referencia = referencia.predict_proba(p.X_test)
    assert np.allclose(proba_propio, proba_referencia, atol=1e-8)

    pred_propio = propio.predict(p.X_test)
    pred_referencia = referencia.predict(p.X_test)
    assert np.array_equal(pred_propio, pred_referencia)


def test_comprobacion_gradientes_mlp_numpy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.integers(0, 3, size=20)
    modelo = MLPNumPy(n_entradas=4, n_ocultas=5, n_clases=3, alpha=1e-3, seed=0)
    error_relativo = comprobar_gradientes(modelo, X, y)
    assert error_relativo < 1e-4


def test_mlp_numpy_aprende_reduce_perdida():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    modelo = MLPNumPy(n_entradas=4, n_ocultas=8, n_clases=2, alpha=1e-4, seed=0)
    modelo.fit(X, y, epocas=300, tasa_aprendizaje=1.0)
    assert modelo.historial_perdida_[-1] < modelo.historial_perdida_[0]


def test_gpr_numpy_coincide_con_sklearn():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(30, 1))
    y = np.sin(X).ravel() + rng.normal(0, 0.05, size=30)

    propio = ProcesoGaussianoNumPy(longitud_escala=1.0, varianza=1.0, ruido=1e-6, normalize_y=True).fit(X, y)
    # optimizer=None: mismo núcleo FIJO que la versión NumPy (sin optimizar hiperparámetros),
    # para comparar la misma fórmula, no un ajuste distinto.
    referencia = GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0), alpha=1e-6, normalize_y=True, optimizer=None
    ).fit(X, y)

    X_nuevo = rng.uniform(-3, 3, size=(10, 1))
    media_propio, sigma_propio = propio.predict(X_nuevo)
    media_referencia, sigma_referencia = referencia.predict(X_nuevo, return_std=True)

    assert np.allclose(media_propio, media_referencia, atol=1e-6)
    assert np.allclose(sigma_propio, sigma_referencia, atol=1e-6)


def test_banda_incertidumbre_cubre_95_por_ciento():
    x_train, y_train, x_malla = D.sintetico_seno1d(n_samples=200, ruido=0.1)
    modelo = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1**2, normalize_y=True, n_restarts_optimizer=3)
    modelo.fit(x_train, y_train)

    rng = np.random.default_rng(1)
    x_prueba = rng.uniform(-6, 6, size=(300, 1))
    x_prueba = x_prueba[(x_prueba[:, 0] < 3) | (x_prueba[:, 0] > 5)].reshape(-1, 1)
    y_real = np.sin(x_prueba).ravel()
    media, sigma = modelo.predict(x_prueba, return_std=True)

    cobertura = np.mean(np.abs(y_real - media) <= 1.96 * sigma)
    assert cobertura > 0.85
