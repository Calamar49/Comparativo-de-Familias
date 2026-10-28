import numpy as np

from familia3 import datos as D


def test_particion_breast_cancer_80_20_estratificada():
    p = D.cargar_breast_cancer()
    n_total = len(p.X_train) + len(p.X_test)
    assert abs(len(p.X_test) / n_total - 0.2) < 0.02
    prop_train = np.bincount(p.y_train) / len(p.y_train)
    prop_test = np.bincount(p.y_test) / len(p.y_test)
    assert np.allclose(prop_train, prop_test, atol=0.05)


def test_sin_fuga_entre_train_y_test():
    p = D.cargar_breast_cancer()
    filas_train = {tuple(fila) for fila in p.X_train}
    filas_test = {tuple(fila) for fila in p.X_test}
    assert filas_train.isdisjoint(filas_test)


def test_semilla_fija_da_mismo_split():
    p1 = D.cargar_breast_cancer(seed=42)
    p2 = D.cargar_breast_cancer(seed=42)
    assert np.array_equal(p1.X_train, p2.X_train)
    assert np.array_equal(p1.y_train, p2.y_train)


def test_contaminar_outliers_no_cambia_filas_no_elegidas():
    p = D.cargar_breast_cancer()
    X_contaminado, y_contaminado = D.contaminar_outliers(p.X_train, p.y_train, fraccion=0.1, tipo="atributos")
    assert X_contaminado.shape == p.X_train.shape
    diferencias = ~np.all(np.isclose(X_contaminado, p.X_train), axis=1)
    assert diferencias.sum() == int(round(0.1 * len(p.X_train)))


def test_sintetico_seno1d_tiene_hueco():
    x_train, _, _ = D.sintetico_seno1d(n_samples=200)
    assert not np.any((x_train.ravel() > 3) & (x_train.ravel() < 5))
