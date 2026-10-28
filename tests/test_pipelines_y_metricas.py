import numpy as np

from familia3 import datos as D
from familia3.metricas import bootstrap_ic, cobertura_intervalo, metricas_clasificacion, metricas_regresion, nlpd_gaussiano
from familia3.pipelines import pipeline_gpr, pipeline_mlp, pipeline_naive_bayes


def test_pipeline_naive_bayes_entrena_y_predice():
    p = D.cargar_breast_cancer()
    modelo = pipeline_naive_bayes().fit(p.X_train, p.y_train)
    y_pred = modelo.predict(p.X_test)
    metricas = metricas_clasificacion(p.y_test, y_pred, modelo.predict_proba(p.X_test))
    assert metricas["f1"] > 0.8


def test_pipeline_mlp_escala_dentro_del_pipeline():
    p = D.cargar_breast_cancer()
    modelo = pipeline_mlp(hidden_layer_sizes=(16,), seed=0).fit(p.X_train, p.y_train)
    assert "escalado" in modelo.named_steps
    y_pred = modelo.predict(p.X_test)
    metricas = metricas_clasificacion(p.y_test, y_pred, modelo.predict_proba(p.X_test))
    assert metricas["roc_auc"] > 0.85


def test_pipeline_gpr_devuelve_media_y_sigma():
    p = D.cargar_diabetes()
    modelo = pipeline_gpr().fit(p.X_train[:150], p.y_train[:150])
    media, sigma = modelo.named_steps["clf"].predict(p.X_test, return_std=True)
    assert media.shape == p.y_test.shape
    assert np.all(sigma >= 0)
    r = metricas_regresion(p.y_test, media)
    assert "rmse" in r and "r2" in r


def test_no_fuga_de_datos_pipeline_mlp():
    """El escalado debe calcularse SOLO con datos de entrenamiento."""
    p = D.cargar_breast_cancer()
    modelo = pipeline_mlp(hidden_layer_sizes=(8,), seed=0)
    modelo.fit(p.X_train, p.y_train)
    media_escalador = modelo.named_steps["escalado"].mean_
    assert np.allclose(media_escalador, p.X_train.mean(axis=0))


def test_bootstrap_ic_contiene_el_punto():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.1, size=200)
    punto, bajo, alto = bootstrap_ic(y_true, y_pred, lambda a, b: np.mean((a - b) ** 2))
    assert bajo <= punto <= alto


def test_nlpd_y_cobertura_gaussianos():
    y_true = np.array([0.0, 1.0, 2.0])
    media_buena = np.array([0.1, 0.9, 2.2])
    media_mala = np.array([3.0, -2.0, 8.0])
    sigma = np.array([0.2, 0.2, 0.2])
    # peores predicciones -> mayor (peor) log-verosimilitud predictiva negativa
    assert nlpd_gaussiano(y_true, media_mala, sigma) > nlpd_gaussiano(y_true, media_buena, sigma)
    assert 0.0 <= cobertura_intervalo(y_true, media_buena, sigma) <= 1.0
