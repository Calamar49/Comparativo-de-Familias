import numpy as np
import pytest

from familia3 import datos as D
from familia3.exportar import exportar_a_onnx, probar_paridad_clasificacion, probar_paridad_regresion
from familia3.pipelines import pipeline_gpr, pipeline_mlp, pipeline_naive_bayes


def test_paridad_naive_bayes_onnx(tmp_path, monkeypatch):
    import familia3.exportar as exportar_mod

    monkeypatch.setattr(exportar_mod, "CARPETA_MODELOS", tmp_path)
    p = D.cargar_breast_cancer()
    modelo = pipeline_naive_bayes().fit(p.X_train, p.y_train)
    ruta = exportar_a_onnx(modelo, p.X_train.astype(np.float32), "nb_test.onnx")
    diferencia = probar_paridad_clasificacion(modelo, ruta, p.X_test.astype(np.float32))
    assert diferencia < 1e-4


def test_paridad_mlp_onnx(tmp_path, monkeypatch):
    import familia3.exportar as exportar_mod

    monkeypatch.setattr(exportar_mod, "CARPETA_MODELOS", tmp_path)
    p = D.cargar_breast_cancer()
    modelo = pipeline_mlp(hidden_layer_sizes=(16,), seed=0).fit(p.X_train, p.y_train)
    ruta = exportar_a_onnx(modelo, p.X_train.astype(np.float32), "mlp_test.onnx")
    diferencia = probar_paridad_clasificacion(modelo, ruta, p.X_test.astype(np.float32))
    assert diferencia < 1e-3


def test_paridad_gpr_rbf_onnx(tmp_path, monkeypatch):
    import familia3.exportar as exportar_mod

    monkeypatch.setattr(exportar_mod, "CARPETA_MODELOS", tmp_path)
    p = D.cargar_diabetes()
    modelo = pipeline_gpr().named_steps["clf"]
    modelo.fit(p.X_train[:150], p.y_train[:150])
    ruta = exportar_a_onnx(modelo, p.X_train[:150].astype(np.float32), "gpr_test.onnx")
    diferencia = probar_paridad_regresion(modelo, ruta, p.X_test.astype(np.float32))
    assert diferencia["diff_media"] < 1e-3
