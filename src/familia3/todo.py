"""Un solo comando para rehacer todos los resultados: entrena, exporta a ONNX,
registra en models/registry.json y publica dist/.

Uso:  python -m familia3.todo
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from familia3 import SEED
from familia3 import datos as D
from familia3.exportar import CARPETA_MODELOS, exportar_a_onnx, probar_paridad_clasificacion, probar_paridad_regresion, registrar_modelo
from familia3.metricas import metricas_clasificacion, metricas_regresion
from familia3.pipelines import kernel_matern, pipeline_gpr, pipeline_mlp, pipeline_naive_bayes
from familia3.publicar import publicar


def _volcar_datos_web(nombre_dataset: str, particion: D.Particion, n_casos: int = 8):
    """Guarda casos de prueba y rangos por atributo (min/max/media) para que la web
    construya deslizadores y un selector de casos, sin reimplementar nada del modelo."""
    idx = np.linspace(0, len(particion.X_test) - 1, n_casos).astype(int)
    casos = [
        {"atributos": particion.X_test[i].tolist(), "etiqueta_real": int(particion.y_test[i])}
        for i in idx
    ]
    rangos = [
        {
            "nombre": nombre,
            "min": float(particion.X_train[:, j].min()),
            "max": float(particion.X_train[:, j].max()),
            "media": float(particion.X_train[:, j].mean()),
        }
        for j, nombre in enumerate(particion.nombres_atributos)
    ]
    salida = {"casos_de_prueba": casos, "rangos_atributos": rangos}
    (CARPETA_MODELOS / f"datos_web_{nombre_dataset}.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _entrenar_y_registrar_clasificador(
    id_modelo, descripcion, modelo, particion: D.Particion, dataset_nombre, hiperparametros
):
    modelo.fit(particion.X_train, particion.y_train)
    y_pred = modelo.predict(particion.X_test)
    y_proba = modelo.predict_proba(particion.X_test)
    metricas = metricas_clasificacion(particion.y_test, y_pred, y_proba)

    ruta_onnx = exportar_a_onnx(modelo, particion.X_train.astype(np.float32), f"{id_modelo}.onnx")
    diferencia = probar_paridad_clasificacion(modelo, ruta_onnx, particion.X_test.astype(np.float32))

    entrada = particion.nombres_atributos
    salida = particion.nombres_clases or ["clase"]
    registrar_modelo(
        id_modelo, descripcion, ruta_onnx, metricas, hiperparametros, dataset_nombre, SEED, diferencia, entrada, salida
    )
    print(f"[{id_modelo}] métricas={metricas} | paridad ONNX={diferencia:.2e}")


def _entrenar_y_registrar_gpr(id_modelo, descripcion, modelo, X_train, y_train, X_test, y_test, dataset_nombre, hiperparametros, entrada):
    # normalize_y=False (ver pipelines.pipeline_gpr): se estandariza "y" a mano
    # (media y desviación) porque skl2onnx 1.20 falla al exportar
    # GaussianProcessRegressor(normalize_y=True, return_std=True). Sin esto, con el
    # target en su escala original el optimizador del núcleo cae en un óptimo local
    # degenerado (todo el ruido explicado por WhiteKernel, predicción ≈ media, R²≈0).
    # El ONNX exportado predice en la escala ESTANDARIZADA; "y_offset" y "y_scale"
    # quedan en el registro para que la web deshaga la transformación (media*scale+offset,
    # sigma*scale), igual que con cualquier otro dato del modelo.
    y_offset = float(np.mean(y_train))
    y_scale = float(np.std(y_train)) or 1.0
    modelo.fit(X_train, (y_train - y_offset) / y_scale)
    media_std, sigma_std = modelo.predict(X_test, return_std=True)
    media = media_std * y_scale + y_offset
    metricas = metricas_regresion(y_test, media)

    ruta_onnx = exportar_a_onnx(modelo, X_train.astype(np.float32), f"{id_modelo}.onnx")
    diferencia = probar_paridad_regresion(modelo, ruta_onnx, X_test.astype(np.float32))

    registrar_modelo(
        id_modelo,
        descripcion,
        ruta_onnx,
        metricas,
        hiperparametros,
        dataset_nombre,
        SEED,
        max(diferencia.values()),
        entrada,
        ["media", "sigma"],
        y_offset=y_offset,
        y_scale=y_scale,
    )
    print(f"[{id_modelo}] métricas={metricas} | paridad ONNX={diferencia} | y_offset={y_offset:.3f} y_scale={y_scale:.3f}")


def main():
    # --- Naive Bayes y MLP sobre Breast Cancer ---
    cancer = D.cargar_breast_cancer()
    _entrenar_y_registrar_clasificador(
        "nb_cancer", "Naive Bayes Gaussiano sobre Breast Cancer Wisconsin",
        pipeline_naive_bayes(), cancer, "breast_cancer", {"var_smoothing": 1e-9},
    )
    _entrenar_y_registrar_clasificador(
        "mlp_cancer", "MLP (con escalado) sobre Breast Cancer Wisconsin",
        pipeline_mlp(), cancer, "breast_cancer",
        {"hidden_layer_sizes": [64, 32], "activation": "relu", "alpha": 1e-4},
    )
    _volcar_datos_web("breast_cancer", cancer)

    # --- NB y MLP sobre make_moons (para fronteras 2D) ---
    moons = D.sintetico_moons()
    _entrenar_y_registrar_clasificador(
        "nb_moons", "Naive Bayes Gaussiano sobre make_moons (frontera 2D)",
        pipeline_naive_bayes(), moons, "make_moons", {"var_smoothing": 1e-9},
    )
    _entrenar_y_registrar_clasificador(
        "mlp_moons", "MLP sobre make_moons (frontera 2D)",
        pipeline_mlp(hidden_layer_sizes=(32, 16)), moons, "make_moons",
        {"hidden_layer_sizes": [32, 16], "activation": "relu"},
    )
    _volcar_datos_web("make_moons", moons)

    # --- GPR sobre Diabetes ---
    diabetes = D.cargar_diabetes()
    _entrenar_y_registrar_gpr(
        "gpr_diabetes", "Proceso Gaussiano (núcleo Matérn) sobre Diabetes",
        pipeline_gpr(kernel=kernel_matern(nu=1.5)).named_steps["clf"],
        diabetes.X_train, diabetes.y_train, diabetes.X_test, diabetes.y_test,
        "diabetes", {"kernel": "Matern(nu=1.5)+WhiteKernel"}, diabetes.nombres_atributos,
    )
    _volcar_datos_web("diabetes", diabetes)

    # --- GPR sobre seno 1D ---
    x_train, y_train, x_malla = D.sintetico_seno1d()
    _entrenar_y_registrar_gpr(
        "gpr_seno1d", "Proceso Gaussiano (núcleo RBF) sobre f(x)=sin(x)+ruido",
        pipeline_gpr().named_steps["clf"],
        x_train, y_train, x_malla[:50], np.sin(x_malla[:50]).ravel(),
        "sintetico_seno1d", {"kernel": "RBF+WhiteKernel"}, ["x"],
    )

    ruta_dist = publicar()
    print(f"\nTodo listo. Sitio publicado en: {ruta_dist}")


if __name__ == "__main__":
    main()
