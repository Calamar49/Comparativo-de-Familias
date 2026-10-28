"""sklearn -> ONNX (skl2onnx), prueba de paridad numérica, y registro en models/registry.json.

Regla del proyecto: la web no lleva escrito a mano ningún dato del modelo; todo
sale de este registro (versión, hash, métricas, hiperparámetros, dataset, semilla, fecha).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_MODELOS = RAIZ / "models"
ARCHIVO_REGISTRO = CARPETA_MODELOS / "registry.json"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    h.update(ruta.read_bytes())
    return h.hexdigest()


def exportar_a_onnx(modelo, X_ejemplo: np.ndarray, nombre_archivo: str, opset: int = 17) -> Path:
    """Convierte un Pipeline/estimador de sklearn a ONNX usando un ejemplo de entrada
    para inferir las formas y tipos."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    tipo_entrada = [("entrada", FloatTensorType([None, X_ejemplo.shape[1]]))]

    opciones = None
    if isinstance(modelo, GaussianProcessRegressor):
        opciones = {id(modelo): {"return_std": True}}
    elif hasattr(modelo, "predict_proba"):
        opciones = {id(modelo): {"zipmap": False}}

    onnx_modelo = to_onnx(
        modelo,
        initial_types=tipo_entrada,
        target_opset=opset,
        options=opciones,
    )
    ruta = CARPETA_MODELOS / nombre_archivo
    ruta.write_bytes(onnx_modelo.SerializeToString())
    return ruta


def probar_paridad_clasificacion(modelo_sklearn, ruta_onnx: Path, X_prueba: np.ndarray) -> float:
    """Diferencia máxima absoluta entre las probabilidades de sklearn y las de ONNX Runtime."""
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    nombre_entrada = sesion.get_inputs()[0].name
    salidas_onnx = sesion.run(None, {nombre_entrada: X_prueba.astype(np.float32)})
    proba_onnx = salidas_onnx[1] if len(salidas_onnx) > 1 else salidas_onnx[0]
    proba_sklearn = modelo_sklearn.predict_proba(X_prueba)
    return float(np.max(np.abs(np.asarray(proba_onnx) - proba_sklearn)))


def probar_paridad_regresion(modelo_sklearn, ruta_onnx: Path, X_prueba: np.ndarray, con_desviacion: bool = True) -> dict:
    """Para GPR: compara media (y, si aplica, sigma) entre sklearn y ONNX Runtime."""
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta_onnx), providers=["CPUExecutionProvider"])
    nombre_entrada = sesion.get_inputs()[0].name
    salidas = sesion.run(None, {nombre_entrada: X_prueba.astype(np.float32)})

    if con_desviacion:
        media_sk, sigma_sk = modelo_sklearn.predict(X_prueba, return_std=True)
        media_onnx = np.asarray(salidas[0]).ravel()
        sigma_onnx = np.asarray(salidas[1]).ravel()
        return {
            "diff_media": float(np.max(np.abs(media_onnx - media_sk))),
            "diff_sigma": float(np.max(np.abs(sigma_onnx - sigma_sk))),
        }
    media_sk = modelo_sklearn.predict(X_prueba)
    media_onnx = np.asarray(salidas[0]).ravel()
    return {"diff_media": float(np.max(np.abs(media_onnx - media_sk)))}


def registrar_modelo(
    id_modelo: str,
    descripcion: str,
    ruta_onnx: Path,
    metricas: dict,
    hiperparametros: dict,
    dataset: str,
    seed: int,
    diferencia_paridad: float,
    entrada: list[str],
    salida: list[str],
    version: str | None = None,
    y_offset: float = 0.0,
    y_scale: float = 1.0,
) -> dict:
    """Agrega (o versiona) una entrada en models/registry.json."""
    ARCHIVO_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVO_REGISTRO.exists():
        registro = json.loads(ARCHIVO_REGISTRO.read_text(encoding="utf-8"))
    else:
        registro = {"modelos": {}}

    entradas_previas = registro["modelos"].get(id_modelo, {"versiones": []})
    if version is None:
        version = f"v{len(entradas_previas['versiones']) + 1}"

    nueva_version = {
        "version": version,
        "archivo": ruta_onnx.name,
        "sha256": _sha256(ruta_onnx),
        "descripcion": descripcion,
        "dataset": dataset,
        "seed": seed,
        "entrada": entrada,
        "salida": salida,
        "metricas": metricas,
        "hiperparametros": hiperparametros,
        "y_offset": y_offset,
        "y_scale": y_scale,
        "diferencia_paridad_onnx": diferencia_paridad,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entradas_previas["versiones"] = [v for v in entradas_previas["versiones"] if v["version"] != version]
    entradas_previas["versiones"].append(nueva_version)
    entradas_previas["version_vigente"] = version
    registro["modelos"][id_modelo] = entradas_previas

    ARCHIVO_REGISTRO.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    return nueva_version
