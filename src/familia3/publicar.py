"""Arma dist/: copia web/ + registry.json + models/*.onnx. Solo librería estándar,
para poder correr en CI sin dependencias extra."""
from __future__ import annotations

import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_WEB = RAIZ / "web"
CARPETA_MODELOS = RAIZ / "models"
CARPETA_DIST = RAIZ / "dist"


def publicar() -> Path:
    if CARPETA_DIST.exists():
        shutil.rmtree(CARPETA_DIST)
    CARPETA_DIST.mkdir(parents=True)

    shutil.copytree(CARPETA_WEB, CARPETA_DIST, dirs_exist_ok=True)

    carpeta_modelos_dist = CARPETA_DIST / "models"
    carpeta_modelos_dist.mkdir(parents=True, exist_ok=True)
    for archivo in CARPETA_MODELOS.glob("*"):
        if archivo.suffix in (".onnx", ".json"):
            shutil.copy2(archivo, carpeta_modelos_dist / archivo.name)

    return CARPETA_DIST


if __name__ == "__main__":
    ruta = publicar()
    print(f"Sitio publicado en: {ruta}")
