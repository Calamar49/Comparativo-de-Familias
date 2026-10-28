"""Comprobaciones antes de publicar: hashes del registro coinciden con los .onnx,
cada modelo tiene una version_vigente, y la web no tiene nada escrito a mano
(ningún id de modelo, versión o métrica aparece literal en el HTML/JS)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_MODELOS = RAIZ / "models"
ARCHIVO_REGISTRO = CARPETA_MODELOS / "registry.json"
CARPETA_WEB = RAIZ / "web"


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    h.update(ruta.read_bytes())
    return h.hexdigest()


def verificar_hashes() -> list[str]:
    errores = []
    if not ARCHIVO_REGISTRO.exists():
        return [f"No existe {ARCHIVO_REGISTRO}"]
    registro = json.loads(ARCHIVO_REGISTRO.read_text(encoding="utf-8"))
    for id_modelo, info in registro["modelos"].items():
        if "version_vigente" not in info:
            errores.append(f"{id_modelo}: falta 'version_vigente'")
            continue
        vigente = next((v for v in info["versiones"] if v["version"] == info["version_vigente"]), None)
        if vigente is None:
            errores.append(f"{id_modelo}: version_vigente '{info['version_vigente']}' no existe en 'versiones'")
            continue
        ruta_onnx = CARPETA_MODELOS / vigente["archivo"]
        if not ruta_onnx.exists():
            errores.append(f"{id_modelo}: no existe el archivo {ruta_onnx}")
            continue
        hash_real = _sha256(ruta_onnx)
        if hash_real != vigente["sha256"]:
            errores.append(f"{id_modelo}: SHA-256 no coincide (registro={vigente['sha256'][:12]}..., real={hash_real[:12]}...)")
    return errores


def verificar_sin_datos_a_mano() -> list[str]:
    """Busca en el JS/HTML de la web ids de modelo, números de versión o métricas
    escritos literalmente, en vez de leídos del registro."""
    errores = []
    patrones_prohibidos = [
        r'"nb_cancer"\s*:', r'"mlp_cancer"\s*:', r'"gpr_diabetes"\s*:',
        r"roc_auc\s*[:=]\s*0\.\d+", r"rmse\s*[:=]\s*\d+\.\d+",
    ]
    for archivo in CARPETA_WEB.rglob("*"):
        if archivo.suffix not in (".html", ".js"):
            continue
        texto = archivo.read_text(encoding="utf-8", errors="ignore")
        for patron in patrones_prohibidos:
            if re.search(patron, texto):
                errores.append(f"{archivo.relative_to(RAIZ)}: coincide con patrón prohibido '{patron}'")
    return errores


def main() -> int:
    errores = verificar_hashes() + verificar_sin_datos_a_mano()
    if errores:
        print("verificar_release: FALLÓ")
        for e in errores:
            print(f"  - {e}")
        return 1
    print("verificar_release: OK (hashes verificados, nada escrito a mano en la web)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
