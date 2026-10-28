# Comparativo de Familias de Aprendizaje Supervisado

**Sitio interactivo:** https://comparador-familias.netlify.app/ · **Bitácora de construcción (infografía paso a paso):** https://comparador-familias.netlify.app/proceso.html

**Curso:** Inteligencia Artificial · **Familia 3:** modelos probabilísticos y redes neuronales multicapa · **Equipo 5**

**Integrantes:**
- Coavoy Cruz, Joseph Gabriel
- Cuchuyrrumi Mamani, Manuel Rodrigo
- Huallpatuiro Rafaile, Brayan
- Mamani Acuña, Frank Joseph

Modelos asignados:
1. Naive Bayes Gaussiano
2. Perceptrón Multicapa para clasificación (MLP Classifier)
3. Regresión por Proceso Gaussiano

Entregables: informe técnico en PDF, código fuente en Python y simulación interactiva (web que consume los modelos ya entrenados, exportados a ONNX).

---

## Reproducir todo con un solo comando

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .

pytest -q                     # tests: sin fuga, matemática vs scikit-learn, métricas, paridad ONNX
python -m familia3.todo       # entrena los 6 modelos, exporta a ONNX, los registra y arma dist/
python -m familia3.verificar_release
```

`dist/` queda listo para servir estático (Netlify, o localmente con `python -m http.server --directory dist`).

Para ejecutar los notebooks de principio a fin:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

---

## Estructura del repositorio

```
Comparativo-de-Familias/
├── README.md
├── requirements.txt            # versiones fijas, instaladas y probadas
├── pyproject.toml              # paquete "familia3" instalable (pip install -e .)
├── .github/workflows/ci.yml    # pytest + entrenamiento/exportación + verificar_release
├── models/                     # registry.json y los .onnx registrados (con su SHA-256)
├── data/                       # datasets de scikit-learn se descargan/cachean aquí (California Housing)
├── src/familia3/
│   ├── datos.py                 # carga, particiones, semillas, contaminación con atípicos
│   ├── pipelines.py             # Pipeline por modelo (escalado + estimador)
│   ├── busqueda.py               # Grid/RandomizedSearch
│   ├── metricas.py              # clasificación, regresión, calibración, bootstrap
│   ├── graficos.py               # fronteras, curvas, bandas GPR, importancia
│   ├── desde_cero.py            # NB, MLP y GPR en NumPy puro (verificación matemática)
│   ├── exportar.py              # sklearn → ONNX (skl2onnx) + paridad + registro
│   ├── publicar.py              # arma dist/: web + registry.json + modelos/*.onnx
│   ├── verificar_release.py     # hashes, versión vigente, nada escrito a mano en la web
│   └── todo.py                   # un solo comando: entrena + exporta + registra + publica
├── notebooks/
│   ├── 01_datos_y_protocolo.ipynb
│   ├── 02_naive_bayes_gaussiano.ipynb
│   ├── 03_mlp_clasificador.ipynb
│   ├── 04_regresion_proceso_gaussiano.ipynb
│   └── 05_comparativo_y_aplicaciones.ipynb
├── tests/                       # pytest: fuga, matemática vs sklearn, métricas, paridad ONNX
├── web/                          # sitio: HTML + JS que lee registry.json y consume los .onnx
├── informe/                     # markdown fuente del informe técnico
├── docs/                         # enunciado de la actividad y plan de trabajo
└── reports/figuras/              # PNG generados por los notebooks
```

---

## La web consume los modelos entrenados (no reimplementa nada)

```
entrenar y elegir hiperparámetros (Python, scikit-learn)
        ↓
exportar a ONNX (skl2onnx) + prueba de paridad Python ↔ ONNX
        ↓
registrar en models/registry.json (versión, SHA-256, métricas, hiperparámetros, dataset, semilla, fecha)
        ↓
publicar: dist/ = sitio web + registry.json + modelos/*.onnx
        ↓
la web lee el registro → descarga el .onnx → verifica el SHA-256 → carga con ONNX Runtime Web → predice
```

La página no tiene escrito a mano el modelo, su versión ni sus métricas: todo sale de `registry.json`. Si el hash no coincide, el modelo se rechaza (`src/familia3/verificar_release.py` lo comprueba en CI). Se puede fijar una versión con `?version=nb_cancer:v1`.

**Modelos registrados:**

| Id | Modelo | Datos | Entrada → salida |
|---|---|---|---|
| `nb_cancer` | Naive Bayes Gaussiano | Breast Cancer, 30 atributos | 30 números → clase y probabilidades |
| `mlp_cancer` | Pipeline (escalado + MLP) | Breast Cancer | 30 números → clase y probabilidades |
| `gpr_diabetes` | Proceso Gaussiano | Diabetes, 10 atributos | 10 números → media y desviación estándar |
| `nb_moons`, `mlp_moons` | NB y MLP | `make_moons` 2D | 2 números → probabilidades (fronteras) |
| `gpr_seno1d` | Proceso Gaussiano | f(x) = sin(x) + ruido | x → media y σ (banda ±2σ) |

**Limitación técnica documentada:** `skl2onnx` 1.20 falla al exportar `GaussianProcessRegressor(normalize_y=True, return_std=True)`. Los dos modelos GPR se entrenan con `normalize_y=False` y el objetivo estandarizado a mano (media/desviación guardadas en el registro como `y_offset`/`y_scale`); la web deshace la transformación al leer el registro, igual que con cualquier otro dato del modelo.

---

## Protocolo de evaluación

- Partición 80/20, estratificada en clasificación. El 20% de prueba no se toca hasta el final.
- Búsqueda de hiperparámetros con validación cruzada de 5 pliegues **solo dentro del entrenamiento**.
- Escalado dentro de un `Pipeline` (sin fuga de datos).
- Semilla fija `SEED = 42`, repetición con varias semillas para media ± desviación estándar.
- Prueba final una sola vez por modelo; intervalos de confianza por bootstrap.

## Dónde se cumple cada parte de la actividad

Ver [`docs/PLAN_DE_TRABAJO.md`](docs/PLAN_DE_TRABAJO.md) (sección 8) para la matriz de trazabilidad completa frente a la rúbrica, y [`docs/Actividad de Investigación.pdf`](docs/Actividad%20de%20Investigaci%C3%B3n.pdf) para el enunciado original.

## Sitio en Netlify

**https://comparador-familias.netlify.app/** — desplegado arrastrando `dist/` a un sitio nuevo de Netlify, con `netlify.toml` sin paso de compilación.
