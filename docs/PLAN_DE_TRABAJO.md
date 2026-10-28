# Plan de trabajo · Familia 3, Equipo 5
**Actividad:** Despliegue y Análisis Comparativo de Familias de Aprendizaje Supervisado · Inteligencia Artificial
**Familia 3:** Modelos probabilísticos y redes neuronales multicapa
**Modelos del Equipo 5:** (1) Naive Bayes Gaussiano · (2) Perceptrón Multicapa clasificador (MLP) · (3) Regresión por Proceso Gaussiano (GPR)
**Repositorio:** https://github.com/Calamar49/Comparativo-de-Familias · **Sitio:** Netlify (sitio estático, nuevo)

---

## 0. Qué se entrega (según el PDF) y cómo se evalúa

| Entregable | Qué es en este proyecto |
|---|---|
| Informe PDF con documento técnico | `informe/INFORME_TECNICO.pdf` (y .docx editable) |
| Código fuente (repositorio / Python) | Repositorio de GitHub con paquete `src/`, notebooks, tests y CI |
| Simulación interactiva | Sitio web en Netlify que **consume los modelos ya entrenados en Python** (archivos ONNX registrados y verificados) y deja interactuar con ellos en el navegador |

| Criterio de la rúbrica | Peso | Cómo lo cubrimos | Dónde se ve |
|---|---|---|---|
| Rigor matemático | 35 % | Ecuaciones, pérdida, supuestos y algoritmo de los 3 modelos, con derivaciones y verificación numérica | Informe cap. 2, anexo A, notebooks 01 a 04 |
| Simulación y código | 35 % | Pipelines sin fuga, búsqueda de hiperparámetros, métricas justificadas, semillas, tests, CI | `src/`, notebooks, `tests/`, informe cap. 3 a 5 |
| Análisis de aplicaciones | 20 % | Un caso real por modelo + comparación crítica de los 3, respaldada con experimentos de atípicos, costo e interpretabilidad | Informe cap. 6 y 7, notebook 05 |
| Conclusiones y presentación | 10 % | Conclusiones con evidencia, informe pulido, diapositivas y sitio web | Informe cap. 8, diapositivas, web |

Regla del proyecto: **todo requisito del PDF debe poder señalarse con el dedo** (sección, archivo o celda). La sección 8 trae la matriz de trazabilidad.

---

## 1. Decisiones de diseño (con su por qué)

### 1.1 Conjuntos de datos (no hay obligatorios, elegimos los más defendibles)
| Uso | Dataset | Por qué |
|---|---|---|
| Clasificación real (Naive Bayes y MLP, **los mismos datos para compararlos justo**) | **Breast Cancer Wisconsin** (569 × 30, viene en scikit-learn, sin descargas) | Caso de diagnóstico médico (lo menciona el PDF); atributos correlacionados, lo que pone a prueba el supuesto de independencia de Naive Bayes; ya conocido, resultados verificables |
| Segundo conjunto de clasificación, más difícil | **Wine** o **Digits** (scikit-learn), a decidir en el notebook 01 | Multiclase, para ver que las conclusiones no dependen de un solo dataset |
| Clasificación sintética 2D (fronteras de decisión) | `make_moons`, `make_circles`, `make_blobs`, XOR | Permite dibujar fronteras en 2D, mostrar **dónde falla Naive Bayes y dónde brilla el MLP** |
| Regresión real (GPR) | **Diabetes** (442 × 10, scikit-learn) y, como segundo, **California Housing** submuestreado a ~2 000 filas | GPR cuesta O(n³): hay que justificar el tamaño. Segundo dataset más grande para discutir el costo |
| Regresión sintética 1D | Función `f(x) = sin(x) + ruido` con huecos en los datos | La gran virtud del GPR es la **incertidumbre**: se ve la banda ±2σ crecer donde no hay datos |

Todo es reproducible sin credenciales ni descargas manuales (salvo California Housing, que scikit-learn descarga y cacheamos en `data/`).

### 1.2 Protocolo de evaluación (anti-trampa, igual de estricto que en X-ray)
- Partición **80 % entrenamiento / 20 % prueba**, estratificada en clasificación (criterio de los talleres del curso). El 20 % de prueba **no se toca** hasta el final.
- La búsqueda de hiperparámetros usa **validación cruzada de 5 pliegues solo dentro del entrenamiento** (`GridSearchCV` / `RandomizedSearchCV` con `StratifiedKFold`).
- Escalado y transformaciones **dentro de un `Pipeline`**, para que no haya fuga de datos al validar.
- Semilla fija global (`SEED = 42`) y repetición con varias semillas para dar media ± desviación estándar.
- Prueba final **una sola vez** por modelo; los intervalos de confianza se estiman por *bootstrap*.

### 1.3 Requisito: la web consume los modelos entrenados (ciclo de vida ML-Ops)
La simulación **no reimplementa** los modelos: usa exactamente los que se entrenaron y evaluaron en Python. El camino, igual que en el proyecto X-ray, es:

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

**Reglas del consumo (las mismas de X-ray):** la página no lleva escrito el modelo, su versión ni sus métricas; todo sale del registro. Si el hash no coincide, el modelo se rechaza. Publicar una versión nueva es registrarla y volver a generar `dist/`; la página no se toca. Se puede fijar una versión con `?version=`.

**Modelos que se registran (todos entrenados en Python):**
| Id en el registro | Modelo | Datos | Entrada → salida |
|---|---|---|---|
| `nb_cancer` | Naive Bayes Gaussiano | Breast Cancer, 30 atributos | 30 números → clase y probabilidades |
| `mlp_cancer` | Pipeline (escalado + MLP) | Breast Cancer | 30 números → clase y probabilidades (el escalado va **dentro** del ONNX) |
| `gpr_diabetes` | Proceso Gaussiano | Diabetes, 10 atributos | 10 números → media y desviación estándar |
| `nb_moons`, `mlp_moons` | NB y MLP | `make_moons` 2D | 2 números → probabilidades (para dibujar fronteras) |
| `gpr_seno1d` | Proceso Gaussiano | f(x) = sin(x) + ruido | x → media y σ (para dibujar la banda ±2σ) |

Cada modelo puede tener **varias versiones** (por ejemplo v1 con hiperparámetros por defecto y v2 ajustada con la búsqueda), y la web permite compararlas.

**Viabilidad comprobada (prueba hecha el 28/10/2026 con scikit-learn 1.9, skl2onnx y ONNX Runtime Web 1.22):**
| Modelo | ¿Exporta a ONNX? | Diferencia máxima frente a Python | ¿Corre en el navegador? |
|---|---|---|---|
| Naive Bayes Gaussiano | Sí | 5e-7 (Python ↔ ONNX); 0 en el navegador | Sí |
| MLP (con escalado) | Sí | 3e-7 (Python ↔ ONNX); 2e-7 en el navegador | Sí |
| GPR con núcleo RBF o Matérn + ruido (media y σ) | Sí | 1e-5 a 7e-5; 1,6e-5 en el navegador | Sí |
| GPR con núcleo RationalQuadratic | Exporta, **pero no coincide** (error de 0,2 en la media) | | **Se excluye de la exportación** |

**Límites a documentar en el informe (honestidad técnica):**
- El navegador **consume** modelos ya entrenados; **no entrena**. Por eso las fronteras y bandas se dibujan consultando el modelo sobre una malla de puntos, y el usuario interactúa con las entradas, no con los datos de entrenamiento.
- La precisión de ONNX es de 32 bits (diferencias del orden de 1e-5 frente a Python), y se informa como tolerancia.
- Solo se exportan los núcleos del GPR cuya paridad se comprobó (RBF y Matérn). El GPR guarda sus datos de entrenamiento dentro del archivo, y por eso se limita a ≤ 2 000 filas (tamaño y costo de predicción O(n)).
- Extra **opcional** si sobra tiempo: un «laboratorio» que entrene en vivo con JavaScript escrito a mano (clic para colocar puntos). Queda fuera del camino crítico; si se hace, se declara aparte y se verifica contra scikit-learn.

---

## 2. Estructura del repositorio

```
Comparativo-de-Familias/
├── README.md                  # qué es, cómo reproducir, enlaces, tabla de resultados
├── requirements.txt           # versiones fijas
├── .github/workflows/ci.yml   # tests + paridad ONNX + build del sitio + verificar_release
├── models/                    # registry.json y los .onnx registrados (con su SHA-256)
├── data/                      # cachés pequeños y README de procedencia
├── src/familia3/
│   ├── datos.py               # carga, particiones, semillas
│   ├── pipelines.py           # Pipeline por modelo (escalado + estimador)
│   ├── busqueda.py            # Grid/RandomizedSearch y rejillas
│   ├── metricas.py            # clasificación, regresión, calibración, bootstrap
│   ├── graficos.py            # fronteras, curvas, importancia, GPR con banda
│   ├── desde_cero.py          # NB, MLP y GPR en NumPy (verificación matemática)
│   ├── exportar.py            # sklearn → ONNX (skl2onnx) + paridad + registro en registry.json
│   ├── publicar.py            # arma dist/: web + registry.json + modelos/*.onnx (solo librería estándar)
│   └── verificar_release.py   # comprobaciones: hashes, versión vigente, nada escrito a mano en la web
├── notebooks/
│   ├── 01_datos_y_protocolo.ipynb
│   ├── 02_naive_bayes_gaussiano.ipynb
│   ├── 03_mlp_clasificador.ipynb
│   ├── 04_regresion_proceso_gaussiano.ipynb
│   └── 05_comparativo_y_aplicaciones.ipynb
├── tests/                     # pytest: fuga, matemática vs sklearn, métricas
├── web/                       # sitio: HTML + JS que lee registry.json y consume los .onnx
├── informe/                   # markdown fuente, figuras, build a DOCX/PDF
├── docs/                      # diapositivas, acta de decisiones, plan
└── reports/figuras/           # PNG de todo lo que va al informe
```

---

## 3. Fase A · Fundamentos matemáticos (35 % de la nota)

Para cada modelo, el informe debe tener **estas cuatro subsecciones** (así se corresponde con lo que pide el PDF): formulación h_θ(x), criterio de optimización, algoritmo de aprendizaje y supuestos.

### A1. Naive Bayes Gaussiano
- **Formulación:** `ŷ = argmax_k  log P(y=k) + Σ_j log N(x_j ; μ_kj , σ²_kj)`; partir de Bayes y de la independencia condicional.
- **Criterio:** maximizar la verosimilitud conjunta; demostrar que equivale a minimizar la log-verosimilitud negativa.
- **Algoritmo:** solución **cerrada** (μ y σ² por clase y atributo por máxima verosimilitud, sin iterar); explicar `var_smoothing` como estabilizador numérico.
- **Supuestos y consecuencias:** independencia condicional y normalidad; frontera cuadrática; probabilidades mal calibradas (tienden a 0 o 1). Aquí se discute con el dataset de cáncer, donde los atributos están muy correlacionados.
- **Verificación:** implementación en NumPy y comparación numérica contra `GaussianNB` (diferencia < 1e-10).

### A2. MLP clasificador
- **Formulación:** `h(x) = softmax(W₂ σ(W₁x + b₁) + b₂)`; notación para una o varias capas ocultas; teorema de aproximación universal (enunciado, con sus límites).
- **Criterio:** entropía cruzada + regularización L2 (`α`); relación con máxima verosimilitud.
- **Algoritmo:** **backpropagation** deducida paso a paso (regla de la cadena para cada capa) + optimizadores (SGD con momento, Adam, L-BFGS); efecto de la tasa de aprendizaje y de la parada temprana.
- **Supuestos y consecuencias:** sensible a la escala (por eso `StandardScaler` dentro del pipeline), función no convexa (varias semillas), riesgo de sobreajuste.
- **Verificación:** **comprobación de gradientes** por diferencias finitas y una red pequeña en NumPy comparada con `MLPClassifier`.

### A3. Regresión por Proceso Gaussiano
- **Formulación:** `f ~ GP(m, k)`; distribución predictiva `μ* = K*ᵀ(K+σ²I)⁻¹y`, `σ*² = k** − K*ᵀ(K+σ²I)⁻¹K*`; deducir de la condicional de una gaussiana multivariante.
- **Criterio:** maximizar la **log-verosimilitud marginal**: `−½yᵀK⁻¹y − ½log|K| − (n/2)log 2π`, con sus tres términos interpretados (ajuste, complejidad, constante).
- **Algoritmo:** factorización de Cholesky y optimización de los hiperparámetros del núcleo con L-BFGS y reinicios; costo O(n³) en tiempo y O(n²) en memoria.
- **Núcleos:** RBF, Matérn (ν=1,5 y 2,5), RationalQuadratic, + `WhiteKernel`; qué supone cada uno sobre la suavidad de f.
- **Supuestos:** ruido gaussiano, estacionariedad del núcleo. Cuándo falla.
- **Verificación:** GPR desde cero en NumPy comparado contra `GaussianProcessRegressor`; comprobar que la banda ±2σ cubre ~95 % de los puntos de prueba.

### A4. Comparación teórica (tabla del informe)
Modelo generativo (NB) frente a discriminativo (MLP) frente a bayesiano no paramétrico (GPR): qué estima cada uno, función de pérdida, costo computacional, número de parámetros, interpretabilidad, calibración, tratamiento de valores atípicos.

---

## 4. Fase B · Simulación y experimentación (35 % de la nota)

### B1. Notebook 01: datos y protocolo
Carga, descripción, EDA breve (balance de clases, correlaciones, distribuciones), partición 80/20, semillas, comprobación de que no hay fuga. Es la base de todo lo demás.

### B2. Notebook 02: Naive Bayes Gaussiano
- Pipeline y entrenamiento base; **búsqueda `GridSearchCV` sobre `var_smoothing`** (escala logarítmica) y, como experimento, con y sin transformación de potencia (`PowerTransformer`) para acercar los datos a la normalidad.
- Métricas: matriz de confusión, precision, recall, F1, ROC-AUC, PR-AUC, **log-loss, Brier y curva de calibración**.
- Gráficos: frontera de decisión en 2D (PCA o dos atributos) y superficie de probabilidad en 3D, curva de aprendizaje, histogramas con las gaussianas ajustadas, importancia por permutación (NB no trae importancia propia).
- Experimento de supuestos: aumentar la correlación entre atributos con datos sintéticos y mostrar cómo se degrada.

### B3. Notebook 03: MLP clasificador
- **`RandomizedSearchCV`** sobre `hidden_layer_sizes`, `activation` (relu/tanh), `alpha`, `learning_rate_init`, `solver`, `batch_size`, con `early_stopping`.
- Curvas de pérdida de entrenamiento y validación, curvas de aprendizaje, efecto de `alpha` (sobreajuste frente a regularización), estabilidad con 10 semillas.
- Mismas métricas y calibración que el NB; frontera en 2D con `make_moons` y `make_circles`; importancia por permutación.
- Experimentos con nombre (E1 a E4) para que el informe pueda citarlos: sin regularización, con regularización, con parada temprana, con distinta profundidad.

### B4. Notebook 04: GPR
- **`GridSearchCV` sobre la familia de núcleo y `alpha`** (los parámetros continuos del núcleo se optimizan solos por log-verosimilitud marginal, y eso se explica).
- Métricas: **RMSE, MAE, R²** y, por ser probabilístico, **NLPD** y cobertura de los intervalos del 95 %.
- Gráficos: ajuste 1D con banda ±2σ y muestras de la posterior, comparación de núcleos, valores reales frente a predichos con barras de error, residuos, curva de aprendizaje (con tamaños 100, 200, 400… para mostrar el costo O(n³)).
- Experimento de ruido y de datos faltantes: la incertidumbre debe crecer donde no hay datos.

### B5. Notebook 05: comparativo y aplicaciones
- Tabla final con media ± desviación (varias semillas), intervalos bootstrap y pruebas estadísticas sencillas (McNemar o t pareada) para saber si NB y MLP difieren de verdad.
- Costo: tiempo de entrenamiento y de predicción, memoria, y escalamiento con n.
- **Experimento de sensibilidad a valores atípicos** (lo pide el PDF para justificar cada aplicación):
  - *Protocolo:* sobre el conjunto de **entrenamiento** se contamina una fracción creciente de filas (0 %, 1 %, 2 %, 5 %, 10 %) con atípicos, de dos tipos: (a) atípicos en los atributos (valores multiplicados por 5 a 10 desviaciones estándar) y (b) ruido en la etiqueta o en el objetivo (etiquetas invertidas en clasificación; valores extremos en regresión). El conjunto de prueba **no se contamina**.
  - *Qué se mide:* ROC-AUC y F1 (NB y MLP) y RMSE y R² (GPR), con 10 semillas y banda de dispersión; curva «degradación frente a % de contaminación» para cada modelo. Para el GPR se compara el núcleo RBF con un ruido fijo frente a Matérn, y se interpreta el efecto sobre la banda de incertidumbre.
  - *Qué se espera discutir:* NB desplaza medias y varianzas (los atípicos inflan σ²); el MLP puede memorizar el ruido si no hay regularización; el GPR gaussiano es sensible porque el ruido gaussiano no admite colas pesadas. Se menciona como alternativa el proceso con verosimilitud Student-t, sin implementarlo.
  - *Resultado en el informe:* una figura y una tabla, que respaldan la fila «robustez a atípicos» de la comparación crítica.
- **Interpretabilidad (evidencia, no solo opinión):** parámetros leídos directamente de NB (medias y varianzas por clase), importancia por permutación en NB y MLP, y longitudes de escala del núcleo (relevancia automática) en el GPR.
- Discusión de qué modelo elegir según el caso.

### B6. Reproducibilidad
`requirements.txt` fijo, semillas, un solo comando para rehacer todos los resultados (`python -m familia3.todo`), notebooks ejecutados sin errores, resultados guardados en `reports/`.

---

## 5. Fase C · Aplicaciones y casos de uso (20 % de la nota)

Un caso real por modelo, cada uno con **justificación técnica** (sensibilidad a valores atípicos, interpretabilidad, costo, calibración, necesidad de incertidumbre):

| Modelo | Caso de estudio | Por qué encaja |
|---|---|---|
| Naive Bayes Gaussiano | Filtrado y triaje rápido de diagnóstico, detección temprana de fallos en sensores | Entrenamiento instantáneo, funciona con pocos datos, probabilidades interpretables; riesgo: independencia falsa y mala calibración |
| MLP | Diagnóstico médico por imagen o por datos clínicos, detección de fraude | Captura relaciones no lineales; pierde interpretabilidad y exige más datos; sensible a la escala |
| Proceso Gaussiano | Optimización bayesiana de hiperparámetros, geoestadística (kriging), calibración de sensores | Da incertidumbre calibrada con pocos datos; costo O(n³) lo limita a miles de puntos |

Cada caso se acompaña de una referencia académica o técnica real (no inventada) y de un mini-experimento con nuestros datos que lo ilustre. En el notebook 05 hay una **comparación crítica** en una tabla: datos necesarios, costo, interpretabilidad, calibración, robustez a atípicos, cuándo elegir cada uno.

---

## 6. Fase D · Simulación interactiva: la web consume los modelos entrenados (Netlify)

**Arranque de la página** (mismo patrón que X-ray): lee `registry.json` → elige la versión vigente (o `?version=`) → descarga cada `.onnx` → verifica el SHA-256 → crea la sesión de ONNX Runtime Web → muestra una insignia «modelo vX · hash verificado». Si algo falla, lo dice con claridad.

**Pestañas del sitio** (todo en el navegador, sin servidor; los modelos son los entrenados en Python):
1. **Naive Bayes:** elegir un paciente de ejemplo del conjunto de prueba o mover los atributos principales con deslizadores; el modelo `nb_cancer` devuelve clase y probabilidades. Con `nb_moons` se dibuja la **frontera y el mapa de probabilidad** consultando el modelo sobre una malla 2D. Se muestran medias y varianzas por clase leídas del registro.
2. **MLP:** lo mismo con `mlp_cancer` y `mlp_moons`; se muestra la arquitectura y la curva de pérdida del entrenamiento (guardada en el registro), y se comparan las versiones v1 y v2.
3. **Proceso Gaussiano:** con `gpr_diabetes`, deslizadores de los atributos y predicción **media ± 2σ**; con `gpr_seno1d`, curva ajustada y **banda de incertidumbre** que crece donde no hay datos, dibujada consultando el modelo.
4. **Comparativo:** el mismo paciente evaluado por NB y MLP en paralelo, con sus probabilidades y la tabla de métricas del registro (test, intervalos); aviso sobre calibración.
5. **Teoría:** las ecuaciones de cada modelo con su explicación breve (renderizadas, p. ej. con KaTeX) y enlaces al informe y al repositorio.

**Calidad:** diseño cuidado, funciona en móvil, modo oscuro, accesible por teclado, y un aviso de que es una herramienta didáctica y no un diagnóstico médico. Pruebas: (a) en Python, paridad scikit-learn ↔ ONNX; (b) en el navegador o Node, los mismos casos de prueba exportados deben dar lo mismo (ya comprobado de forma manual para los 3 modelos); (c) `verificar_release.py` comprueba que la página no tiene modelo, versión ni umbral escritos a mano.
**Despliegue:** carpeta `dist/` arrastrada a **un sitio nuevo de Netlify** (para no pisar el de X-ray, que ya está en `californication1`), con un `netlify.toml` sin compilación y `_headers` como en X-ray. Hace falta que ustedes elijan el nombre del sitio.

---

## 7. Fase E · Informe técnico (PDF)

**Estructura:**
0. Carátula (curso, docente, equipo, integrantes, fecha) · Resumen · Índice · Índice de figuras y tablas
1. Introducción y objetivos (los 3 del PDF) y alcance
2. Marco teórico y fundamentos matemáticos (A1 a A4)
3. Datos y metodología (datasets, protocolo, métricas y su justificación, búsqueda de hiperparámetros)
4. Experimentos y resultados por modelo (notebooks 02 a 04)
5. Comparación de los tres modelos (notebook 05)
6. Aplicaciones y casos de uso, con justificación técnica
7. Simulación interactiva (capturas, guía de uso, enlace)
8. Conclusiones y trabajo futuro
Referencias (formato APA, fuentes reales: Bishop *Pattern Recognition and ML*; Murphy *Probabilistic ML*; Rasmussen y Williams *Gaussian Processes for ML*; Goodfellow et al. *Deep Learning*; documentación de scikit-learn; los artículos originales)
Anexos: derivaciones completas · tablas de hiperparámetros probados · enlaces (GitHub, Netlify, diapositivas) · guía de reproducción · matriz de trazabilidad con la rúbrica

**Formato:** Word + PDF, con las mismas herramientas que ya usamos en X-ray (Markdown → DOCX → PDF), ecuaciones bien maquetadas, figuras numeradas, tablas legibles.
**Diapositivas:** una presentación corta (≈ 15 a 20) con los resultados y una demo de la web; el PDF no las exige, pero el 10 % de "exposición" las pide en la práctica.

---

## 8. Matriz de trazabilidad: PDF frente al proyecto

| Pide el PDF | Dónde se cumple |
|---|---|
| Formulación / hipótesis h_θ(x) | Informe 2.x.1; notebook 0x, primera celda |
| Criterio de optimización | Informe 2.x.2 |
| Algoritmo de aprendizaje | Informe 2.x.3 + implementación desde cero |
| Preprocesamiento (escalado, categóricas, división) | `pipelines.py`, informe 3.2 |
| Búsqueda de hiperparámetros (Grid/Randomized) | `busqueda.py`, notebooks 02 a 04 |
| Matriz de confusión, Precision, Recall, F1, ROC-AUC | notebooks 02 y 03 |
| RMSE y R² | notebook 04 |
| Fronteras de decisión 2D/3D | notebooks 02 y 03, web |
| Curvas de aprendizaje / importancia de características | notebooks 02 a 04 |
| Caso de estudio real por modelo y justificación técnica | informe cap. 6 |
| Justificar por sensibilidad a *outliers*, interpretabilidad y costo computacional | Notebook 05: experimento de atípicos, evidencia de interpretabilidad y curvas de costo; informe cap. 5 y 6 |
| Comparativa entre los 3 modelos | notebook 05, informe cap. 5 |
| Informe PDF | `informe/INFORME_TECNICO.pdf` |
| Código fuente en repositorio | GitHub: Calamar49/Comparativo-de-Familias |
| Simulación interactiva | Sitio Netlify (sección 6) |
| **El modelo entrenado debe poder consumirse en la aplicación web** | `exportar.py` → `models/registry.json` + `.onnx` → `publicar.py` → la web los descarga, verifica el hash y predice; informe cap. 7 |

---

## 9. Cronograma por etapas (esfuerzo relativo; falta la fecha de entrega para asignar días)

| Etapa | Contenido | Esfuerzo |
|---|---|---|
| 0 | Confirmar datos del equipo, fecha de entrega y nombre del sitio; crear repositorio y estructura | Pequeño |
| 1 | `src/` base: datos, protocolo, métricas; notebook 01 | Medio |
| 2 | Notebooks 02, 03 y 04 con sus implementaciones desde cero y tests de matemática | **Grande** |
| 3 | Notebook 05, tablas finales, pruebas estadísticas | Medio |
| 4a | Exportar a ONNX, registrar en `registry.json` y probar paridad Python ↔ ONNX | Medio |
| 4b | Sitio web que lee el registro y consume los modelos; `publicar.py` y `verificar_release.py` | **Grande** |
| 5 | Informe, figuras definitivas, diapositivas | **Grande** |
| 6 | Revisión cruzada contra rúbrica, CI en verde, despliegue, publicación | Pequeño |

Reparto sugerido si son 4 personas: una por modelo (NB, MLP, GPR) con su notebook, su capítulo y su pestaña de la web; la cuarta persona integra, escribe el comparativo y el informe. Cada persona entiende el modelo que presenta, que es lo que el docente suele preguntar.

---

## 10. Riesgos y cómo los controlamos

| Riesgo | Control |
|---|---|
| GPR muy lento con muchos datos | Submuestrear a ≤ 2 000, justificarlo y mostrar la curva de costo |
| MLP no converge o da avisos | Escalado, `max_iter` alto, parada temprana, varias semillas, informar la variabilidad |
| Resultados "demasiado buenos" por fuga | Pipelines, validación cruzada solo en entrenamiento, prueba final única, test de fuga |
| ONNX distinto de scikit-learn, o núcleo del GPR no soportado | Prueba de paridad Python ↔ ONNX en el CI; solo se exportan los núcleos verificados (RBF y Matérn; RationalQuadratic queda excluido) |
| Un operador de ONNX no corre en el navegador | Ya verificado con los 3 modelos en ONNX Runtime Web 1.22; el CI repite la prueba con los casos exportados |
| Modelo del GPR pesado o lento en el navegador | Limitar a ≤ 2 000 filas, medir tamaño y tiempo de predicción e informarlos |
| La web usa un modelo distinto del evaluado | Registro con SHA-256, verificación en el navegador y `verificar_release.py` |
| Informe demasiado largo o desordenado | Plantilla fija, matriz de trazabilidad, revisión con la rúbrica en la mano |
| Ecuaciones mal escritas | Revisión cruzada y comprobación numérica contra la implementación desde cero |
| Mala calibración mal interpretada | Curvas de calibración y Brier; explicar por qué NB y MLP difieren |
| Copia o texto sin fuente | Referencias reales y verificables; redacción propia |

---

## 11. Definición de "terminado" (checklist final)
- [ ] El experimento de atípicos, la evidencia de interpretabilidad y las curvas de costo están en el notebook 05 y en el informe.
- [ ] Cada modelo tiene: formulación, criterio, algoritmo, supuestos, implementación desde cero verificada, búsqueda de hiperparámetros, métricas, gráficos y caso de uso.
- [ ] Los 5 notebooks se ejecutan de principio a fin sin errores, con salidas guardadas.
- [ ] `pytest`, la paridad Python ↔ ONNX y `verificar_release.py` en verde, y CI en verde en GitHub.
- [ ] **Los 3 modelos entrenados están registrados y la web los descarga, verifica y usa**, sin modelo, versión ni umbral escritos a mano en la página.
- [ ] El informe PDF cubre todo el PDF de la actividad (matriz de trazabilidad completa) y cita fuentes reales.
- [ ] El sitio de Netlify funciona en escritorio y móvil, y se explica en el informe.
- [ ] Repositorio con README claro, enlaces correctos (informe, web, diapositivas) y una sola orden para reproducir.
- [ ] Revisión final contra la rúbrica: 35 / 35 / 20 / 10.

---

## 12. Datos que necesitamos del equipo antes de empezar
1. Nombres de los integrantes del equipo 5 (para la carátula) y confirmar el docente y el ciclo.
2. Fecha de entrega y si hay exposición oral.
3. Nombre deseado para el sitio de Netlify.
4. (Resuelto) La web consume los modelos entrenados en Python, exportados a ONNX. Solo queda confirmar con el docente si además quiere un laboratorio que **entrene** en el navegador (extra opcional).
