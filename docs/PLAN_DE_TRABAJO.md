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
| Simulación interactiva | Sitio web en Netlify donde se juegan los 3 modelos en el navegador |

| Criterio de la rúbrica | Peso | Cómo lo cubrimos | Dónde se ve |
|---|---|---|---|
| Rigor matemático | 35 % | Ecuaciones, pérdida, supuestos y algoritmo de los 3 modelos, con derivaciones y verificación numérica | Informe cap. 2, anexo A, notebooks 01 a 04 |
| Simulación y código | 35 % | Pipelines sin fuga, búsqueda de hiperparámetros, métricas justificadas, semillas, tests, CI | `src/`, notebooks, `tests/`, informe cap. 3 a 5 |
| Análisis de aplicaciones | 20 % | Un caso real por modelo + comparación crítica de los 3 | Informe cap. 6 y 7, notebook 05 |
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

### 1.3 Cómo será la simulación interactiva
Opción elegida: **los tres modelos programados desde cero en JavaScript**, corriendo en el navegador, sin servidor.
- Por qué: muestra las ecuaciones funcionando (suma puntos en rigor matemático), carga al instante y es 100 % estática (Netlify).
- Riesgo: que el código JavaScript no coincida con scikit-learn. **Mitigación:** una prueba de paridad. Python exporta casos de prueba a JSON y un script de Node comprueba que el JavaScript da lo mismo (mismo estilo que la paridad Keras/ONNX de X-ray).
- Alternativa descartada: Pyodide (scikit-learn real en el navegador). Descarga 20 a 30 MB, tarda en cargar y no deja ver la matemática. Queda como plan B si el JS se complicara.

---

## 2. Estructura del repositorio

```
Comparativo-de-Familias/
├── README.md                  # qué es, cómo reproducir, enlaces, tabla de resultados
├── requirements.txt           # versiones fijas
├── .github/workflows/ci.yml   # tests + build del sitio + parity JS
├── data/                      # cachés pequeños y README de procedencia
├── src/familia3/
│   ├── datos.py               # carga, particiones, semillas
│   ├── pipelines.py           # Pipeline por modelo (escalado + estimador)
│   ├── busqueda.py            # Grid/RandomizedSearch y rejillas
│   ├── metricas.py            # clasificación, regresión, calibración, bootstrap
│   ├── graficos.py            # fronteras, curvas, importancia, GPR con banda
│   ├── desde_cero.py          # NB, MLP y GPR en NumPy (verificación matemática)
│   └── exportar_web.py        # fixtures JSON para la paridad con el JS
├── notebooks/
│   ├── 01_datos_y_protocolo.ipynb
│   ├── 02_naive_bayes_gaussiano.ipynb
│   ├── 03_mlp_clasificador.ipynb
│   ├── 04_regresion_proceso_gaussiano.ipynb
│   └── 05_comparativo_y_aplicaciones.ipynb
├── tests/                     # pytest: fuga, matemática vs sklearn, métricas
├── web/                       # sitio de la simulación (HTML + JS + datos)
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

## 6. Fase D · Simulación interactiva (sitio en Netlify)

**Pestañas del sitio** (todo en el navegador, sin servidor):
1. **Naive Bayes:** el usuario hace clic para colocar puntos de 2 o 3 clases; se dibujan las gaussianas por clase, la frontera y el mapa de probabilidad; controles de `var_smoothing`.
2. **MLP:** capas y neuronas ajustables, activación, tasa de aprendizaje y `α`; entrena en vivo, con curva de pérdida y frontera que se actualiza por época.
3. **Proceso Gaussiano:** clic para colocar puntos; núcleo (RBF, Matérn, RQ), longitud de escala y ruido ajustables; se ve la media, la banda ±2σ y muestras de la posterior.
4. **Comparativo:** los mismos datos sintéticos con NB y MLP en paralelo (con GPR en su pestaña, porque es regresión), y la tabla de resultados reales de los notebooks.
5. **Teoría:** las ecuaciones de cada modelo con su explicación breve (renderizadas, p. ej. con KaTeX) y enlaces al informe y al repositorio.

**Calidad:** diseño cuidado, funciona en móvil, modo oscuro, accesible por teclado, aviso de la parte "didáctica". Paridad con scikit-learn verificada por pruebas (`web/tests/paridad.mjs`) en el CI.
**Despliegue:** carpeta `dist/` arrastrada a **un sitio nuevo de Netlify** (para no pisar el de X-ray, que ya está en `californication1`). Hace falta que ustedes elijan el nombre del sitio.

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
| Comparativa entre los 3 modelos | notebook 05, informe cap. 5 |
| Informe PDF | `informe/INFORME_TECNICO.pdf` |
| Código fuente en repositorio | GitHub: Calamar49/Comparativo-de-Familias |
| Simulación interactiva | Sitio Netlify |

---

## 9. Cronograma por etapas (esfuerzo relativo; falta la fecha de entrega para asignar días)

| Etapa | Contenido | Esfuerzo |
|---|---|---|
| 0 | Confirmar datos del equipo, fecha de entrega y nombre del sitio; crear repositorio y estructura | Pequeño |
| 1 | `src/` base: datos, protocolo, métricas; notebook 01 | Medio |
| 2 | Notebooks 02, 03 y 04 con sus implementaciones desde cero y tests de matemática | **Grande** |
| 3 | Notebook 05, tablas finales, pruebas estadísticas | Medio |
| 4 | Sitio web + implementación JS + paridad | **Grande** |
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
| JavaScript distinto de scikit-learn | Fixtures y prueba de paridad en el CI |
| Informe demasiado largo o desordenado | Plantilla fija, matriz de trazabilidad, revisión con la rúbrica en la mano |
| Ecuaciones mal escritas | Revisión cruzada y comprobación numérica contra la implementación desde cero |
| Mala calibración mal interpretada | Curvas de calibración y Brier; explicar por qué NB y MLP difieren |
| Copia o texto sin fuente | Referencias reales y verificables; redacción propia |

---

## 11. Definición de "terminado" (checklist final)
- [ ] Cada modelo tiene: formulación, criterio, algoritmo, supuestos, implementación desde cero verificada, búsqueda de hiperparámetros, métricas, gráficos y caso de uso.
- [ ] Los 5 notebooks se ejecutan de principio a fin sin errores, con salidas guardadas.
- [ ] `pytest` y la paridad JS en verde, y CI en verde en GitHub.
- [ ] El informe PDF cubre todo el PDF de la actividad (matriz de trazabilidad completa) y cita fuentes reales.
- [ ] El sitio de Netlify funciona en escritorio y móvil, y se explica en el informe.
- [ ] Repositorio con README claro, enlaces correctos (informe, web, diapositivas) y una sola orden para reproducir.
- [ ] Revisión final contra la rúbrica: 35 / 35 / 20 / 10.

---

## 12. Datos que necesitamos del equipo antes de empezar
1. Nombres de los integrantes del equipo 5 (para la carátula) y confirmar el docente y el ciclo.
2. Fecha de entrega y si hay exposición oral.
3. Nombre deseado para el sitio de Netlify.
4. Si el docente quiere que la simulación use los modelos reales de Python (Pyodide) o si basta con una versión interactiva programada en JavaScript.
