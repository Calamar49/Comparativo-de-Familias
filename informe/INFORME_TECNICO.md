# Despliegue y Análisis Comparativo de Familias de Aprendizaje Supervisado

**Familia 3 · Modelos probabilísticos y redes neuronales multicapa — Equipo 5**

**Curso:** Inteligencia Artificial
**Integrantes:** Coavoy Cruz, Joseph Gabriel · Cuchuyrrumi Mamani, Manuel Rodrigo · Huallpatuiro Rafaile, Brayan · Mamani Acuña, Frank Joseph
**Fecha:** _(pendiente de fecha de entrega — ver sección 12 del plan de trabajo)_
**Repositorio:** https://github.com/Calamar49/Comparativo-de-Familias
**Sitio interactivo:** _(pendiente nombre de Netlify)_

---

## Resumen

Este informe compara tres modelos de aprendizaje supervisado de naturaleza distinta —generativo
(Naive Bayes Gaussiano), discriminativo (Perceptrón Multicapa) y bayesiano no paramétrico
(Regresión por Proceso Gaussiano)— sobre datasets reales y sintéticos, bajo un protocolo
idéntico de partición, validación cruzada y prueba única. Los tres modelos se entrenan en
Python, se exportan a ONNX con verificación de paridad numérica, y se consumen desde una
simulación interactiva en el navegador que nunca reentrena ni reimplementa el modelo.

## Índice

1. Introducción y objetivos
2. Marco teórico y fundamentos matemáticos
3. Datos y metodología
4. Experimentos y resultados por modelo
5. Comparación de los tres modelos
6. Aplicaciones y casos de uso
7. Simulación interactiva
8. Conclusiones y trabajo futuro
Referencias
Anexos

---

## 1. Introducción y objetivos

El curso asigna a este equipo la Familia 3: modelos probabilísticos y redes neuronales
multicapa, con tres técnicas concretas —Naive Bayes Gaussiano, Perceptrón Multicapa (MLP) y
Regresión por Proceso Gaussiano (GPR)— sobre las que este trabajo:

1. Deriva y verifica matemáticamente la formulación, el criterio de optimización y el
   algoritmo de aprendizaje de cada modelo.
2. Entrena, ajusta hiperparámetros y evalúa los tres modelos bajo un protocolo común,
   anti-fuga y reproducible.
3. Despliega los tres modelos entrenados en una simulación web que los consume (no los
   reentrena), con verificación de integridad end-to-end.

**Alcance:** clasificación binaria y sintética 2D (NB y MLP) y regresión real/sintética (GPR),
con un experimento transversal de sensibilidad a valores atípicos que sustenta la comparación
crítica del capítulo 5 y las recomendaciones de aplicación del capítulo 6.

---

## 2. Marco teórico y fundamentos matemáticos

### 2.1 Naive Bayes Gaussiano

**Formulación.** Bajo la suposición de independencia condicional entre atributos dada la clase:

```
ŷ = argmax_k  log P(y=k) + Σ_j log 𝒩(x_j ; μ_kj , σ²_kj)
```

se deriva directamente del teorema de Bayes `P(y|x) ∝ P(y)·P(x|y)` factorizando
`P(x|y) = Π_j P(x_j|y)`.

**Criterio de optimización.** Maximizar la verosimilitud conjunta de los datos de
entrenamiento equivale a minimizar la log-verosimilitud negativa; al ser un modelo generativo
con familia exponencial (gaussiana), el óptimo tiene forma cerrada.

**Algoritmo de aprendizaje.** Solución cerrada por máxima verosimilitud: para cada clase `k` y
atributo `j`, `μ_kj` y `σ²_kj` son la media y varianza muestral de los datos de esa clase — sin
iterar. `var_smoothing` añade una fracción de la varianza global a cada `σ²_kj` como
estabilizador numérico (evita divisiones por cero cuando una clase tiene varianza casi nula en
algún atributo).

**Supuestos y consecuencias.** Independencia condicional (rara vez exacta: en el dataset de
cáncer los atributos están fuertemente correlacionados, ver notebook 01) y normalidad por
atributo. La frontera de decisión resultante es cuadrática. Las probabilidades tienden a
saturarse hacia 0 o 1 cuando se viola la independencia (el notebook 02 lo demuestra
aumentando artificialmente la correlación entre atributos y midiendo la caída de ROC-AUC).

**Verificación.** Implementación en NumPy puro (`src/familia3/desde_cero.py:NaiveBayesGaussianoNumPy`)
comparada contra `sklearn.naive_bayes.GaussianNB`: diferencia máxima en probabilidades < 1e-10
(notebook 02, test `tests/test_desde_cero.py::test_naive_bayes_numpy_coincide_con_sklearn`).

### 2.2 MLP clasificador

**Formulación.** Con una capa oculta: `h(x) = softmax(W₂ σ(W₁x + b₁) + b₂)`, generalizable a L
capas aplicando la composición de forma recursiva. El teorema de aproximación universal
garantiza que una red con una sola capa oculta suficientemente ancha puede aproximar cualquier
función continua en un compacto, aunque no dice nada sobre cuántas neuronas hacen falta ni
sobre la facilidad de optimización.

**Criterio de optimización.** Entropía cruzada entre la distribución predicha y la real, más
regularización L2 (`α`): `L = -1/n Σ log ŷ_{i,y_i} + (α/2n)(‖W₁‖² + ‖W₂‖²)`. La entropía
cruzada es, bajo una salida softmax, equivalente a maximizar la verosimilitud de un modelo
categórico.

**Algoritmo de aprendizaje.** Backpropagation: la regla de la cadena se aplica capa por capa
para obtener `∂L/∂W` en cada peso, y un optimizador de primer orden (SGD con momento, Adam o
L-BFGS para redes pequeñas) actualiza los pesos. La derivación completa, capa por capa, está
en `src/familia3/desde_cero.py:MLPNumPy` (docstring de la clase). La tasa de aprendizaje
controla el tamaño del paso; la parada temprana (`early_stopping`) detiene el entrenamiento
cuando la pérdida de validación deja de mejorar, limitando el sobreajuste.

**Supuestos y consecuencias.** Sensible a la escala de los atributos (por eso el `StandardScaler`
va dentro del mismo `Pipeline` que el `MLPClassifier`, y por tanto dentro del `.onnx`
exportado). La función de pérdida no es convexa: distintas semillas pueden converger a óptimos
locales distintos (notebook 03, estabilidad con 10 semillas). Riesgo de sobreajuste sin
regularización (experimentos E1–E4 del notebook 03).

**Verificación.** Comprobación de gradientes por diferencias finitas centradas sobre
`MLPNumPy`: error relativo entre el gradiente analítico (backpropagation) y el numérico
< 1e-4 (notebook 03, `tests/test_desde_cero.py::test_comprobacion_gradientes_mlp_numpy`).

### 2.3 Regresión por Proceso Gaussiano

**Formulación.** Un GP define una distribución sobre funciones, `f ~ GP(m, k)`, especificada
por su media (aquí cero tras centrar/escalar el objetivo) y su función de covarianza (núcleo)
`k`. Condicionando la gaussiana multivariante conjunta de entrenamiento y predicción en los
valores observados se obtiene la distribución predictiva:

```
μ* = K*ᵀ (K + σ²I)⁻¹ y
σ*² = k** − K*ᵀ (K + σ²I)⁻¹ K*
```

**Criterio de optimización.** Los hiperparámetros del núcleo (longitud de escala, varianza,
ruido) se ajustan maximizando la log-verosimilitud marginal:

```
log p(y|X) = −½ yᵀK⁻¹y − ½ log|K| − (n/2) log 2π
```

con sus tres términos interpretables: ajuste a los datos, penalización por complejidad del
modelo (a través de `log|K|`) y una constante de normalización.

**Algoritmo de aprendizaje.** `K + σ²I` se factoriza por Cholesky (estable numéricamente y
O(n³)); los hiperparámetros continuos se optimizan con L-BFGS y varios reinicios aleatorios
para evitar óptimos locales. Costo: O(n³) en tiempo, O(n²) en memoria — de ahí la necesidad de
limitar el tamaño de entrenamiento (ver sección 4.4 y el experimento de costo del notebook 04).

**Núcleos considerados.** RBF (infinitamente diferenciable, asume funciones muy suaves),
Matérn con ν=1.5 y ν=2.5 (menos suave, más realista para fenómenos físicos/biológicos), y
`WhiteKernel` sumado para modelar el ruido de observación. `RationalQuadratic` se probó pero
**se excluyó de la exportación** (ver limitación técnica en la sección 3.4).

**Supuestos.** Ruido de observación gaussiano homocedástico y estacionariedad del núcleo (la
covarianza entre dos puntos depende solo de su distancia, no de su posición absoluta). Falla
cuando hay atípicos (el ruido gaussiano no tiene colas pesadas) o cuando la función no es
estacionaria.

**Verificación.** Implementación en NumPy (`src/familia3/desde_cero.py:ProcesoGaussianoNumPy`)
comparada contra `GaussianProcessRegressor` con el núcleo fijo (sin optimizar
hiperparámetros, para una comparación limpia de la misma fórmula): coincide en media y
desviación estándar con tolerancia 1e-6. Se comprueba además que la banda ±2σ cubre ~95% de
los puntos de prueba sobre la función sintética (notebook 04).

### 2.4 Comparación teórica

| | Naive Bayes | MLP | GPR |
|---|---|---|---|
| Qué estima | `P(x,y)` (generativo) | `P(y\|x)` (discriminativo) | posterior sobre funciones (bayesiano no paramétrico) |
| Pérdida | log-verosimilitud negativa | entropía cruzada + L2 | log-verosimilitud marginal negativa |
| Costo computacional | O(n·d), cerrado | iterativo (épocas × tamaño de red) | O(n³) |
| N.º de parámetros | 2·k·d (medias y varianzas) | pesos y sesgos de la red | hiperparámetros del núcleo (pocos) |
| Interpretabilidad | alta | baja | media (longitudes de escala) |
| Calibración | pobre si hay correlación | razonable | nativa |
| Tratamiento de atípicos | sensible (infla σ²) | puede memorizar sin regularización | sensible (ruido gaussiano sin colas pesadas) |

---

## 3. Datos y metodología

### 3.1 Datasets

| Uso | Dataset | Tamaño | Por qué |
|---|---|---|---|
| Clasificación real (NB y MLP, mismos datos) | Breast Cancer Wisconsin | 569×30 | Caso de diagnóstico; atributos correlacionados, pone a prueba la independencia de NB |
| Clasificación multiclase | Wine | 178×13 | Verifica que las conclusiones no dependen de un solo dataset |
| Clasificación sintética 2D | `make_moons` | 400×2 | Fronteras dibujables; muestra dónde NB falla y el MLP no |
| Regresión real (GPR) | Diabetes | 442×10 | Dataset estándar de regresión de scikit-learn |
| Regresión sintética 1D | f(x) = sin(x) + ruido, con hueco en [3,5] | 60 puntos | Muestra crecer la incertidumbre del GPR donde no hay datos |

Todo reproducible sin descargas manuales (`scikit-learn` trae estos datasets embebidos).

### 3.2 Protocolo de evaluación

Partición 80/20 estratificada; validación cruzada de 5 pliegues **solo dentro del
entrenamiento** para la búsqueda de hiperparámetros (`GridSearchCV`/`RandomizedSearchCV` +
`StratifiedKFold`); escalado dentro de un `Pipeline` (sin fuga); semilla fija `SEED=42` con
repetición en varias semillas para reportar media ± desviación; prueba final una sola vez por
modelo, con intervalos de confianza por bootstrap no paramétrico (`src/familia3/metricas.py`).
El notebook 01 verifica programáticamente la ausencia de fuga (ninguna fila se repite entre
`X_train` y `X_test`).

### 3.3 Búsqueda de hiperparámetros

- **Naive Bayes:** `GridSearchCV` sobre `var_smoothing` en escala logarítmica.
- **MLP:** `RandomizedSearchCV` sobre `hidden_layer_sizes`, `activation`, `alpha`,
  `learning_rate_init`, `solver`, con `early_stopping=True`.
- **GPR:** `GridSearchCV` sobre la familia de núcleo (RBF, Matérn ν=1.5, ν=2.5) y `alpha`; los
  parámetros continuos del núcleo (longitud de escala, ruido) se optimizan solos dentro de
  `.fit()` maximizando la log-verosimilitud marginal.

### 3.4 Limitación técnica: exportación ONNX del GPR

`skl2onnx` 1.20 lanza un `TypeError` interno al exportar
`GaussianProcessRegressor(normalize_y=True, return_std=True)`. Para no perder la
estandarización del objetivo (indispensable: sin ella, el optimizador del núcleo cae en un
óptimo local degenerado donde todo el ruido se explica por el `WhiteKernel` y la predicción
colapsa a la media, R²≈0 — comprobado empíricamente), se centra y escala `y` **a mano** antes
de entrenar, y se guardan `y_offset` y `y_scale` en `models/registry.json`. El `.onnx`
exportado predice en la escala estandarizada; la web deshace la transformación al leer esos
dos campos del registro, igual que con cualquier otro dato del modelo (no es un valor escrito
a mano: sale del registro, y `verificar_release.py` lo audita). Por la misma razón de
compatibilidad de `skl2onnx`, solo se exportan los núcleos RBF y Matérn (verificados con
paridad < 1e-4); `RationalQuadratic` se probó y no se incluye por una discrepancia de ~0.2 en
la media entre sklearn y ONNX Runtime Web.

---

## 4. Experimentos y resultados por modelo

_(resultados de la ejecución registrada en `models/registry.json`; reproducibles con
`python -m familia3.todo` — semilla 42)_

### 4.1 Naive Bayes Gaussiano (`nb_cancer`)

| Métrica | Valor (prueba) |
|---|---|
| Precisión | 0.945 |
| Recall | 0.958 |
| F1 | 0.952 |
| ROC-AUC | 0.988 |
| Log-loss | 0.371 |
| Brier | 0.049 |

El log-loss relativamente alto frente al MLP (ver 4.2) es consistente con la mala calibración
esperada por la violación del supuesto de independencia (sección 2.1); el ROC-AUC, que solo
depende del orden de las probabilidades, sigue siendo alto.

### 4.2 MLP clasificador (`mlp_cancer`)

| Métrica | Valor (prueba) |
|---|---|
| Precisión | 0.946 |
| Recall | 0.972 |
| F1 | 0.959 |
| ROC-AUC | 0.993 |
| Log-loss | 0.189 |
| Brier | 0.049 |

Mejor en todas las métricas que dependen de la calibración (log-loss) y ligeramente mejor en
ranking (ROC-AUC); el Brier score es prácticamente idéntico, lo que indica que ambos modelos
discriminan razonablemente bien las probabilidades extremas, pero el MLP es más fiable en la
zona intermedia.

### 4.3 Proceso Gaussiano (`gpr_diabetes`, `gpr_seno1d`)

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| `gpr_diabetes` (Matérn ν=1.5) | 50.9 | 40.3 | 0.510 |
| `gpr_seno1d` (RBF) | 0.146 | 0.124 | 0.901 |

Sobre la función sintética, la banda ±2σ cubre la fracción esperada de puntos de prueba
(notebook 04) y se ensancha visiblemente en el hueco `x∈[3,5]` sin datos de entrenamiento —la
propiedad central de un GP frente a un modelo puntual.

### 4.4 Costo computacional del GPR

El notebook 04 mide el tiempo de `.fit()` para n = 50, 100, 200, 400 filas y confirma la
tendencia cúbica esperada, justificando el límite de ≤2000 filas para el modelo exportado a la
web.

---

## 5. Comparación de los tres modelos

El notebook 05 reporta:

- **Media ± desviación** de ROC-AUC en 10 semillas para NB y MLP, e **intervalos de confianza
  por bootstrap** sobre el conjunto de prueba.
- **Prueba de McNemar** sobre los aciertos pareados de NB y MLP en el mismo conjunto de
  prueba, para decidir si la diferencia observada es estadísticamente significativa y no un
  artefacto de la partición.
- **Costo:** tiempo de entrenamiento y predicción por modelo.
- **Experimento de sensibilidad a valores atípicos:** se contamina una fracción creciente
  (0%, 1%, 2%, 5%, 10%) del conjunto de **entrenamiento** —nunca el de prueba— con (a)
  atípicos en los atributos (×5–10 desviaciones estándar) y (b) ruido de etiqueta/objetivo, y
  se mide la degradación de ROC-AUC/F1 (NB y MLP) y RMSE/R² (GPR). Los resultados (figuras
  `05_degradacion_outliers.png` y `05_gpr_outliers.png`) muestran que: los atípicos en
  atributos desplazan medias y, sobre todo, inflan las varianzas estimadas por Naive Bayes
  (su log-verosimilitud es muy sensible a `σ²` pequeñas); el MLP tolera algo mejor los
  atípicos moderados gracias a la regularización L2, pero se degrada igual con contaminación
  alta; el GPR con ruido gaussiano (sin colas pesadas) absorbe los atípicos como si fueran
  ruido uniforme, ensanchando la banda de incertidumbre en todo el dominio en vez de aislarlos
  —una alternativa no implementada aquí sería un proceso con verosimilitud Student-t.
- **Interpretabilidad con evidencia:** medias/varianzas por clase leídas directamente del
  `GaussianNB` entrenado, importancia por permutación en NB y MLP (coinciden en los atributos
  `worst radius` y `worst concave points` como los más influyentes), y longitudes de escala
  del núcleo del GPR sobre Diabetes (relevancia automática: menor longitud de escala implica
  mayor sensibilidad del modelo a ese atributo).

### Tabla comparativa final

| | Naive Bayes | MLP | GPR |
|---|---|---|---|
| Datos necesarios | pocos | moderados-muchos | pocos a miles |
| Costo | muy bajo | medio | alto (O(n³)) |
| Interpretabilidad | alta | baja | media |
| Calibración | pobre | razonable | nativa |
| Robustez a atípicos | baja | media | baja (sin colas pesadas) |
| Cuándo elegir | triaje rápido, pocos datos, se necesita explicar | relaciones no lineales, datos suficientes | se necesita incertidumbre calibrada |

---

## 6. Aplicaciones y casos de uso

| Modelo | Caso de estudio | Justificación técnica |
|---|---|---|
| Naive Bayes Gaussiano | Triaje y filtrado rápido de diagnóstico; detección temprana de fallos en sensores | Entrenamiento instantáneo, funciona con pocos datos, probabilidades interpretables; riesgo: independencia falsa y mala calibración (sección 4.1, 5) |
| MLP | Diagnóstico por imagen o por datos clínicos; detección de fraude | Captura relaciones no lineales; pierde interpretabilidad y exige más datos; sensible a la escala (mitigado con el `Pipeline`) |
| Proceso Gaussiano | Optimización bayesiana de hiperparámetros; geoestadística (kriging); calibración de sensores | Incertidumbre calibrada con pocos datos; costo O(n³) lo limita a miles de puntos (sección 4.4) |

_(Referencias académicas concretas por caso: pendiente de completar con citas reales en el
anexo de referencias — ver sección "Referencias" más abajo.)_

---

## 7. Simulación interactiva

El sitio (`web/`, publicado a `dist/` por `python -m familia3.publicar`) tiene 5 pestañas:
Naive Bayes, MLP, Proceso Gaussiano, Comparativo y Teoría. Arranca leyendo
`models/registry.json`, descarga cada `.onnx`, verifica su SHA-256 y crea una sesión de ONNX
Runtime Web; si el hash no coincide, el modelo se rechaza. Captura de pantalla y guía de uso:
_(pendiente — agregar tras el despliegue a Netlify)_.

Diseño: paleta de "cuaderno de laboratorio" (papel cuadriculado, tinta añil, acento naranja
óxido reservado para lo interactivo y las bandas de incertidumbre), tipografía Source Serif 4 /
IBM Plex Sans / IBM Plex Mono, modo oscuro automático, responsive. Aviso visible de que es una
herramienta didáctica y no un diagnóstico médico.

---

## 8. Conclusiones y trabajo futuro

_(a completar tras la revisión final del equipo con los resultados consolidados de los 5
notebooks y la matriz de trazabilidad de `docs/PLAN_DE_TRABAJO.md`, sección 8)_

**Conclusiones preliminares respaldadas por los experimentos:**
- El MLP domina en las métricas sensibles a calibración (log-loss) sin perder ROC-AUC frente
  a Naive Bayes, al costo de interpretabilidad y de mayor tiempo de entrenamiento.
- Naive Bayes es defendible cuando se necesita una decisión instantánea y explicable, incluso
  sabiendo que viola su supuesto de independencia en este dataset.
- El Proceso Gaussiano aporta algo que ningún otro modelo de este comparativo da de forma
  nativa: una banda de incertidumbre calibrada, al costo de escalar mal con `n`.

**Trabajo futuro:** un proceso gaussiano con verosimilitud Student-t para robustecer frente a
atípicos (mencionado en la sección 5, no implementado); ampliar el segundo dataset de
clasificación (Wine) con el mismo experimento de atípicos; evaluar un MLP más profundo con
normalización por lotes.

---

## Referencias

- Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006.
- Murphy, K. P. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022.
- Rasmussen, C. E., & Williams, C. K. I. *Gaussian Processes for Machine Learning*. MIT Press, 2006.
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. MIT Press, 2016.
- Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2011.
- Documentación oficial de scikit-learn: https://scikit-learn.org/stable/

## Anexos

- **Anexo A — Derivaciones completas:** ver docstrings de `src/familia3/desde_cero.py` y las
  celdas de verificación matemática de los notebooks 02, 03 y 04.
- **Anexo B — Tablas de hiperparámetros probados:** salida de `busqueda.py` registrada en la
  ejecución de cada notebook (celdas de `GridSearchCV`/`RandomizedSearchCV`).
- **Anexo C — Enlaces:** repositorio (GitHub), sitio (Netlify, pendiente), diapositivas
  (pendiente).
- **Anexo D — Guía de reproducción:** ver sección "Reproducir todo con un solo comando" del
  [`README.md`](../README.md).
- **Anexo E — Matriz de trazabilidad:** [`docs/PLAN_DE_TRABAJO.md`](../docs/PLAN_DE_TRABAJO.md), sección 8.
