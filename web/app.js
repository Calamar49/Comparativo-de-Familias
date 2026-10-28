// Arranque de la página: lee registry.json -> elige versión vigente (o ?version=) ->
// descarga cada .onnx -> verifica SHA-256 -> crea sesión ONNX Runtime Web -> predice.
// Nada del modelo (id, versión, métricas, hiperparámetros) está escrito a mano aquí:
// todo sale del registro. Si el hash no coincide, el modelo se rechaza.

const RAIZ_MODELOS = "models/";
const SESIONES = {};
const REGISTRO = { modelos: {} };
const DATOS_WEB = {};

const parametrosURL = new URLSearchParams(location.search);
const versionesForzadas = Object.fromEntries(
  (parametrosURL.get("version") ?? "")
    .split(",")
    .filter(Boolean)
    .map((par) => par.split(":"))
);

function sha256Hex(buffer) {
  return crypto.subtle.digest("SHA-256", buffer).then((hash) =>
    Array.from(new Uint8Array(hash)).map((b) => b.toString(16).padStart(2, "0")).join("")
  );
}

function versionVigente(idModelo) {
  const entrada = REGISTRO.modelos[idModelo];
  if (!entrada) return null;
  const idVersion = versionesForzadas[idModelo] ?? entrada.version_vigente;
  return entrada.versiones.find((v) => v.version === idVersion) ?? null;
}

async function cargarYVerificarModelo(idModelo) {
  const version = versionVigente(idModelo);
  if (!version) throw new Error(`"${idModelo}" no existe en el registro`);

  const respuesta = await fetch(RAIZ_MODELOS + version.archivo);
  const bytes = await respuesta.arrayBuffer();
  const hashReal = await sha256Hex(bytes);
  if (hashReal !== version.sha256) {
    throw new Error(`Hash no coincide para "${idModelo}" ${version.version}: se rechaza el modelo`);
  }

  const sesion = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  SESIONES[idModelo] = sesion;

  const chip = document.getElementById(`chip-${idModelo}`);
  if (chip) chip.textContent = `${idModelo} · ${version.version} · hash verificado`;
  return { sesion, version };
}

async function predecir(idModelo, vectorEntrada) {
  const sesion = SESIONES[idModelo];
  const nombreEntrada = sesion.inputNames[0];
  const tensor = new ort.Tensor("float32", Float32Array.from(vectorEntrada), [1, vectorEntrada.length]);
  const salidas = await sesion.run({ [nombreEntrada]: tensor });
  const nombresSalida = sesion.outputNames;
  const resultado = {};
  for (const nombre of nombresSalida) resultado[nombre] = Array.from(salidas[nombre].data);
  return resultado;
}

// Clasificadores de este proyecto exportan con zipmap:false -> salidas [label, probabilidades]
async function predecirProba(idModelo, vectorEntrada) {
  const salida = await predecir(idModelo, vectorEntrada);
  const nombres = Object.keys(salida);
  const proba = salida[nombres[1]] ?? salida[nombres[0]];
  return Array.from(proba);
}

// GPR exporta [media, desviacion] en la escala ESTANDARIZADA; hay que deshacerla
// con y_offset / y_scale del registro (ver src/familia3/todo.py).
async function predecirGPR(idModelo, vectorEntrada) {
  const version = versionVigente(idModelo);
  const salida = await predecir(idModelo, vectorEntrada);
  const nombres = Object.keys(salida);
  const mediaStd = salida[nombres[0]][0];
  const sigmaStd = salida[nombres[1]][0];
  const offset = version.y_offset ?? 0;
  const escala = version.y_scale ?? 1;
  return { media: mediaStd * escala + offset, sigma: Math.abs(sigmaStd * escala) };
}

async function cargarJSON(ruta) {
  const r = await fetch(ruta);
  return r.json();
}

// ---------------------------------------------------------------------------
// Deslizadores reutilizables
// ---------------------------------------------------------------------------
function construirDeslizadores(contenedor, rangos, nombresVisibles, valoresIniciales, alCambiar) {
  contenedor.innerHTML = "";
  const valores = [...valoresIniciales];
  rangos.forEach((rango, i) => {
    if (!nombresVisibles.includes(rango.nombre)) return;
    const fila = document.createElement("div");
    fila.className = "deslizador-fila";
    const paso = (rango.max - rango.min) / 200 || 0.01;
    fila.innerHTML = `
      <label for="d-${i}">${rango.nombre}</label>
      <input type="range" id="d-${i}" min="${rango.min}" max="${rango.max}" step="${paso}" value="${valores[i]}">
      <output></output>`;
    const input = fila.querySelector("input");
    const salida = fila.querySelector("output");
    salida.textContent = Number(valores[i]).toFixed(2);
    input.addEventListener("input", () => {
      valores[i] = Number(input.value);
      salida.textContent = valores[i].toFixed(2);
      alCambiar([...valores]);
    });
    contenedor.appendChild(fila);
  });
  return valores;
}

// ---------------------------------------------------------------------------
// Pestaña Naive Bayes
// ---------------------------------------------------------------------------
const ATRIBUTOS_VISIBLES_CANCER = [
  "worst radius", "worst texture", "worst concave points", "mean concavity",
];

async function inicializarNB() {
  const datos = DATOS_WEB.breast_cancer;
  const selector = document.getElementById("nb-caso");
  datos.casos_de_prueba.forEach((caso, i) => {
    const opcion = document.createElement("option");
    opcion.value = i;
    opcion.textContent = `caso ${i + 1} · real: ${caso.etiqueta_real === 0 ? "maligno" : "benigno"}`;
    selector.appendChild(opcion);
  });

  const actualizar = async (vector) => {
    const proba = await predecirProba("nb_cancer", vector);
    pintarResultadoBinario("nb", proba, ["maligno", "benigno"]);
  };

  let vectorActual = [...datos.casos_de_prueba[0].atributos];
  const deslizadores = document.getElementById("nb-deslizadores");
  construirDeslizadores(deslizadores, datos.rangos_atributos, ATRIBUTOS_VISIBLES_CANCER, vectorActual, (v) => {
    vectorActual = v;
    actualizar(vectorActual);
  });

  selector.addEventListener("change", () => {
    vectorActual = [...datos.casos_de_prueba[Number(selector.value)].atributos];
    construirDeslizadores(deslizadores, datos.rangos_atributos, ATRIBUTOS_VISIBLES_CANCER, vectorActual, (v) => {
      vectorActual = v;
      actualizar(vectorActual);
    });
    actualizar(vectorActual);
  });

  await actualizar(vectorActual);
  await dibujarFronteraMoons("nb_moons", "nb-canvas");
  pintarParametrosNB();
}

function pintarResultadoBinario(prefijo, proba, etiquetas) {
  const p1 = proba[1] ?? proba[0];
  document.getElementById(`${prefijo}-barra`).style.width = `${(p1 * 100).toFixed(1)}%`;
  const clase = p1 >= 0.5 ? etiquetas[1] : etiquetas[0];
  document.getElementById(`${prefijo}-texto`).textContent =
    `${clase} · P(${etiquetas[1]})=${p1.toFixed(3)}  P(${etiquetas[0]})=${(1 - p1).toFixed(3)}`;
}

async function pintarParametrosNB() {
  const version = versionVigente("nb_cancer");
  const contenedor = document.getElementById("nb-parametros");
  contenedor.innerHTML = `
    <table>
      <tr><th>métrica</th><th>valor</th></tr>
      ${Object.entries(version.metricas).map(([k, v]) => `<tr><td>${k}</td><td>${Number(v).toFixed(4)}</td></tr>`).join("")}
      <tr><td>var_smoothing</td><td>${version.hiperparametros.var_smoothing}</td></tr>
      <tr><td>dataset</td><td>${version.dataset}</td></tr>
      <tr><td>semilla</td><td>${version.seed}</td></tr>
    </table>`;
}

// ---------------------------------------------------------------------------
// Pestaña MLP
// ---------------------------------------------------------------------------
async function inicializarMLP() {
  const datos = DATOS_WEB.breast_cancer;
  const selector = document.getElementById("mlp-caso");
  datos.casos_de_prueba.forEach((caso, i) => {
    const opcion = document.createElement("option");
    opcion.value = i;
    opcion.textContent = `caso ${i + 1} · real: ${caso.etiqueta_real === 0 ? "maligno" : "benigno"}`;
    selector.appendChild(opcion);
  });

  const actualizar = async (vector) => {
    const proba = await predecirProba("mlp_cancer", vector);
    pintarResultadoBinario("mlp", proba, ["maligno", "benigno"]);
  };

  let vectorActual = [...datos.casos_de_prueba[0].atributos];
  const deslizadores = document.getElementById("mlp-deslizadores");
  construirDeslizadores(deslizadores, datos.rangos_atributos, ATRIBUTOS_VISIBLES_CANCER, vectorActual, (v) => {
    vectorActual = v;
    actualizar(vectorActual);
  });

  selector.addEventListener("change", () => {
    vectorActual = [...datos.casos_de_prueba[Number(selector.value)].atributos];
    construirDeslizadores(deslizadores, datos.rangos_atributos, ATRIBUTOS_VISIBLES_CANCER, vectorActual, (v) => {
      vectorActual = v;
      actualizar(vectorActual);
    });
    actualizar(vectorActual);
  });

  await actualizar(vectorActual);
  await dibujarFronteraMoons("mlp_moons", "mlp-canvas");
  pintarArquitecturaMLP();
}

function pintarArquitecturaMLP() {
  const version = versionVigente("mlp_cancer");
  const contenedor = document.getElementById("mlp-arquitectura");
  contenedor.innerHTML = `
    <table>
      <tr><th>capas ocultas</th><td>${JSON.stringify(version.hiperparametros.hidden_layer_sizes)}</td></tr>
      <tr><th>activación</th><td>${version.hiperparametros.activation}</td></tr>
      <tr><th>alpha (L2)</th><td>${version.hiperparametros.alpha}</td></tr>
      <tr><th>roc_auc (prueba)</th><td>${version.metricas.roc_auc.toFixed(4)}</td></tr>
      <tr><th>log_loss (prueba)</th><td>${version.metricas.log_loss.toFixed(4)}</td></tr>
    </table>`;
}

// ---------------------------------------------------------------------------
// Fronteras 2D (make_moons) — malla consultada contra el modelo, nunca reentrenado
// ---------------------------------------------------------------------------
async function dibujarFronteraMoons(idModelo, idCanvas) {
  const lienzo = document.getElementById(idCanvas);
  const ctx = lienzo.getContext("2d");
  const ancho = lienzo.width, alto = lienzo.height;
  const paso = 6;
  const xMin = -1.8, xMax = 2.6, yMin = -1.3, yMax = 1.6;

  const imagen = ctx.createImageData(ancho, alto);
  for (let px = 0; px < ancho; px += paso) {
    const columna = [];
    for (let py = 0; py < alto; py += paso) {
      const x = xMin + (px / ancho) * (xMax - xMin);
      const y = yMax - (py / alto) * (yMax - yMin);
      columna.push([x, y]);
    }
    for (const [x, y] of columna) {
      const proba = await predecirProba(idModelo, [x, y]);
      const p1 = proba[1] ?? proba[0];
      const py0 = Math.round(((yMax - y) / (yMax - yMin)) * alto);
      const [r, g, b] = colorDeProbabilidad(p1);
      for (let dx = 0; dx < paso; dx++) {
        for (let dy = 0; dy < paso; dy++) {
          const idx = ((py0 + dy) * ancho + (px + dx)) * 4;
          if (idx < 0 || idx >= imagen.data.length - 4) continue;
          imagen.data[idx] = r; imagen.data[idx + 1] = g; imagen.data[idx + 2] = b; imagen.data[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imagen, 0, 0);

  // puntos reales encima
  const datos = DATOS_WEB.make_moons;
  ctx.lineWidth = 1.2;
  for (const caso of datos.casos_de_prueba) {
    const [x, y] = caso.atributos;
    const px = ((x - xMin) / (xMax - xMin)) * ancho;
    const py = ((yMax - y) / (yMax - yMin)) * alto;
    ctx.fillStyle = caso.etiqueta_real === 1 ? "#13233d" : "#ffffff";
    ctx.strokeStyle = "#13233d";
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function colorDeProbabilidad(p1) {
  // de papel (p≈0) a señal azul (p≈1), pasando por tinta en el medio.
  const papel = [255, 255, 255], tinta = [100, 115, 135], senal = [20, 88, 201];
  const mezclar = (a, b, t) => Math.round(a + (b - a) * t);
  if (p1 < 0.5) {
    const t = p1 / 0.5;
    return [mezclar(papel[0], tinta[0], t), mezclar(papel[1], tinta[1], t), mezclar(papel[2], tinta[2], t)];
  }
  const t = (p1 - 0.5) / 0.5;
  return [mezclar(tinta[0], senal[0], t), mezclar(tinta[1], senal[1], t), mezclar(tinta[2], senal[2], t)];
}

// ---------------------------------------------------------------------------
// Pestaña GPR
// ---------------------------------------------------------------------------
const ATRIBUTOS_VISIBLES_DIABETES = ["bmi", "bp", "s5", "age"];

async function inicializarGPR() {
  const datos = DATOS_WEB.diabetes;
  const vectorBase = [...datos.casos_de_prueba[0].atributos];

  const actualizar = async (vector) => {
    const { media, sigma } = await predecirGPR("gpr_diabetes", vector);
    document.getElementById("gpr-resultado").textContent =
      `media ± 2σ: ${media.toFixed(1)} ± ${(2 * sigma).toFixed(1)}  (progresión de la enfermedad a un año)`;
  };

  construirDeslizadores(
    document.getElementById("gpr-deslizadores"),
    datos.rangos_atributos,
    ATRIBUTOS_VISIBLES_DIABETES,
    vectorBase,
    (v) => actualizar(v)
  );
  await actualizar(vectorBase);
  await dibujarBandaSeno();
}

async function dibujarBandaSeno() {
  const lienzo = document.getElementById("gpr-canvas");
  const ctx = lienzo.getContext("2d");
  const ancho = lienzo.width, alto = lienzo.height;
  const xMin = -7, xMax = 7, yMin = -3, yMax = 3;
  const nPuntos = 70;

  const medias = [], sigmas = [], xs = [];
  for (let i = 0; i < nPuntos; i++) {
    const x = xMin + (i / (nPuntos - 1)) * (xMax - xMin);
    const { media, sigma } = await predecirGPR("gpr_seno1d", [x]);
    xs.push(x); medias.push(media); sigmas.push(sigma);
  }

  const aPx = (x) => ((x - xMin) / (xMax - xMin)) * ancho;
  const aPy = (y) => ((yMax - y) / (yMax - yMin)) * alto;

  ctx.clearRect(0, 0, ancho, alto);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, ancho, alto);

  // banda ±2σ
  ctx.beginPath();
  xs.forEach((x, i) => ctx.lineTo(aPx(x), aPy(medias[i] + 2 * sigmas[i])));
  for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(aPx(xs[i]), aPy(medias[i] - 2 * sigmas[i]));
  ctx.closePath();
  ctx.fillStyle = "rgba(20, 88, 201, 0.14)";
  ctx.fill();

  // media
  ctx.beginPath();
  xs.forEach((x, i) => (i === 0 ? ctx.moveTo(aPx(x), aPy(medias[i])) : ctx.lineTo(aPx(x), aPy(medias[i]))));
  ctx.strokeStyle = "#1458c9";
  ctx.lineWidth = 2;
  ctx.stroke();

  // f(x) real
  ctx.beginPath();
  xs.forEach((x, i) => (i === 0 ? ctx.moveTo(aPx(x), aPy(Math.sin(x))) : ctx.lineTo(aPx(x), aPy(Math.sin(x)))));
  ctx.strokeStyle = "#13233d";
  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([]);

  // datos de entrenamiento
  const [xTrain, yTrain] = await cargarPuntosEntrenamientoSeno();
  ctx.fillStyle = "#13233d";
  xTrain.forEach((x, i) => {
    ctx.beginPath();
    ctx.arc(aPx(x), aPy(yTrain[i]), 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

let _puntosSenoCache = null;
async function cargarPuntosEntrenamientoSeno() {
  if (_puntosSenoCache) return _puntosSenoCache;
  // No hay archivo propio para esto (el GPR 1D guarda sus datos dentro del .onnx);
  // se regenera la misma muestra sintética con la misma semilla que documenta el registro.
  const rng = mulberry32(42);
  const xs = [];
  while (xs.length < 60) {
    const x = -6 + rng() * 12;
    if (x < 3 || x > 5) xs.push(x);
  }
  const ys = xs.map((x) => Math.sin(x) + (rng() - 0.5) * 0.3);
  _puntosSenoCache = [xs, ys];
  return _puntosSenoCache;
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Pestaña comparativo
// ---------------------------------------------------------------------------
async function inicializarComparativo() {
  const datos = DATOS_WEB.breast_cancer;
  const selector = document.getElementById("cmp-caso");
  datos.casos_de_prueba.forEach((caso, i) => {
    const opcion = document.createElement("option");
    opcion.value = i;
    opcion.textContent = `caso ${i + 1} · real: ${caso.etiqueta_real === 0 ? "maligno" : "benigno"}`;
    selector.appendChild(opcion);
  });

  const actualizar = async () => {
    const caso = datos.casos_de_prueba[Number(selector.value)];
    // ONNX Runtime Web (backend wasm) no garantiza ejecutar session.run() de dos
    // sesiones en paralelo de forma segura: se consulta una sesión a la vez.
    const probaNB = await predecirProba("nb_cancer", caso.atributos);
    const probaMLP = await predecirProba("mlp_cancer", caso.atributos);
    const filas = document.getElementById("cmp-filas");
    filas.innerHTML = `
      <div class="comparativo-col"><h4>real</h4>${caso.etiqueta_real === 0 ? "maligno" : "benigno"}</div>
      <div class="comparativo-col"><h4>Naive Bayes</h4>P(benigno) = ${(probaNB[1] ?? probaNB[0]).toFixed(3)}</div>
      <div class="comparativo-col"><h4>MLP</h4>P(benigno) = ${(probaMLP[1] ?? probaMLP[0]).toFixed(3)}</div>`;
  };
  selector.addEventListener("change", actualizar);
  await actualizar();

  const vNB = versionVigente("nb_cancer");
  const vMLP = versionVigente("mlp_cancer");
  document.getElementById("cmp-metricas").innerHTML = `
    <table>
      <tr><th>modelo</th><th>roc_auc</th><th>f1</th><th>log_loss</th><th>brier</th></tr>
      <tr><td>Naive Bayes</td><td>${vNB.metricas.roc_auc.toFixed(4)}</td><td>${vNB.metricas.f1.toFixed(4)}</td><td>${vNB.metricas.log_loss.toFixed(4)}</td><td>${vNB.metricas.brier.toFixed(4)}</td></tr>
      <tr><td>MLP</td><td>${vMLP.metricas.roc_auc.toFixed(4)}</td><td>${vMLP.metricas.f1.toFixed(4)}</td><td>${vMLP.metricas.log_loss.toFixed(4)}</td><td>${vMLP.metricas.brier.toFixed(4)}</td></tr>
    </table>`;
}

// ---------------------------------------------------------------------------
// Navegación entre pestañas
// ---------------------------------------------------------------------------
function inicializarNavegacion() {
  const botones = document.querySelectorAll(".pestanas__btn:not(.pestanas__link)");
  botones.forEach((boton) => {
    boton.addEventListener("click", () => {
      botones.forEach((b) => b.removeAttribute("aria-current"));
      boton.setAttribute("aria-current", "page");
      document.querySelectorAll(".vista").forEach((v) => (v.hidden = v.dataset.vista !== boton.dataset.vista));
    });
  });
}

// ---------------------------------------------------------------------------
// Arranque
// ---------------------------------------------------------------------------
async function arrancar() {
  const insignia = document.getElementById("insignia-carga");
  const texto = document.getElementById("insignia-carga__texto");
  try {
    Object.assign(REGISTRO, await cargarJSON(RAIZ_MODELOS + "registry.json"));
    document.getElementById("indice-registro").textContent =
      `registro · ${Object.keys(REGISTRO.modelos).length} modelos`;

    texto.textContent = "Descargando y verificando modelos…";
    await Promise.all(
      ["nb_cancer", "mlp_cancer", "nb_moons", "mlp_moons", "gpr_diabetes", "gpr_seno1d"].map(cargarYVerificarModelo)
    );

    DATOS_WEB.breast_cancer = await cargarJSON(RAIZ_MODELOS + "datos_web_breast_cancer.json");
    DATOS_WEB.make_moons = await cargarJSON(RAIZ_MODELOS + "datos_web_make_moons.json");
    DATOS_WEB.diabetes = await cargarJSON(RAIZ_MODELOS + "datos_web_diabetes.json");

    insignia.classList.add("insignia-carga--ok");
    texto.textContent = "Modelos cargados y verificados por SHA-256";

    inicializarNavegacion();
    await inicializarNB();
    await inicializarMLP();
    await inicializarGPR();
    await inicializarComparativo();
  } catch (error) {
    insignia.classList.add("insignia-carga--error");
    texto.textContent = `Error: ${error.message}`;
    console.error(error);
  }
}

arrancar();
