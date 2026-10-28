"""Implementaciones en NumPy puro de los 3 modelos, para verificar matemáticamente
que entendemos la derivación y que coincide numéricamente con scikit-learn."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# A1. Naive Bayes Gaussiano: solución cerrada por máxima verosimilitud
# ---------------------------------------------------------------------------
class NaiveBayesGaussianoNumPy:
    """ŷ = argmax_k log P(y=k) + Σ_j log N(x_j; μ_kj, σ²_kj)."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesGaussianoNumPy":
        self.clases_ = np.unique(y)
        n_atributos = X.shape[1]
        self.medias_ = np.zeros((len(self.clases_), n_atributos))
        self.varianzas_ = np.zeros((len(self.clases_), n_atributos))
        self.log_priors_ = np.zeros(len(self.clases_))
        epsilon = self.var_smoothing * X.var(axis=0).max()
        for i, clase in enumerate(self.clases_):
            X_clase = X[y == clase]
            self.medias_[i] = X_clase.mean(axis=0)
            self.varianzas_[i] = X_clase.var(axis=0) + epsilon
            self.log_priors_[i] = np.log(len(X_clase) / len(X))
        return self

    def _log_verosimilitud_conjunta(self, X: np.ndarray) -> np.ndarray:
        salida = np.zeros((X.shape[0], len(self.clases_)))
        for i in range(len(self.clases_)):
            termino = -0.5 * np.sum(np.log(2 * np.pi * self.varianzas_[i]))
            termino -= 0.5 * np.sum(((X - self.medias_[i]) ** 2) / self.varianzas_[i], axis=1)
            salida[:, i] = termino + self.log_priors_[i]
        return salida

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        log_conjunta = self._log_verosimilitud_conjunta(X)
        log_normalizador = np.logaddexp.reduce(log_conjunta, axis=1, keepdims=True)
        return log_conjunta - log_normalizador

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_conjunta = self._log_verosimilitud_conjunta(X)
        return self.clases_[np.argmax(log_conjunta, axis=1)]


# ---------------------------------------------------------------------------
# A2. MLP: forward + backpropagation deducida con la regla de la cadena
# ---------------------------------------------------------------------------
def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


class MLPNumPy:
    """Una capa oculta, activación sigmoide, salida softmax, entropía cruzada + L2.

    Forward:  a1 = σ(X W1 + b1);  z2 = a1 W2 + b2;  ŷ = softmax(z2)
    Pérdida:  L = -1/n Σ log ŷ_{i, y_i} + (α/2n)(||W1||² + ||W2||²)
    Backprop (regla de la cadena, capa por capa):
        dz2 = ŷ - Y_onehot
        dW2 = a1ᵀ dz2 / n + (α/n) W2 ;  db2 = mean(dz2)
        da1 = dz2 W2ᵀ
        dz1 = da1 ⊙ a1 ⊙ (1 - a1)          (derivada de la sigmoide)
        dW1 = Xᵀ dz1 / n + (α/n) W1 ;      db1 = mean(dz1)
    """

    def __init__(self, n_entradas: int, n_ocultas: int, n_clases: int, alpha: float = 1e-4, seed: int = 42):
        rng = np.random.default_rng(seed)
        escala1 = np.sqrt(2.0 / n_entradas)
        escala2 = np.sqrt(2.0 / n_ocultas)
        self.W1 = rng.normal(0, escala1, size=(n_entradas, n_ocultas))
        self.b1 = np.zeros(n_ocultas)
        self.W2 = rng.normal(0, escala2, size=(n_ocultas, n_clases))
        self.b2 = np.zeros(n_clases)
        self.alpha = alpha
        self.n_clases = n_clases

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _sigmoide(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = _softmax(z2)
        return a1, y_hat

    def predict_proba(self, X):
        _, y_hat = self._forward(X)
        return y_hat

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _perdida_y_gradientes(self, X, y_onehot):
        n = X.shape[0]
        a1, y_hat = self._forward(X)
        perdida = -np.sum(y_onehot * np.log(np.clip(y_hat, 1e-12, 1))) / n
        perdida += (self.alpha / (2 * n)) * (np.sum(self.W1**2) + np.sum(self.W2**2))

        dz2 = (y_hat - y_onehot) / n
        dW2 = a1.T @ dz2 + (self.alpha / n) * self.W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * a1 * (1 - a1)
        dW1 = X.T @ dz1 + (self.alpha / n) * self.W1
        db1 = dz1.sum(axis=0)
        return perdida, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def fit(self, X, y, epocas: int = 500, tasa_aprendizaje: float = 0.5):
        y_onehot = np.eye(self.n_clases)[y]
        historial = []
        for _ in range(epocas):
            perdida, grad = self._perdida_y_gradientes(X, y_onehot)
            self.W1 -= tasa_aprendizaje * grad["W1"]
            self.b1 -= tasa_aprendizaje * grad["b1"]
            self.W2 -= tasa_aprendizaje * grad["W2"]
            self.b2 -= tasa_aprendizaje * grad["b2"]
            historial.append(perdida)
        self.historial_perdida_ = historial
        return self

    def parametros_planos(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def cargar_parametros_planos(self, plano):
        n1 = self.W1.size
        n2 = n1 + self.b1.size
        n3 = n2 + self.W2.size
        n4 = n3 + self.b2.size
        self.W1 = plano[:n1].reshape(self.W1.shape)
        self.b1 = plano[n1:n2].reshape(self.b1.shape)
        self.W2 = plano[n2:n3].reshape(self.W2.shape)
        self.b2 = plano[n3:n4].reshape(self.b2.shape)


def comprobar_gradientes(modelo: MLPNumPy, X, y, epsilon: float = 1e-5) -> float:
    """Diferencias finitas centradas: compara el gradiente analítico (backprop) con
    uno numérico. Devuelve el error relativo máximo; debe ser muy pequeño (< 1e-4)."""
    y_onehot = np.eye(modelo.n_clases)[y]
    _, grad_analitico = modelo._perdida_y_gradientes(X, y_onehot)
    grad_analitico_plano = np.concatenate(
        [grad_analitico["W1"].ravel(), grad_analitico["b1"].ravel(), grad_analitico["W2"].ravel(), grad_analitico["b2"].ravel()]
    )

    theta0 = modelo.parametros_planos()
    grad_numerico = np.zeros_like(theta0)
    for i in range(len(theta0)):
        theta_mas = theta0.copy()
        theta_mas[i] += epsilon
        modelo.cargar_parametros_planos(theta_mas)
        perdida_mas, _ = modelo._perdida_y_gradientes(X, y_onehot)

        theta_menos = theta0.copy()
        theta_menos[i] -= epsilon
        modelo.cargar_parametros_planos(theta_menos)
        perdida_menos, _ = modelo._perdida_y_gradientes(X, y_onehot)

        grad_numerico[i] = (perdida_mas - perdida_menos) / (2 * epsilon)

    modelo.cargar_parametros_planos(theta0)
    diferencia = np.linalg.norm(grad_analitico_plano - grad_numerico)
    norma = np.linalg.norm(grad_analitico_plano) + np.linalg.norm(grad_numerico)
    return float(diferencia / norma) if norma > 0 else 0.0


# ---------------------------------------------------------------------------
# A3. Proceso Gaussiano: factorización de Cholesky, igual que Rasmussen & Williams
# ---------------------------------------------------------------------------
def kernel_rbf_numpy(X1: np.ndarray, X2: np.ndarray, longitud_escala: float = 1.0, varianza: float = 1.0) -> np.ndarray:
    dist2 = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return varianza * np.exp(-0.5 * dist2 / longitud_escala**2)


class ProcesoGaussianoNumPy:
    """GP con núcleo RBF fijo (sin optimizar hiperparámetros, eso ya lo hace sklearn):
    μ* = K*ᵀ (K + σ²I)⁻¹ y ;  σ*² = k** − K*ᵀ (K + σ²I)⁻¹ K*
    Se usa la factorización de Cholesky de (K + σ²I) por estabilidad numérica.

    normalize_y replica exactamente a sklearn: se centra y se escala a varianza
    unitaria antes de ajustar, y la predicción se re-escala de vuelta (media y
    varianza), para poder compararse numéricamente con GaussianProcessRegressor.
    """

    def __init__(self, longitud_escala: float = 1.0, varianza: float = 1.0, ruido: float = 1e-6, normalize_y: bool = False):
        self.longitud_escala = longitud_escala
        self.varianza = varianza
        self.ruido = ruido
        self.normalize_y = normalize_y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProcesoGaussianoNumPy":
        self.X_train_ = X
        if self.normalize_y:
            self.media_y_ = y.mean()
            self.std_y_ = y.std() if y.std() > 0 else 1.0
        else:
            self.media_y_ = 0.0
            self.std_y_ = 1.0
        self.y_centrado_ = (y - self.media_y_) / self.std_y_
        K = kernel_rbf_numpy(X, X, self.longitud_escala, self.varianza)
        K += self.ruido * np.eye(len(X))
        self.L_ = np.linalg.cholesky(K)
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, self.y_centrado_))
        return self

    def log_verosimilitud_marginal(self) -> float:
        """-½ yᵀK⁻¹y - ½ log|K| - (n/2) log(2π), los tres términos: ajuste, complejidad, constante."""
        n = len(self.y_centrado_)
        ajuste = -0.5 * self.y_centrado_ @ self.alpha_
        complejidad = -np.sum(np.log(np.diagonal(self.L_)))
        constante = -0.5 * n * np.log(2 * np.pi)
        return float(ajuste + complejidad + constante)

    def predict(self, X_nuevo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K_estrella = kernel_rbf_numpy(self.X_train_, X_nuevo, self.longitud_escala, self.varianza)
        media = (K_estrella.T @ self.alpha_) * self.std_y_ + self.media_y_
        v = np.linalg.solve(self.L_, K_estrella)
        k_doble_estrella = self.varianza
        varianza_pred = k_doble_estrella - np.sum(v**2, axis=0)
        varianza_pred = np.clip(varianza_pred, 1e-12, None) * self.std_y_**2
        return media, np.sqrt(varianza_pred)
