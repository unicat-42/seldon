"""Объяснение прогноза логит-модели для веб-витрины: вклады признаков и интервал оценки.

Все четыре логит-модели витрины устроены одинаково: `StandardScaler` → `LogisticRegression`.
Для такой модели лог-шансы (логит вероятности) — это сумма

    η_i = b0 + Σ_j β_j · z_ij,      z_ij = (x_ij − среднее_j) / σ_j,      p_i = 1 / (1 + e^−η_i),

поэтому вклад признака j в прогноз i — точное число `β_j · z_ij` В ЛОГ-ШАНСАХ. Это не
процентные пункты вероятности: вклады складываются на шкале логита, а не на шкале p. Ноль
вклада — значение признака равно среднему по обучающей выборке.

Интервал оценки (95 %). Неопределённость коэффициентов переносится на прогноз: дисперсия
η_i = d_i′ V d_i, где d_i = (1, z_i), а V — «сэндвич»-оценка ковариации коэффициентов
H⁻¹ M H⁻¹: H — гессиан штрафованного правдоподобия (с учётом регуляризации `C`), M — сумма
внешних произведений вкладов кластеров в градиент. Наблюдения панели не независимы (одна
страна много лет подряд; один год — общий мировой шок), поэтому M считается дважды — с
кластерами по странам и по периодам, — и для каждого прогноза берётся БОЛЬШАЯ из двух
дисперсий. Границы — сигмоида от η ± 1,96·se.

Чего интервал НЕ учитывает (и это написано на сайте): ошибку спецификации модели, выбор
силы регуляризации, оценку средних и σ в стандартизации, порогов обрезки и медиан для
пропусков. Это интервал «при условии, что модель верна», а не интервал предсказания события.
Для деревьев (`ExtraTrees`) такого разложения и такой ковариации нет — там ни вкладов, ни
интервала не публикуем.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class Explained:
    """Разбор прогнозов одной логит-модели на строках теста."""
    features: list[str]
    intercept: float
    beta: np.ndarray      # (k,) коэффициенты при стандартизованных признаках
    z: np.ndarray         # (n, k) стандартизованные значения
    contrib: np.ndarray   # (n, k) вклады β_j · z_ij, лог-шансы
    eta: np.ndarray       # (n,) лог-шансы
    p: np.ndarray         # (n,) вероятность
    lo: np.ndarray        # (n,) нижняя граница 95 % интервала оценки
    hi: np.ndarray        # (n,) верхняя
    se: np.ndarray        # (n,) стандартная ошибка η
    n_train: int
    base_rate: float      # доля событий в обучающей выборке


def _cluster_meat(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Σ_g s_g s_g′, s_g — сумма вкладов наблюдений кластера в градиент; поправка G / (G − 1)."""
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    G = int(inv.max()) + 1
    s = np.zeros((G, scores.shape[1]))
    np.add.at(s, inv, scores)
    return (s.T @ s) * (G / (G - 1) if G > 1 else 1.0)


def _standardize(scaler, X) -> np.ndarray:
    """То же, что `scaler.transform`, но для голой матрицы (без имён колонок)."""
    return (np.asarray(X, dtype=float) - scaler.mean_) / scaler.scale_


def coef_covariances(pipe, X_train, y_train, clusters: list) -> list[np.ndarray]:
    """Кластерные ковариации вектора (b0, β) — по одной на каждое разбиение на кластеры."""
    scaler, lr = pipe[0], pipe[-1]
    Z = _standardize(scaler, X_train)
    D = np.column_stack([np.ones(len(Z)), Z])
    theta = np.concatenate([lr.intercept_, lr.coef_[0]])
    p = sigmoid(D @ theta)
    w = p * (1.0 - p)
    C = float(lr.C)
    ridge = np.zeros(D.shape[1])
    if np.isfinite(C):
        ridge[1:] = 1.0 / C  # sklearn: C·Σпотери + ½‖β‖², свободный член не штрафуется
    H = D.T @ (D * w[:, None]) + np.diag(ridge)
    Hinv = np.linalg.pinv(H)
    scores = D * (np.asarray(y_train, dtype=float) - p)[:, None]
    return [Hinv @ _cluster_meat(scores, g) @ Hinv for g in clusters]


def explain_logit(pipe, features: list[str], X_train, y_train, X_test, clusters: list) -> Explained:
    """Вклады признаков и интервал оценки для `Pipeline(StandardScaler, LogisticRegression)`.
    `X_train`, `X_test` — матрицы признаков в том виде, в каком их видит модель (после обрезки
    и заполнения пропусков); `clusters` — список массивов меток кластеров строк обучения."""
    scaler, lr = pipe[0], pipe[-1]
    Xt = np.asarray(X_test, dtype=float)
    z = _standardize(scaler, Xt) if len(Xt) else np.zeros((0, len(features)))
    beta, b0 = lr.coef_[0].astype(float), float(lr.intercept_[0])
    contrib = z * beta
    eta = b0 + contrib.sum(axis=1)
    d = np.column_stack([np.ones(len(z)), z])
    var = np.zeros(len(z))
    for V in coef_covariances(pipe, X_train, y_train, clusters):
        var = np.maximum(var, np.einsum("ij,jk,ik->i", d, V, d))
    se = np.sqrt(np.clip(var, 0.0, None))
    y = np.asarray(y_train, dtype=float)
    return Explained(list(features), b0, beta, z, contrib, eta, sigmoid(eta), sigmoid(eta - Z95 * se),
                     sigmoid(eta + Z95 * se), se, len(y), float(y.mean()))
