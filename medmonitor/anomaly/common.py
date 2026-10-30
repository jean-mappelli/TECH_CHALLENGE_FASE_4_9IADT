"""Utilitários compartilhados pelos detectores de anomalia."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def rolling_robust_z(series: pd.Series, window: int = 30, min_periods: int = 10) -> pd.Series:
    """Z-score robusto (mediana/MAD) contra uma janela *anterior* ao ponto atual.

    A linha de base exclui o ponto avaliado (shift(1)) para que um valor
    anômalo não "contamine" a própria referência.
    """
    s = series.astype(float)
    med = s.rolling(window, min_periods=min_periods).median().shift(1)
    mad = (
        s.rolling(window, min_periods=min_periods)
        .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        .shift(1)
    )
    scale = np.maximum(mad * MAD_TO_SIGMA, 0.02 * np.abs(med) + 1e-6)
    return (s - med) / scale


def global_robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * MAD_TO_SIGMA
    scale = max(mad, 0.02 * abs(med) + 1e-6)
    return (values - med) / scale


def group_events(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """Agrupa índices True em eventos (início, fim) inclusivos.

    `max_gap` = quantas amostras não marcadas podem existir entre duas marcadas
    sem separar o evento (0 = só índices consecutivos).
    """
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    events: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > max_gap:
            events.append((start, prev))
            start = i
        prev = i
    events.append((start, prev))
    return events
