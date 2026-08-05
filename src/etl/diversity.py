"""Funciones puras de diversidad diagnóstica.

Implementa:
- `shannon_normalizada(frecuencias)`: entropía de Shannon en [0, 1] (Req 4.1, 4.2, 4.3).
- `comorbilidades_por_egreso(diagnosticos_secundarios)`: cuenta no-nulos en todos los diagnósticos secundarios disponibles por fila (Req 4.4).

Estas funciones son **puras** (sin efectos laterales) y se prueban con PBT en
`tests/property/test_shannon.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shannon_normalizada(frecuencias: pd.Series) -> float:
    """Entropía de Shannon normalizada por log₂(K).

    Mide la diversidad de una distribución de frecuencias. Toma valores en
    [0, 1]:

    - 0  → distribución totalmente concentrada (un solo elemento, K=1).
    - 1  → distribución totalmente uniforme entre los K elementos.

    Convenciones (Req 4.2, 4.3):

    - Serie vacía o todo cero → 0.0.
    - K = 1 (solo un GRD distinto) → 0.0.
    - p_i = 0 → término excluido para evitar `log(0)`.

    Parameters
    ----------
    frecuencias : pd.Series
        Serie con conteos enteros ≥ 0 (típicamente `value_counts()` sobre
        IR_29301_COD_GRD por hospital).

    Returns
    -------
    float
        Entropía normalizada en [0, 1].

    Examples
    --------
    >>> # K=1: cero diversidad
    >>> shannon_normalizada(pd.Series([100]))
    0.0

    >>> # Distribución uniforme: máxima diversidad
    >>> round(shannon_normalizada(pd.Series([10, 10, 10, 10])), 6)
    1.0

    >>> # Concentrada: baja diversidad
    >>> round(shannon_normalizada(pd.Series([95, 5])), 4)
    0.2864
    """
    if frecuencias is None or len(frecuencias) == 0:
        return 0.0
    total = frecuencias.sum()
    if total == 0:
        return 0.0
    p = frecuencias / total
    p = p[p > 0]  # Req 4.3: excluir términos con p=0
    K = len(p)
    if K <= 1:
        return 0.0
    H = -(p * np.log2(p)).sum()
    return float(H / np.log2(K))


def comorbilidades_por_egreso(diagnosticos_secundarios: pd.DataFrame) -> pd.Series:
    """Cuenta el número de diagnósticos secundarios no nulos por egreso.

    Parameters
    ----------
    diagnosticos_secundarios : pd.DataFrame
        DataFrame con las columnas secundarias disponibles, normalmente
        `DIAGNOSTICO2` a `DIAGNOSTICO35`. El número de columnas determina el
        rango máximo del resultado.

    Returns
    -------
    pd.Series
        Serie de enteros en [0, n_columnas]. Con el esquema GRD completo, el
        rango es [0, 34].

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'DIAGNOSTICO2': ['A09', None, 'I21'],
    ...     'DIAGNOSTICO3': ['B05', 'C50', None],
    ...     'DIAGNOSTICO4': [None, None, None],
    ...     'DIAGNOSTICO5': [None, None, None],
    ... })
    >>> comorbilidades_por_egreso(df).tolist()
    [2, 1, 1]
    """
    if diagnosticos_secundarios.shape[0] == 0:
        return pd.Series([], dtype="int64", index=diagnosticos_secundarios.index)
    if diagnosticos_secundarios.shape[1] == 0:
        return pd.Series(0, dtype="int64", index=diagnosticos_secundarios.index)
    # `notna()` cuenta tanto NaN como None como nulos.
    # También consideramos strings vacíos como nulos.
    no_nulos = diagnosticos_secundarios.apply(
        lambda col: col.notna() & (col.astype(str).str.strip() != "")
    )
    return no_nulos.sum(axis=1).astype("int64")
