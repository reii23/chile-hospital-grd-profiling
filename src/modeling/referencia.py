"""Valores de referencia del pipeline definitivo, calculados y no fijados a mano.

Los análisis de sensibilidad comparan sus resultados contra la solución
definitiva (Ward K=4 sobre la Hospital_Matrix_Integrada). Fijar esa cifra como
literal en cada script la deja obsoleta en cuanto el pipeline se recalcula, de
modo que aquí se deriva siempre desde la matriz persistida.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from src.utils.io import PROCESSED_DIR

LOG1P_COLS = ("egresos_por_anio", "pabellones_promedio")
COLS_DROP_CLUSTERING = ("COD_HOSPITAL", "peso_medio_cma", "peso_medio_cma_imputado")
K_REFERENCIA = 4


def preprocesar_matriz(matriz: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Aplica log1p y escalado robusto igual que el pipeline definitivo."""
    matriz = matriz.copy()
    matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
    matriz = matriz.sort_values("COD_HOSPITAL").reset_index(drop=True)
    feats = matriz.drop(
        columns=[c for c in COLS_DROP_CLUSTERING if c in matriz.columns]
    ).fillna(0.0)
    for col in LOG1P_COLS:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    return RobustScaler().fit_transform(feats), matriz["COD_HOSPITAL"].tolist()


def particion_referencia(
    path: Path | None = None, k: int = K_REFERENCIA
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Devuelve (matriz escalada, etiquetas Ward, códigos de hospital)."""
    ruta = path or (PROCESSED_DIR / "hospital_matrix.parquet")
    X, ids = preprocesar_matriz(pd.read_parquet(ruta))
    labels = fcluster(linkage(X, method="ward"), t=k, criterion="maxclust") - 1
    return X, labels, ids


def silhouette_referencia(path: Path | None = None, k: int = K_REFERENCIA) -> float:
    """Coeficiente de Silhouette de la solución definitiva Ward K=k."""
    X, labels, _ = particion_referencia(path, k)
    return float(silhouette_score(X, labels))
