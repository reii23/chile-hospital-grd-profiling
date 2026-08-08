"""Sensibilidad R-05 al retirar el indicador técnico ``tasa_partos``.

Compara Ward K=4 sobre la matriz canónica con la misma matriz sin la columna
``tasa_partos``. El análisis no modifica el pipeline: evalúa si la redundancia
con indicadores obstétricos cambia materialmente la partición.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, preprocesar_matriz
from src.utils.io import PROCESSED_DIR, TABLES_DIR

K_WARD = 4
INDICADOR = "tasa_partos"


def etiquetas_ward(X):
    return fcluster(linkage(X, method="ward"), t=K_WARD, criterion="maxclust") - 1


def metricas(nombre: str, X, labels, referencia) -> dict[str, object]:
    return {
        "escenario": nombre,
        "n_variables": X.shape[1],
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "tamanos": str(pd.Series(labels).value_counts().sort_index().tolist()),
        "ARI_vs_canonica": float(adjusted_rand_score(referencia, labels)),
        "NMI_vs_canonica": float(normalized_mutual_info_score(referencia, labels)),
    }


def main() -> None:
    matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
    if INDICADOR not in matriz.columns:
        raise ValueError(f"La matriz no contiene {INDICADOR!r}")

    X_base, ids = preprocesar_matriz(matriz)
    labels_base = etiquetas_ward(X_base)
    matriz_sin = matriz.drop(columns=INDICADOR)
    X_sin, ids_sin = preprocesar_matriz(matriz_sin)
    if ids != ids_sin:
        raise AssertionError("El orden de hospitales cambió durante el análisis de sensibilidad")
    labels_sin = etiquetas_ward(X_sin)

    resumen = pd.DataFrame([
        metricas("canonica", X_base, labels_base, labels_base),
        metricas("sin_tasa_partos", X_sin, labels_sin, labels_base),
    ])
    asignaciones = pd.DataFrame({
        "COD_HOSPITAL": ids,
        "cluster_canonico": labels_base,
        "cluster_sin_tasa_partos": labels_sin,
    })
    contingencia = pd.crosstab(
        asignaciones["cluster_canonico"],
        asignaciones["cluster_sin_tasa_partos"],
        margins=True,
    )
    resumen.to_csv(TABLES_DIR / "sensibilidad_indicador_neonatal.csv", index=False)
    asignaciones.to_csv(TABLES_DIR / "sensibilidad_indicador_neonatal_asignaciones.csv", index=False)
    contingencia.to_csv(TABLES_DIR / "sensibilidad_indicador_neonatal_contingencia.csv")

    print("R-05 — ANÁLISIS DE SENSIBILIDAD AL RETIRAR tasa_partos")
    print(resumen.to_string(index=False, float_format="%.4f"))
    print("\nContingencia:")
    print(contingencia.to_string())


if __name__ == "__main__":
    main()
