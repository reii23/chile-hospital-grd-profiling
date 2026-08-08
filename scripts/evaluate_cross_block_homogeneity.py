"""Evalúa homogeneidad Ward vs. MINSAL en un bloque de variables no usado para agrupar.

Implementa la corrección R-01 mediante validación cruzada por bloques:

* Ward K=4 se ajusta en las 27 variables clínico-operativas directas y se
  evalúa, junto con MINSAL, en las 8 componentes PCA de casuística.
* Ward K=4 se ajusta en las 8 componentes PCA de casuística y se evalúa,
  junto con MINSAL, en las 27 variables directas.

En cada dirección, ninguna de las dos particiones optimiza el bloque de
validación. Se reportan WSS/TSS, silhouette y la diferencia de WSS/TSS
(MINSAL - Ward), con un intervalo percentilar bootstrap sobre hospitales.

Uso:
    python3 scripts/evaluate_cross_block_homogeneity.py

Salida:
    reports/tables/homogeneidad_validacion_bloques.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import PROCESSED_DIR, TABLES_DIR

BASE_EST = ROOT / "info-hospitales" / "Base de Establecimientos 2023.xlsx"
K_WARD_VALUES = (2, 4)
N_BOOTSTRAP = 5_000
RANDOM_STATE = 42


def transformar_bloque(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Aplica las transformaciones del pipeline solo al bloque indicado."""
    values = frame.loc[:, columns].copy().fillna(0.0)
    for column in LOG1P_COLS:
        if column in values.columns:
            values[column] = np.log1p(values[column].clip(lower=0))
    return RobustScaler().fit_transform(values)


def wss_tss(X: np.ndarray, labels: np.ndarray) -> float:
    """Calcula la fracción de dispersión total que permanece intra-grupo."""
    global_center = X.mean(axis=0)
    tss = float(np.sum((X - global_center) ** 2))
    if tss == 0:
        return float("nan")

    wss = 0.0
    for group in np.unique(labels):
        group_values = X[labels == group]
        center = group_values.mean(axis=0)
        wss += float(np.sum((group_values - center) ** 2))
    return wss / tss


def resumen_particion(X_eval: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Devuelve WSS/TSS y silhouette de una partición sobre el bloque de evaluación."""
    return wss_tss(X_eval, labels), float(silhouette_score(X_eval, labels))


def bootstrap_delta_wss(
    X_eval: np.ndarray,
    ward_labels: np.ndarray,
    minsal_labels: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """IC 95 % para WSS/TSS(MINSAL) - WSS/TSS(Ward) por remuestreo hospitalario."""
    rng = np.random.default_rng(random_state)
    n_hospitals = len(ward_labels)
    deltas = np.empty(n_bootstrap)
    for replicate in range(n_bootstrap):
        indices = rng.integers(0, n_hospitals, size=n_hospitals)
        deltas[replicate] = (
            wss_tss(X_eval[indices], minsal_labels[indices])
            - wss_tss(X_eval[indices], ward_labels[indices])
        )
    return tuple(float(value) for value in np.percentile(deltas, [2.5, 97.5]))


def cargar_etiquetas_minsal(ids: list[str]) -> np.ndarray:
    """Carga y valida la clasificación MINSAL para todos los hospitales analizados."""
    base = pd.read_excel(BASE_EST, skiprows=1)
    base.columns = [str(column).strip() for column in base.columns]
    base["COD_HOSPITAL"] = (
        base["Código Vigente"].astype(str).str.replace(".0", "", regex=False).str.strip()
    )
    labels = (
        pd.Series(ids)
        .map(
            base.dropna(subset=["Nivel de Complejidad"])
            .drop_duplicates("COD_HOSPITAL")
            .set_index("COD_HOSPITAL")["Nivel de Complejidad"]
        )
        .map({"Alta Complejidad": 0, "Mediana Complejidad": 1})
    )
    if labels.isna().any():
        missing = pd.Series(ids)[labels.isna()].tolist()
        raise ValueError(f"Hospitales sin clasificación MINSAL utilizable: {missing}")
    return labels.to_numpy(dtype=int)


def main() -> None:
    matrix = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet").copy()
    matrix["COD_HOSPITAL"] = matrix["COD_HOSPITAL"].astype(str)
    matrix = matrix.sort_values("COD_HOSPITAL", kind="stable").reset_index(drop=True)
    ids = matrix["COD_HOSPITAL"].tolist()

    clustering_columns = [
        column for column in matrix.columns if column not in COLS_DROP_CLUSTERING
    ]
    direct_columns = [column for column in clustering_columns if not column.startswith("dim_")]
    casuistica_columns = [column for column in clustering_columns if column.startswith("dim_")]
    if len(direct_columns) != 27 or len(casuistica_columns) != 8:
        raise ValueError(
            "La partición por bloques esperada es 27 variables directas y 8 componentes PCA; "
            f"se encontraron {len(direct_columns)} y {len(casuistica_columns)}."
        )

    minsal_labels = cargar_etiquetas_minsal(ids)
    blocks = {
        "variables directas (27)": direct_columns,
        "componentes PCA de casuística (8)": casuistica_columns,
    }
    rows: list[dict[str, object]] = []

    for training_name, training_columns in blocks.items():
        evaluation_name, evaluation_columns = next(
            (name, columns) for name, columns in blocks.items() if name != training_name
        )
        X_train = transformar_bloque(matrix, training_columns)
        X_eval = transformar_bloque(matrix, evaluation_columns)
        hierarchy = linkage(X_train, method="ward")
        minsal_wss, minsal_silhouette = resumen_particion(X_eval, minsal_labels)

        for k_ward in K_WARD_VALUES:
            ward_labels = fcluster(hierarchy, t=k_ward, criterion="maxclust") - 1
            ward_wss, ward_silhouette = resumen_particion(X_eval, ward_labels)
            ci_lower, ci_upper = bootstrap_delta_wss(X_eval, ward_labels, minsal_labels)
            rows.append({
                "bloque_agrupamiento": training_name,
                "bloque_evaluacion": evaluation_name,
                "K_ward": k_ward,
                "tamanos_ward": str(pd.Series(ward_labels).value_counts().sort_index().tolist()),
                "tamanos_minsal": str(pd.Series(minsal_labels).value_counts().sort_index().tolist()),
                "wss_tss_ward": ward_wss,
                "wss_tss_minsal": minsal_wss,
                "delta_wss_tss_minsal_menos_ward": minsal_wss - ward_wss,
                "ic95_delta_wss_tss_inferior": ci_lower,
                "ic95_delta_wss_tss_superior": ci_upper,
                "silhouette_ward_en_bloque_evaluacion": ward_silhouette,
                "silhouette_minsal_en_bloque_evaluacion": minsal_silhouette,
                "n_hospitales": len(ids),
                "n_bootstrap": N_BOOTSTRAP,
            })

    result = pd.DataFrame(rows)
    output = TABLES_DIR / "homogeneidad_validacion_bloques.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False, float_format="%.4f"))
    print(f"\nPersistido: {output}")


if __name__ == "__main__":
    main()
