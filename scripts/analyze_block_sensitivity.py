"""Cuantifica la contribución de bloques y evalúa sensibilidad demográfica (R-02).

El análisis opera sobre la matriz institucional canónica preprocesada con
log1p + RobustScaler. Define cuatro macro-bloques disjuntos:

* gestión, diversidad y CMA (9 variables);
* perfil operativo no demográfico (11 variables);
* demanda demográfica-obstétrica (7 variables);
* componentes PCA de casuística (8 variables).

Se reporta la fracción de distancia euclidiana cuadrada atribuible a cada
bloque. Luego se compara la solución Ward K=4 de referencia con dos
sensibilidades: ponderación por raíz del número de variables del bloque y
exclusión del bloque demográfico-obstétrico.

Uso:
    python3 scripts/analyze_block_sensitivity.py

Salidas en reports/tables/:
    contribucion_distancia_por_bloque.csv
    sensibilidad_bloques.csv
    sensibilidad_bloques_particiones.csv
    sensibilidad_bloques_contingencia.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, preprocesar_matriz
from src.utils.io import PROCESSED_DIR, TABLES_DIR

K_WARD = 4

BLOQUES: dict[str, tuple[str, ...]] = {
    "gestion_diversidad_cma": (
        "egresos_por_anio", "estancia_media", "estancia_mediana",
        "peso_medio_grd", "severidad_media", "mortalidad_media",
        "entropia_grd", "comorbilidades_promedio", "tasa_cma",
    ),
    "perfil_operativo_no_demografico": (
        "pct_urgencia", "pct_programada", "pct_alta_domicilio",
        "pct_alta_fallecido", "pct_alta_traslado", "pct_origen_emergencia",
        "pct_origen_referencia", "pct_uso_pabellon", "pabellones_promedio",
        "cv_estancia", "pct_estancia_larga",
    ),
    "demografico_obstetrico": (
        "pct_pediatrico", "pct_geriatrico", "pct_femenino_fertil",
        "edad_mediana", "pct_obstetrica_ingreso", "tasa_partos",
        "tasa_prematurez",
    ),
    "casuistica_pca": tuple(f"dim_{index:02d}" for index in range(1, 9)),
}


def etiquetas_ward(X: np.ndarray) -> np.ndarray:
    """Construye la partición Ward K=4 para una matriz ya preprocesada."""
    return fcluster(linkage(X, method="ward"), t=K_WARD, criterion="maxclust") - 1


def perfil_particion(nombre: str, X: np.ndarray, labels: np.ndarray, referencia: np.ndarray) -> dict[str, object]:
    """Resume métricas y concordancia contra la solución base."""
    return {
        "escenario": nombre,
        "n_variables": X.shape[1],
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "tamanos": str(pd.Series(labels).value_counts().sort_index().tolist()),
        "ARI_vs_base": float(adjusted_rand_score(referencia, labels)),
        "NMI_vs_base": float(normalized_mutual_info_score(referencia, labels)),
    }


def contribucion_distancia(X: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Descompone la suma de distancias euclidianas cuadradas por bloque."""
    total = 0.0
    contributions: dict[str, float] = {}
    for name, block_columns in BLOQUES.items():
        indices = [columns.index(column) for column in block_columns]
        block = X[:, indices]
        value = float(np.sum((block[:, None, :] - block[None, :, :]) ** 2) / 2)
        contributions[name] = value
        total += value

    rows = []
    for name, value in contributions.items():
        rows.append({
            "bloque": name,
            "n_variables": len(BLOQUES[name]),
            "suma_distancias_cuadradas": value,
            "fraccion_distancia_total": value / total if total else np.nan,
            "peso_igualitario_por_variable": 1 / np.sqrt(len(BLOQUES[name])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 80)
    print("R-02 — CONTRIBUCIÓN DE BLOQUES Y SENSIBILIDAD DEMOGRÁFICA")
    print("=" * 80)

    matrix = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
    matrix["COD_HOSPITAL"] = matrix["COD_HOSPITAL"].astype(str)
    matrix = matrix.sort_values("COD_HOSPITAL", kind="stable").reset_index(drop=True)
    X_base, ids = preprocesar_matriz(matrix)
    columns = [column for column in matrix.columns if column not in COLS_DROP_CLUSTERING]

    expected = {column for block in BLOQUES.values() for column in block}
    if set(columns) != expected:
        missing = sorted(set(columns) - expected)
        extra = sorted(expected - set(columns))
        raise ValueError(f"Los bloques no cubren exactamente la matriz. Sin clasificar={missing}; inexistentes={extra}")

    print(f"\n[1/4] Matriz canónica: {X_base.shape[0]} hospitales x {X_base.shape[1]} variables")
    contributions = contribucion_distancia(X_base, columns)
    contributions.to_csv(TABLES_DIR / "contribucion_distancia_por_bloque.csv", index=False)
    print(contributions.to_string(index=False, float_format="%.4f"))

    print("\n[2/4] Partición Ward base K=4...")
    labels_base = etiquetas_ward(X_base)

    print("\n[3/4] Ponderando cada macro-bloque por 1/sqrt(n_variables)...")
    X_weighted = X_base.copy()
    for name, block_columns in BLOQUES.items():
        indices = [columns.index(column) for column in block_columns]
        X_weighted[:, indices] *= 1 / np.sqrt(len(block_columns))
    labels_weighted = etiquetas_ward(X_weighted)

    print("\n[4/4] Excluyendo el bloque demográfico-obstétrico (7 variables)...")
    retained_columns = [column for column in columns if column not in BLOQUES["demografico_obstetrico"]]
    retained_indices = [columns.index(column) for column in retained_columns]
    X_without_demographic = X_base[:, retained_indices]
    labels_without_demographic = etiquetas_ward(X_without_demographic)

    metrics = pd.DataFrame([
        perfil_particion("base", X_base, labels_base, labels_base),
        perfil_particion("ponderacion_igualitaria_por_bloque", X_weighted, labels_weighted, labels_base),
        perfil_particion("sin_demografico_obstetrico", X_without_demographic, labels_without_demographic, labels_base),
    ])
    metrics.to_csv(TABLES_DIR / "sensibilidad_bloques.csv", index=False)
    print("\nMétricas de escenarios:")
    print(metrics.to_string(index=False, float_format="%.4f"))

    assignments = pd.DataFrame({
        "COD_HOSPITAL": ids,
        "cluster_base": labels_base,
        "cluster_ponderado": labels_weighted,
        "cluster_sin_demografico_obstetrico": labels_without_demographic,
    })
    assignments.to_csv(TABLES_DIR / "sensibilidad_bloques_particiones.csv", index=False)
    contingency = pd.crosstab(
        assignments["cluster_base"],
        assignments["cluster_sin_demografico_obstetrico"],
        rownames=["base"],
        colnames=["sin_demografico_obstetrico"],
    )
    contingency.to_csv(TABLES_DIR / "sensibilidad_bloques_contingencia.csv")
    print("\nContingencia base vs. sin bloque demográfico-obstétrico:")
    print(contingency.to_string())
    print("\nTablas persistidas en reports/tables/.")


if __name__ == "__main__":
    main()
