"""Evalúa el efecto de atípicos globales detectados antes del clustering (R-03).

Isolation Forest se ajusta sobre la matriz canónica preprocesada de 65
hospitales antes de Ward. Las observaciones marcadas con contamination=0.10 se
excluyen temporalmente, Ward K=4 se ajusta sobre los hospitales restantes y las
marcas excluidas se asignan al centroide más próximo solo para comparar ambas
particiones sobre el universo completo.

La solución publicada no elimina hospitales: este es un análisis de sensibilidad
que determina si los atípicos globales distorsionan materialmente la partición.

Uso:
    python3 scripts/analyze_preclustering_outlier_impact.py

Salidas en reports/tables/:
    sensibilidad_atipicos_preclustering_resumen.csv
    sensibilidad_atipicos_preclustering_marcas.csv
    sensibilidad_atipicos_preclustering_asignaciones.csv
    sensibilidad_atipicos_preclustering_contingencia.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import particion_referencia
from src.utils.io import PROCESSED_DIR, TABLES_DIR

CONTAMINATION = 0.10
N_ESTIMATORS = 300
RANDOM_STATE = 42
K_WARD = 4


def ward_labels(X: np.ndarray) -> np.ndarray:
    """Ajusta Ward con el número de grupos de la solución principal."""
    return fcluster(linkage(X, method="ward"), t=K_WARD, criterion="maxclust") - 1


def metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Calcula métricas internas de una partición con al menos dos grupos."""
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "tamanos": str(pd.Series(labels).value_counts().sort_index().tolist()),
    }


def main() -> None:
    print("=" * 80)
    print("R-03 — SENSIBILIDAD DE ATÍPICOS PREVIA AL CLUSTERING")
    print("=" * 80)

    # Reutilizar  preprocesamiento y la particion canon para evitar que
    # el análisis de sensibilidad difiera por transformación u orden de filas
    X, labels_base, ids = particion_referencia(PROCESSED_DIR / "hospital_matrix.parquet")
    names = (
        pd.read_csv(TABLES_DIR / "hospitales_por_cluster_nivel1.csv", dtype=str)
        .assign(COD_HOSPITAL=lambda frame: frame["COD_HOSPITAL"].astype(str))
        .set_index("COD_HOSPITAL")["NOMBRE"]
        .to_dict()
    )

    print(f"\n[1/4] Matriz canónica: {X.shape[0]} hospitales x {X.shape[1]} variables")
    detector = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    predictions = detector.fit_predict(X)
    scores = detector.score_samples(X)
    excluded = predictions == -1
    retained = ~excluded
    print(f"  Isolation Forest previo: {int(excluded.sum())} marcas de {len(ids)} "
          f"(contamination={CONTAMINATION:.2f})")

    print("\n[2/4] Reajustando Ward K=4 sin las marcas de Isolation Forest...")
    X_retained = X[retained]
    labels_retained = ward_labels(X_retained)
    centroids = np.vstack([
        X_retained[labels_retained == cluster].mean(axis=0)
        for cluster in sorted(np.unique(labels_retained))
    ])

    # Para la comparación de 65 hospitales: cada caso excluido recibe el grupo
    # cuyo centroide entrenado sin el queda más cercano. No participa en el
    # ajuste ni altera los centroides.
    labels_sensitivity = np.empty(len(ids), dtype=int)
    labels_sensitivity[retained] = labels_retained
    distances = np.linalg.norm(X[excluded, None, :] - centroids[None, :, :], axis=2)
    labels_sensitivity[excluded] = distances.argmin(axis=1)

    print("\n[3/4] Comparando particiones base y sin marcas pre-clustering...")
    ari_retained = adjusted_rand_score(labels_base[retained], labels_retained)
    nmi_retained = normalized_mutual_info_score(labels_base[retained], labels_retained)
    ari_full = adjusted_rand_score(labels_base, labels_sensitivity)
    nmi_full = normalized_mutual_info_score(labels_base, labels_sensitivity)

    baseline_metrics = metrics(X, labels_base)
    retained_metrics = metrics(X_retained, labels_retained)
    reassigned_metrics = metrics(X, labels_sensitivity)
    summary = pd.DataFrame([
        {
            "escenario": "base_65_hospitales",
            "n_hospitales_ajuste": len(ids),
            "n_excluidos_isolation_forest": 0,
            **baseline_metrics,
            "ARI_vs_base_65": 1.0,
            "NMI_vs_base_65": 1.0,
            "ARI_vs_base_retenidos": 1.0,
            "NMI_vs_base_retenidos": 1.0,
        },
        {
            "escenario": "ward_sin_marcas_IF_58_hospitales",
            "n_hospitales_ajuste": int(retained.sum()),
            "n_excluidos_isolation_forest": int(excluded.sum()),
            **retained_metrics,
            "ARI_vs_base_65": np.nan,
            "NMI_vs_base_65": np.nan,
            "ARI_vs_base_retenidos": ari_retained,
            "NMI_vs_base_retenidos": nmi_retained,
        },
        {
            "escenario": "sin_IF_con_marcas_reasignadas_65_hospitales",
            "n_hospitales_ajuste": int(retained.sum()),
            "n_excluidos_isolation_forest": int(excluded.sum()),
            **reassigned_metrics,
            "ARI_vs_base_65": ari_full,
            "NMI_vs_base_65": nmi_full,
            "ARI_vs_base_retenidos": ari_retained,
            "NMI_vs_base_retenidos": nmi_retained,
        },
    ])
    summary.to_csv(TABLES_DIR / "sensibilidad_atipicos_preclustering_resumen.csv", index=False)
    print(summary.to_string(index=False, float_format="%.4f"))

    labels_trained_full = np.full(len(ids), np.nan)
    labels_trained_full[retained] = labels_retained
    assignments = pd.DataFrame({
        "COD_HOSPITAL": ids,
        "NOMBRE": [names.get(hospital, "?") for hospital in ids],
        "marcado_isolation_forest_preclustering": excluded,
        "iso_score": scores,
        "cluster_base": labels_base,
        "cluster_sin_IF": labels_sensitivity,
        "cluster_sin_IF_entrenado": labels_trained_full,
    })
    assignments.to_csv(TABLES_DIR / "sensibilidad_atipicos_preclustering_asignaciones.csv", index=False)
    marks = assignments[assignments["marcado_isolation_forest_preclustering"]].copy()
    marks["asignacion_por_centroide_mas_proximo"] = marks["cluster_sin_IF"]
    marks.to_csv(TABLES_DIR / "sensibilidad_atipicos_preclustering_marcas.csv", index=False)
    contingency = pd.crosstab(
        assignments["cluster_base"],
        assignments["cluster_sin_IF"],
        rownames=["base"],
        colnames=["sin_IF"],
    )
    contingency.to_csv(TABLES_DIR / "sensibilidad_atipicos_preclustering_contingencia.csv")

    print("\n[4/4] Marcas previas y asignación al centroide más próximo:")
    print(marks[["COD_HOSPITAL", "NOMBRE", "cluster_base", "asignacion_por_centroide_mas_proximo"]]
          .to_string(index=False))
    print("\nContingencia base vs. sensibilidad con marcas reasignadas:")
    print(contingency.to_string())
    print("\nTablas persistidas en reports/tables/.")


if __name__ == "__main__":
    main()
