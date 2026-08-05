"""Reconstruye la matriz institucional final y las métricas Ward de Nivel 1.

Uso:
    python3 scripts/build_dataset.py

El script usa el GRD enriquecido de 91 columnas almacenado en
``data/processed/grd_filtrado.parquet``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.etl.casuistica import Constructor_Features
from src.etl.cie_mappers import Mapeador_CIE10, Mapeador_CIE9
from src.etl.features_extendidas import Constructor_Features_Extendidas
from src.modeling.matriz import Constructor_Matriz
from src.modeling.reducer import Reductor_Dimensional
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import TABLES_DIR, cargar_grd_compacto, get_path, write_parquet

GRD_PATH = get_path("grd_filtrado")
MATRIX_PATH = get_path("hospital_matrix")
SCALED_MATRIX_PATH = get_path("hospital_matrix_scaled")
METRICS_PATH = TABLES_DIR / "clustering_ward_metricas.csv"


def preparar_matriz(matriz: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    matriz = matriz.copy()
    matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
    features = matriz.drop(columns=[column for column in COLS_DROP_CLUSTERING if column in matriz.columns])
    features = features.fillna(0.0)
    for column in LOG1P_COLS:
        if column in features.columns:
            features[column] = np.log1p(features[column].clip(lower=0))
    return RobustScaler().fit_transform(features), matriz["COD_HOSPITAL"].tolist(), features.columns.tolist()


def evaluar_ward(X: np.ndarray) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    linkage_matrix = linkage(X, method="ward")
    rows: list[dict[str, object]] = []
    assignments: dict[int, np.ndarray] = {}
    for k in range(2, 11):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
        assignments[k] = labels
        sizes = pd.Series(labels).value_counts().sort_index().tolist()
        rows.append({
            "K": k,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "tamanos": str(sizes),
            "mainstream_size": max(sizes),
        })
    return pd.DataFrame(rows), assignments


def main() -> None:
    if not GRD_PATH.exists():
        raise FileNotFoundError(f"No existe el insumo GRD enriquecido: {GRD_PATH}")

    print("[1/4] Cargando egresos GRD enriquecidos y tablas CIE...")
    df_grd = cargar_grd_compacto(GRD_PATH)
    cie10 = pd.read_excel(ROOT / "insumos" / "maestras" / "CIE-10.xlsx")
    cie9 = pd.read_excel(ROOT / "insumos" / "maestras" / "CIE-9 .xlsx")
    mapper_cie10 = Mapeador_CIE10(cie10)
    mapper_cie9 = Mapeador_CIE9(cie9)
    mapper_cie10.construir()
    mapper_cie9.construir()

    print("[2/4] Construyendo indicadores y vectores de casuística...")
    constructor = Constructor_Features(df_grd, mapper_cie10, mapper_cie9)
    features_tradicionales = constructor.features_tradicionales()
    features_diversidad = constructor.features_diversidad()
    features_cma = constructor.features_cma()
    capitulos = constructor.vector_capitulos_ponderado()
    procedimientos = constructor.vector_secciones()
    top20 = constructor.top20_grds_nacionales()
    vector_top20 = constructor.vector_top20(top20)
    write_parquet(features_tradicionales, "features_tradicionales", ["COD_HOSPITAL"])
    write_parquet(features_diversidad, "features_diversidad", ["COD_HOSPITAL"])
    write_parquet(features_cma, "features_cma", ["COD_HOSPITAL"])

    print("[3/4] Reduciendo casuística y ensamblando matriz final...")
    casuistica = capitulos.merge(procedimientos, on="COD_HOSPITAL", how="inner").merge(vector_top20, on="COD_HOSPITAL", how="inner")
    pca = Reductor_Dimensional(casuistica).fit_pca()
    write_parquet(pca["componentes"], "casuistica_reducida", ["COD_HOSPITAL"])
    pca["loadings"].to_csv(TABLES_DIR / "reduccion_dimensional.csv")
    elegibles = pd.Index(features_tradicionales["COD_HOSPITAL"].astype(str))
    df_grd["COD_HOSPITAL"] = df_grd["COD_HOSPITAL"].astype(str)
    features_extendidas = Constructor_Features_Extendidas(df_grd, elegibles).construir_todas()
    write_parquet(features_extendidas, "features_extendidas", ["COD_HOSPITAL"])
    matrix = Constructor_Matriz(features_tradicionales, features_diversidad, features_cma, pca["componentes"], features_extendidas).construir()
    matrix["COD_HOSPITAL"] = matrix["COD_HOSPITAL"].astype(str)
    matrix.to_parquet(MATRIX_PATH, index=False, compression="zstd")

    print("[4/4] Escalando y evaluando Ward entre K=2 y K=10...")
    X_scaled, hospital_ids, feature_names = preparar_matriz(matrix)
    pd.DataFrame(X_scaled, columns=feature_names).assign(COD_HOSPITAL=hospital_ids).loc[:, ["COD_HOSPITAL", *feature_names]].to_parquet(SCALED_MATRIX_PATH, index=False, compression="zstd")
    metrics, _ = evaluar_ward(X_scaled)
    metrics.to_csv(METRICS_PATH, index=False)
    print(f"Matriz: {MATRIX_PATH}")
    print(f"Métricas Ward: {METRICS_PATH}")
    # La asignacion reportada la fija build_hierarchical_clustering.py en
    # reports/tables/asignacion_jerarquica_final.csv, que es la unica canonica.
    print("Asignación reportada: ejecutar scripts/build_hierarchical_clustering.py")


if __name__ == "__main__":
    main()
