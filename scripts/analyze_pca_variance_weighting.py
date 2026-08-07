"""Ponderación de componentes PCA por varianza explicada.

Recalcula la varianza individual de dim_01..dim_08 desde los mismos vectores
ponderados de casuística usados por el pipeline. Luego compara Ward K=4
canónico con una sensibilidad que, tras log1p y RobustScaler, multiplica cada
componente PCA por la raíz de su proporción de varianza explicada. Esto evita
que las componentes tardías contribuyan igual a la distancia que dim_01

No modifica la configuración actual se trata como analisis complementario
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import PROCESSED_DIR, TABLES_DIR

K_COMPONENTES = 8
K_WARD = 4


def preprocesar(matriz: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    datos = matriz.copy()
    datos["COD_HOSPITAL"] = datos["COD_HOSPITAL"].astype(str)
    datos = datos.sort_values("COD_HOSPITAL").reset_index(drop=True)
    columnas = [c for c in datos.columns if c not in COLS_DROP_CLUSTERING]
    features = datos[columnas].fillna(0.0)
    for columna in LOG1P_COLS:
        if columna in features:
            features[columna] = np.log1p(features[columna].clip(lower=0))
    return RobustScaler().fit_transform(features), datos["COD_HOSPITAL"].tolist(), columnas


def ward(X: np.ndarray) -> np.ndarray:
    return fcluster(linkage(X, method="ward"), t=K_WARD, criterion="maxclust") - 1


def metricas(nombre: str, X: np.ndarray, labels: np.ndarray, referencia: np.ndarray) -> dict[str, object]:
    return {
        "escenario": nombre,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "tamanos": str(pd.Series(labels).value_counts().sort_index().tolist()),
        "ARI_vs_canonica": adjusted_rand_score(referencia, labels),
        "NMI_vs_canonica": normalized_mutual_info_score(referencia, labels),
    }


def cargar_varianza_pca() -> pd.DataFrame:
    caps = pd.read_parquet(PROCESSED_DIR / "casuistica_capitulos.parquet")
    secciones = pd.read_parquet(PROCESSED_DIR / "casuistica_procedimientos.parquet")
    grds = pd.read_parquet(PROCESSED_DIR / "casuistica_top20_grds.parquet")
    caps = caps[caps["variante"] == "ponderado"].drop(columns="variante")
    casuistica = caps.merge(secciones, on="COD_HOSPITAL", validate="1:1").merge(
        grds, on="COD_HOSPITAL", validate="1:1"
    )
    X = StandardScaler().fit_transform(casuistica.drop(columns="COD_HOSPITAL"))
    pca = PCA(n_components=K_COMPONENTES, random_state=42).fit(X)
    evr = pca.explained_variance_ratio_
    return pd.DataFrame({
        "componente": [f"dim_{i:02d}" for i in range(1, K_COMPONENTES + 1)],
        "varianza_explicada": evr,
        "varianza_acumulada": np.cumsum(evr),
        "peso_sqrt_varianza": np.sqrt(evr),
    })


def main() -> None:
    matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
    X, ids, columnas = preprocesar(matriz)
    labels_base = ward(X)
    varianza = cargar_varianza_pca()
    pesos = dict(zip(varianza["componente"], varianza["peso_sqrt_varianza"]))

    X_ponderada = X.copy()
    for componente, peso in pesos.items():
        X_ponderada[:, columnas.index(componente)] *= peso
    labels_ponderada = ward(X_ponderada)

    resumen = pd.DataFrame([
        metricas("canonica_componentes_peso_igual", X, labels_base, labels_base),
        metricas("pca_ponderada_por_sqrt_varianza", X_ponderada, labels_ponderada, labels_base),
    ])
    asignaciones = pd.DataFrame({
        "COD_HOSPITAL": ids,
        "cluster_canonico": labels_base,
        "cluster_pca_ponderada": labels_ponderada,
    })
    contingencia = pd.crosstab(asignaciones["cluster_canonico"], asignaciones["cluster_pca_ponderada"], margins=True)

    varianza.to_csv(TABLES_DIR / "pca_varianza_componentes.csv", index=False)
    resumen.to_csv(TABLES_DIR / "sensibilidad_pca_ponderacion_varianza.csv", index=False)
    asignaciones.to_csv(TABLES_DIR / "sensibilidad_pca_ponderacion_asignaciones.csv", index=False)
    contingencia.to_csv(TABLES_DIR / "sensibilidad_pca_ponderacion_contingencia.csv")

    print("R-06 — PONDERACIÓN PCA POR VARIANZA EXPLICADA")
    print("\nVarianza por componente:")
    print(varianza.to_string(index=False, float_format="%.4f"))
    print("\nComparación Ward K=4:")
    print(resumen.to_string(index=False, float_format="%.4f"))
    print("\nContingencia:")
    print(contingencia.to_string())


if __name__ == "__main__":
    main()
