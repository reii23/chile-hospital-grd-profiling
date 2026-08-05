"""Clustering alternativos para análisis de sensibilidad (notebook 05).

Implementa 5 técnicas distintas para comparar contra K-means++ K=2 del notebook 03:

1. **K-means forzado** con K ∈ {2..8}: tabla con métricas para cada K
2. **Aglomerativo (Ward)** + dendrograma: corte a K=2..8
3. **HDBSCAN**: density-based, NO especifica K, identifica outliers (-1)
4. **KNN-graph + Leiden**: community detection con resolución variable
5. **Gaussian Mixture Model**: probabilístico, criterio BIC para selección K

Toma la `Hospital_Matrix_Integrada` ya escalada y produce, para cada método y
configuración:
- Etiquetas de cluster
- Silhouette + Calinski-Harabasz + Davies-Bouldin
- Composición (n hospitales por cluster)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import kneighbors_graph

# Imports condicionales (instalar con `pip install hdbscan python-igraph leidenalg`)
try:
    import hdbscan as hdbscan_lib
    HDBSCAN_DISPONIBLE = True
except ImportError:
    HDBSCAN_DISPONIBLE = False

try:
    import igraph as ig
    import leidenalg
    LEIDEN_DISPONIBLE = True
except ImportError:
    LEIDEN_DISPONIBLE = False


RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers métricas
# ---------------------------------------------------------------------------
def _calcular_metricas(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette + CH + DB. Maneja casos degenerados (cluster único, outliers)."""
    # Filtrar outliers (label = -1) antes de calcular Silhouette
    mask_no_outlier = labels != -1
    if mask_no_outlier.sum() < 3:
        return {"silhouette": -1.0, "calinski_harabasz": 0.0, "davies_bouldin": np.inf}

    X_clean = X[mask_no_outlier]
    labels_clean = labels[mask_no_outlier]
    n_clusters = len(set(labels_clean))

    if n_clusters < 2:
        return {"silhouette": -1.0, "calinski_harabasz": 0.0, "davies_bouldin": np.inf}

    try:
        sil = silhouette_score(X_clean, labels_clean)
        ch = calinski_harabasz_score(X_clean, labels_clean)
        db = davies_bouldin_score(X_clean, labels_clean)
    except ValueError:
        sil, ch, db = -1.0, 0.0, np.inf

    return {
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
    }


# ---------------------------------------------------------------------------
# 1. K-means con K ∈ {2..8}
# ---------------------------------------------------------------------------
def kmeans_rango(
    X: np.ndarray, k_grid: tuple[int, ...] = tuple(range(2, 9)),
    n_init: int = 50, random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Ejecuta K-means para cada K en el grid. Retorna métricas + etiquetas."""
    rows = []
    labels_dict = {}
    for k in k_grid:
        if k >= len(X):
            continue
        km = KMeans(
            n_clusters=k, init="k-means++",
            n_init=n_init, random_state=random_state,
        )
        labels = km.fit_predict(X)
        m = _calcular_metricas(X, labels)
        rows.append({
            "metodo": "K-means++",
            "config": f"K={k}",
            "n_clusters_efectivos": k,
            **m,
        })
        labels_dict[k] = labels
    return pd.DataFrame(rows), labels_dict


# ---------------------------------------------------------------------------
# 2. Clustering aglomerativo (Ward) con dendrograma
# ---------------------------------------------------------------------------
def aglomerativo_rango(
    X: np.ndarray, k_grid: tuple[int, ...] = tuple(range(2, 9)),
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Aglomerativo con linkage Ward, corte para cada K."""
    rows = []
    labels_dict = {}
    for k in k_grid:
        if k >= len(X):
            continue
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        m = _calcular_metricas(X, labels)
        rows.append({
            "metodo": "Aglomerativo (Ward)",
            "config": f"K={k}",
            "n_clusters_efectivos": k,
            **m,
        })
        labels_dict[k] = labels
    return pd.DataFrame(rows), labels_dict


# ---------------------------------------------------------------------------
# 3. HDBSCAN — density-based, NO requiere K
# ---------------------------------------------------------------------------
def hdbscan_rango(
    X: np.ndarray,
    min_cluster_sizes: tuple[int, ...] = (3, 4, 5, 6, 8),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """HDBSCAN con varios min_cluster_size. Identifica outliers como -1."""
    if not HDBSCAN_DISPONIBLE:
        return pd.DataFrame(), {}

    rows = []
    labels_dict = {}
    for mcs in min_cluster_sizes:
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=mcs,
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        n_outliers = int((labels == -1).sum())
        m = _calcular_metricas(X, labels)
        config = f"min_cluster_size={mcs}"
        rows.append({
            "metodo": "HDBSCAN",
            "config": config,
            "n_clusters_efectivos": n_clusters,
            "n_outliers": n_outliers,
            **m,
        })
        labels_dict[config] = labels
    return pd.DataFrame(rows), labels_dict


# ---------------------------------------------------------------------------
# 4. KNN-graph + Leiden community detection
# ---------------------------------------------------------------------------
def knn_leiden_rango(
    X: np.ndarray,
    n_neighbors_list: tuple[int, ...] = (10, 15, 20),
    resoluciones: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Construye grafo k-NN y aplica algoritmo Leiden con varias resoluciones."""
    if not LEIDEN_DISPONIBLE:
        return pd.DataFrame(), {}

    rows = []
    labels_dict = {}
    n = X.shape[0]
    for k_nn in n_neighbors_list:
        if k_nn >= n:
            continue
        # Grafo KNN simétrico (modo conectividad: pesos binarios)
        graph_sparse = kneighbors_graph(
            X, n_neighbors=k_nn, mode="connectivity", include_self=False
        )
        # Convertir a igraph
        coo = graph_sparse.tocoo()
        edges = list(zip(coo.row.tolist(), coo.col.tolist()))
        g = ig.Graph(n=n, edges=edges, directed=False)
        g.simplify()  # eliminar duplicados y self-loops

        for res in resoluciones:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res,
                seed=random_state,
            )
            labels = np.array(partition.membership)
            n_clusters = len(set(labels))
            m = _calcular_metricas(X, labels)
            config = f"k_nn={k_nn}, res={res}"
            rows.append({
                "metodo": "KNN-graph + Leiden",
                "config": config,
                "n_clusters_efectivos": n_clusters,
                **m,
            })
            labels_dict[config] = labels
    return pd.DataFrame(rows), labels_dict


# ---------------------------------------------------------------------------
# 5. Gaussian Mixture Model con BIC
# ---------------------------------------------------------------------------
def gmm_rango(
    X: np.ndarray,
    k_grid: tuple[int, ...] = tuple(range(2, 9)),
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Gaussian Mixture con BIC. Selecciona K donde BIC es mínimo."""
    rows = []
    labels_dict = {}
    for k in k_grid:
        if k >= len(X):
            continue
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=random_state, max_iter=200,
        )
        try:
            gmm.fit(X)
            labels = gmm.predict(X)
            bic = gmm.bic(X)
            aic = gmm.aic(X)
        except (np.linalg.LinAlgError, ValueError):
            continue
        m = _calcular_metricas(X, labels)
        rows.append({
            "metodo": "GMM",
            "config": f"K={k}",
            "n_clusters_efectivos": k,
            "bic": float(bic),
            "aic": float(aic),
            **m,
        })
        labels_dict[k] = labels
    return pd.DataFrame(rows), labels_dict


# ---------------------------------------------------------------------------
# Comparación: ARI cruzado entre métodos
# ---------------------------------------------------------------------------
def comparar_etiquetas(labels_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    """ARI entre todos los pares de etiquetados."""
    from sklearn.metrics import adjusted_rand_score

    keys = list(labels_dict.keys())
    rows = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            try:
                ari = adjusted_rand_score(labels_dict[k1], labels_dict[k2])
            except ValueError:
                ari = np.nan
            rows.append({"metodo_1": k1, "metodo_2": k2, "ARI": ari})
    return pd.DataFrame(rows).sort_values("ARI", ascending=False).reset_index(drop=True)
