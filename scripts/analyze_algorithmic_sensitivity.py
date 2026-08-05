"""Concordancia entre la partición Ward K=4 final y alternativas algorítmicas.

Compara Ward con K-means (K=4), GMM (K=4) y MST-kNN (k=6, modo mutuo),
utilizando la misma matriz y el mismo preprocesamiento del análisis principal.

Salida: `reports/tables/concordancia_algoritmica.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "reports" / "tables"
RANDOM_STATE = 42

import sys
sys.path.insert(0, str(ROOT))
from src.modeling.mstknn import MSTkNN
from src.modeling.referencia import COLS_DROP_CLUSTERING as COLS_DROP, LOG1P_COLS

# ---------------------------------------------------------------------------
# 1. Cargar matriz final y aplicar el preprocesamiento del análisis principal.
# ---------------------------------------------------------------------------
matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)

features_clu = matriz.drop(columns=[c for c in COLS_DROP if c in matriz.columns]).copy()
features_clu.index = matriz["COD_HOSPITAL"].values
features_clu = features_clu.fillna(0.0)

for col in LOG1P_COLS:
    if col in features_clu.columns:
        features_clu[col] = np.log1p(features_clu[col].clip(lower=0))

X_scaled = RobustScaler().fit_transform(features_clu.values)
ids = features_clu.index.tolist()

# ---------------------------------------------------------------------------
# 2. Cargar partición Ward K=4 (solución principal, Nivel 1)
# ---------------------------------------------------------------------------
ward = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
ward["COD_HOSPITAL"] = ward["COD_HOSPITAL"].astype(str)
ward_labels = ward.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int)

# ---------------------------------------------------------------------------
# 3. Particiones alternativas: K-means (K=4), GMM (K=4), MST-kNN (k=6, mutuo)
# ---------------------------------------------------------------------------
km = KMeans(n_clusters=4, n_init=50, random_state=RANDOM_STATE).fit(X_scaled)
km_labels = pd.Series(km.labels_, index=ids)

gmm = GaussianMixture(
    n_components=4, n_init=10, random_state=RANDOM_STATE, max_iter=200,
).fit(X_scaled)
gmm_labels = pd.Series(gmm.predict(X_scaled), index=ids)

mst_clu = MSTkNN(k=6, metric="euclidean", knn_simetrico=True)
mst_raw_labels, mst_info = mst_clu.fit_predict(X_scaled)
mst_labels = pd.Series(mst_raw_labels, index=ids)

filas = []
for nombre, labels in [
    ("K-means (K=4)", km_labels),
    ("GMM (K=4)", gmm_labels),
    ("MST-kNN (k=6, mutuo)", mst_labels),
]:
    common = ward_labels.index.intersection(labels.index)
    y_ward = ward_labels.loc[common].values
    y_alt = labels.loc[common].values
    ari = adjusted_rand_score(y_ward, y_alt)
    nmi = normalized_mutual_info_score(y_ward, y_alt)

    # Mainstream Ward (grupo mas grande) vs grupo mas grande de la alternativa
    id_mainstream_ward = pd.Series(y_ward).value_counts().idxmax()
    id_mainstream_alt = pd.Series(y_alt).value_counts().idxmax()
    mainstream_ward = set(common[y_ward == id_mainstream_ward])
    mainstream_alt = set(common[y_alt == id_mainstream_alt])
    jaccard_mainstream = (
        len(mainstream_ward & mainstream_alt) / len(mainstream_ward | mainstream_alt)
    )
    recall_mainstream = len(mainstream_ward & mainstream_alt) / len(mainstream_ward)

    # Especializados Ward (fuera del mainstream): ¿quedan fuera del mainstream alt?
    especializados_ward = set(common) - mainstream_ward
    fuera_mainstream_alt = set(common) - mainstream_alt
    recall_especializados = (
        len(especializados_ward & fuera_mainstream_alt) / len(especializados_ward)
    )

    filas.append({
        "particion_alternativa": nombre,
        "n_grupos": len(set(y_alt)),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "jaccard_mainstream": round(jaccard_mainstream, 4),
        "recall_mainstream": round(recall_mainstream, 4),
        "recall_especializados_fuera_mainstream": round(recall_especializados, 4),
        "n_mainstream_ward": len(mainstream_ward),
        "n_mainstream_alt": len(mainstream_alt),
    })
    print(
        f"{nombre}: ARI={ari:.4f} NMI={nmi:.4f} "
        f"Jaccard(mainstream)={jaccard_mainstream:.4f} "
        f"recall(mainstream)={recall_mainstream:.4f} "
        f"recall(especializados fuera)={recall_especializados:.4f}"
    )

df_concordancia = pd.DataFrame(filas)
df_concordancia.to_csv(TABLES_DIR / "concordancia_algoritmica.csv", index=False)
print(f"\n>>> Persistido: concordancia_algoritmica.csv ({len(df_concordancia)} filas)")
