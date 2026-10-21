"""Evaluación posterior de las alternativas de K del Nivel 1 (observaciones 23 y 26).

La elección de K=4 se hizo con un criterio descriptivo (Sección 3.5.3); este
script NO interviene en esa elección. Documenta, después de elegir, dos cosas
que permiten juzgar el corte K=3 -> K=4:

1. Estabilidad bajo submuestreo de cada K en {2, 3, 4, 5}, con el mismo
   esquema que `analyze_resampling_stability.py` (100 submuestras sin
   reposición de 52 de los 65 hospitales, preprocesamiento re-ajustado en cada
   réplica, semilla 42). Para K=4 reproduce `estabilidad_remuestreo.txt`.
2. Diferencias en escala original entre los dos grupos en que K=4 divide el
   grupo de ocho hospitales de K=3 (C2 y C3): medianas, rangos y si los rangos
   se solapan.

Salidas (reports/tables/):
  - estabilidad_por_k.csv           (ARI/NMI por K y Jaccard por grupo)
  - comparacion_c2_c3_escala_original.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import PROCESSED_DIR, TABLES_DIR

K_VALUES = (2, 3, 4, 5)
N_ITERACIONES = 100
FRAC_SUBMUESTRA = 0.80
RANDOM_STATE = 42

matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
matriz = matriz.sort_values("COD_HOSPITAL", kind="stable").reset_index(drop=True)
ids = matriz["COD_HOSPITAL"].to_numpy()
cols_clustering = [c for c in matriz.columns if c not in COLS_DROP_CLUSTERING]
n_total = len(matriz)
n_sub = int(round(FRAC_SUBMUESTRA * n_total))


def preprocesar(frame: pd.DataFrame) -> np.ndarray:
    """log1p + RobustScaler re-ajustados sobre el subconjunto recibido."""
    feats = frame[cols_clustering].fillna(0.0).copy()
    for col in LOG1P_COLS:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    return RobustScaler().fit_transform(feats.values)


def jaccard_mejor_pareja(miembros: np.ndarray, etiquetas_sub: np.ndarray) -> float:
    """Jaccard entre un grupo original y el grupo de la réplica que más se le parece."""
    original = set(np.flatnonzero(miembros))
    return max(
        len(original & set(np.flatnonzero(etiquetas_sub == g)))
        / len(original | set(np.flatnonzero(etiquetas_sub == g)))
        for g in np.unique(etiquetas_sub)
    )


X_full = preprocesar(matriz)
Z_full = linkage(X_full, method="ward")

filas_k = []
for k in K_VALUES:
    labels_full = fcluster(Z_full, t=k, criterion="maxclust")
    tamanos = pd.Series(labels_full).value_counts().sort_index()
    rng = np.random.default_rng(RANDOM_STATE)
    aris, nmis = [], []
    jaccard = {g: [] for g in tamanos.index}
    for _ in range(N_ITERACIONES):
        idx = np.sort(rng.choice(n_total, size=n_sub, replace=False))
        labels_sub = fcluster(
            linkage(preprocesar(matriz.iloc[idx]), method="ward"), t=k, criterion="maxclust"
        )
        orig = labels_full[idx]
        aris.append(adjusted_rand_score(orig, labels_sub))
        nmis.append(normalized_mutual_info_score(orig, labels_sub))
        for g in jaccard:
            miembros = orig == g
            if miembros.any():
                jaccard[g].append(jaccard_mejor_pareja(miembros, labels_sub))
    # Lista (tamaño, Jaccard) para no colapsar grupos de igual tamaño.
    jaccard_medio = sorted(
        ((int(tamanos[g]), round(float(np.mean(v)), 3)) for g, v in jaccard.items()),
        reverse=True,
    )
    filas_k.append({
        "K": k,
        "tamanos": str(sorted(tamanos.tolist(), reverse=True)),
        "ari_mediana": float(np.median(aris)),
        "ari_p2_5": float(np.percentile(aris, 2.5)),
        "ari_p97_5": float(np.percentile(aris, 97.5)),
        "nmi_mediana": float(np.median(nmis)),
        "jaccard_minimo_por_grupo": float(min(np.mean(v) for v in jaccard.values())),
        "jaccard_por_grupo_tamano_valor": str(jaccard_medio),
        "n_iteraciones": N_ITERACIONES,
        "n_submuestra": n_sub,
    })

df_k = pd.DataFrame(filas_k)
df_k.to_csv(TABLES_DIR / "estabilidad_por_k.csv", index=False)
print(df_k.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------------------------------------------------------------------------
# C2 frente a C3 en escala original (el corte K=3 -> K=4)
# ---------------------------------------------------------------------------
asig = (
    pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
    .set_index("COD_HOSPITAL")["nivel1_K4"].astype(int).reindex(ids)
)
labels_k3 = fcluster(Z_full, t=3, criterion="maxclust")
mask_ocho = asig.isin([2, 3]).to_numpy()
if len(set(labels_k3[mask_ocho])) != 1 or mask_ocho.sum() != 8:
    raise ValueError("C2 y C3 no coinciden con el grupo de ocho hospitales de K=3.")

cols_directas = [
    c for c in matriz.columns
    if c not in ("COD_HOSPITAL", "peso_medio_cma_imputado") and not c.startswith("dim_")
]
filas_c = []
for col in cols_directas:
    c2 = matriz.loc[(asig == 2).to_numpy(), col].astype(float)
    c3 = matriz.loc[(asig == 3).to_numpy(), col].astype(float)
    filas_c.append({
        "variable": col,
        "mediana_C2": c2.median(), "min_C2": c2.min(), "max_C2": c2.max(),
        "mediana_C3": c3.median(), "min_C3": c3.min(), "max_C3": c3.max(),
        "rangos_sin_solape": bool(c2.max() < c3.min() or c3.max() < c2.min()),
    })
df_c = pd.DataFrame(filas_c).sort_values(["rangos_sin_solape", "variable"], ascending=[False, True])
df_c.to_csv(TABLES_DIR / "comparacion_c2_c3_escala_original.csv", index=False)
print("\nVariables con rangos sin solape entre C2 y C3:")
print(df_c[df_c.rangos_sin_solape].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nPersistido: estabilidad_por_k.csv, comparacion_c2_c3_escala_original.csv")
