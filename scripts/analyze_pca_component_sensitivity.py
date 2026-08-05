"""Sensibilidad al numero de componentes PCA de casuistica: K=8 (oficial) vs K=11
(umbral real de 80% de varianza acumulada declarado en VAR_THRESHOLD).

Contexto: src/modeling/reducer.py fija VAR_THRESHOLD=0.80 pero N_MAX_COMPONENTES=8,
por lo que en la practica nunca se alcanza el 80% declarado (se retiene ~72% con
K=8). Este script recalcula el pipeline completo con K=11 (el numero real que
alcanza >=80%) sobre la variante ponderada, y compara la particion resultante
contra la solucion oficial K=8.

Salidas (reports/tables/):
  - v5_comparacion_k8_vs_k11.csv
  - v5_contingencia_k8_k11.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS, silhouette_referencia
from src.utils.io import PROCESSED_DIR, TABLES_DIR

RANDOM_STATE = 42
K_PCA_NUEVO = 11

print("=" * 80)
print("SENSIBILIDAD AL NUMERO DE COMPONENTES PCA: K=8 (oficial) vs K=11 (80% real)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar vectores de casuistica (variante ponderada, ya seleccionada) y
#    matriz institucional / particion oficial (K=8)
# ---------------------------------------------------------------------------
print("\n[1/5] Cargando vectores de casuistica y matriz institucional oficial (K=8)...")
v_caps = pd.read_parquet(PROCESSED_DIR / "casuistica_capitulos.parquet")
v_secs = pd.read_parquet(PROCESSED_DIR / "casuistica_procedimientos.parquet")
v_top20 = pd.read_parquet(PROCESSED_DIR / "casuistica_top20_grds.parquet")
v_cap_ponderado = v_caps[v_caps["variante"] == "ponderado"].drop(columns="variante").reset_index(drop=True)

cas_ponderado = (
    v_cap_ponderado.merge(v_secs, on="COD_HOSPITAL", how="inner", suffixes=("", "_sec"))
    .merge(v_top20, on="COD_HOSPITAL", how="inner", suffixes=("", "_top"))
)

matriz_k8 = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz_k8["COD_HOSPITAL"] = matriz_k8["COD_HOSPITAL"].astype(str)

asig_oficial = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig_oficial["COD_HOSPITAL"] = asig_oficial["COD_HOSPITAL"].astype(str)
labels_oficial = asig_oficial.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int)

# ---------------------------------------------------------------------------
# 2. PCA con K=11 sobre la variante ponderada
# ---------------------------------------------------------------------------
print(f"\n[2/5] Ejecutando PCA con K={K_PCA_NUEVO} componentes sobre variante ponderada...")
ids_cas = cas_ponderado["COD_HOSPITAL"].values
X_cas = cas_ponderado.drop(columns=["COD_HOSPITAL"]).values
scaler_cas = StandardScaler()
X_cas_scaled = scaler_cas.fit_transform(X_cas)

pca11 = PCA(n_components=K_PCA_NUEVO, random_state=RANDOM_STATE)
scores11 = pca11.fit_transform(X_cas_scaled)
var_acum_11 = float(np.cumsum(pca11.explained_variance_ratio_)[-1])

# La varianza de la solución oficial se recalcula sobre los mismos vectores de
# casuística: fijarla como literal la deja obsoleta al recomputar el pipeline.
pca8 = PCA(n_components=8, random_state=RANDOM_STATE).fit(X_cas_scaled)
var_acum_8 = float(np.cumsum(pca8.explained_variance_ratio_)[-1])

print(f"  Varianza acumulada con K={K_PCA_NUEVO}: {var_acum_11:.4f}")
print(f"  Varianza acumulada oficial con K=8: {var_acum_8:.4f}")

cols_11 = [f"dim_{i+1:02d}" for i in range(K_PCA_NUEVO)]
componentes_11 = pd.DataFrame(scores11, columns=cols_11)
componentes_11.insert(0, "COD_HOSPITAL", ids_cas)

# ---------------------------------------------------------------------------
# 3. Reconstruir matriz institucional con K=11 (mismas variables directas,
#    componentes de casuistica ampliadas)
# ---------------------------------------------------------------------------
print("\n[3/5] Ensamblando matriz institucional con K=11 componentes de casuistica...")
cols_dim8 = [f"dim_{i+1:02d}" for i in range(8)]
cols_directas = [c for c in matriz_k8.columns if c not in cols_dim8 and c != "COD_HOSPITAL"]

matriz_k11 = matriz_k8[["COD_HOSPITAL"] + cols_directas].merge(
    componentes_11, on="COD_HOSPITAL", how="inner", validate="1:1",
)
print(f"  Matriz K=8 (oficial): {matriz_k8.shape} | Matriz K=11: {matriz_k11.shape}")


def preprocesar(matriz: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats = matriz.drop(columns=[c for c in COLS_DROP_CLUSTERING if c in matriz.columns]).copy()
    ids = matriz["COD_HOSPITAL"].astype(str).tolist()
    feats.index = ids
    feats = feats.fillna(0.0)
    for col in LOG1P_COLS:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    X = RobustScaler().fit_transform(feats.values)
    return X, ids


X_k11, ids_k11 = preprocesar(matriz_k11)
Z_k11 = linkage(X_k11, method="ward")
labels_k11_arr = fcluster(Z_k11, t=4, criterion="maxclust") - 1
sil_k11 = silhouette_score(X_k11, labels_k11_arr)
ch_k11 = calinski_harabasz_score(X_k11, labels_k11_arr)
db_k11 = davies_bouldin_score(X_k11, labels_k11_arr)
sizes_k11 = pd.Series(labels_k11_arr).value_counts().sort_index().tolist()
print(f"  Ward K=4 (con PCA K={K_PCA_NUEVO}): Silhouette={sil_k11:.3f} | "
      f"CH={ch_k11:.2f} | DB={db_k11:.3f} | tamanos={sizes_k11}")

# ---------------------------------------------------------------------------
# 4. Comparar particion K=11 vs particion oficial K=8
# ---------------------------------------------------------------------------
print("\n[4/5] Comparando particion K=11 vs particion oficial K=8...")

comparacion = pd.DataFrame({
    "COD_HOSPITAL": ids_k11,
    "cluster_oficial_k8": [int(labels_oficial.get(h, -1)) for h in ids_k11],
    "cluster_k11": labels_k11_arr,
})
mask_validos = comparacion["cluster_oficial_k8"] != -1
ari = ari_score(
    comparacion.loc[mask_validos, "cluster_oficial_k8"],
    comparacion.loc[mask_validos, "cluster_k11"],
)
nmi = normalized_mutual_info_score(
    comparacion.loc[mask_validos, "cluster_oficial_k8"],
    comparacion.loc[mask_validos, "cluster_k11"],
)
print(f"  ARI (K=8 oficial vs K=11): {ari:.3f}")
print(f"  NMI (K=8 oficial vs K=11): {nmi:.3f}")

contingencia = pd.crosstab(
    comparacion["cluster_oficial_k8"], comparacion["cluster_k11"],
    rownames=["oficial_K8"], colnames=["K11"],
)
print("\n  Tabla de contingencia (oficial_K8 x K11):")
print(contingencia.to_string())

n_cambian = int((comparacion["cluster_oficial_k8"] != comparacion["cluster_k11"]).sum())

# ---------------------------------------------------------------------------
# 5. Barrido adicional de K entre 8 y 11 para contextualizar la sensibilidad
#    de las metricas de validacion interna al numero de componentes
# ---------------------------------------------------------------------------
print("\n[5/5] Barrido adicional K_pca in [8, 9, 10, 11] para contextualizar métricas...")
filas_barrido = []
for k_pca in [8, 9, 10, 11]:
    if k_pca == 8:
        X_bar, ids_bar = preprocesar(matriz_k8)
    else:
        pca_k = PCA(n_components=k_pca, random_state=RANDOM_STATE)
        scores_k = pca_k.fit_transform(X_cas_scaled)
        var_k = float(np.cumsum(pca_k.explained_variance_ratio_)[-1])
        cols_k = [f"dim_{i+1:02d}" for i in range(k_pca)]
        comp_k = pd.DataFrame(scores_k, columns=cols_k)
        comp_k.insert(0, "COD_HOSPITAL", ids_cas)
        matriz_k = matriz_k8[["COD_HOSPITAL"] + cols_directas].merge(
            comp_k, on="COD_HOSPITAL", how="inner", validate="1:1",
        )
        X_bar, ids_bar = preprocesar(matriz_k)
    Z_bar = linkage(X_bar, method="ward")
    labels_bar = fcluster(Z_bar, t=4, criterion="maxclust") - 1
    sil_bar = silhouette_score(X_bar, labels_bar)
    ch_bar = calinski_harabasz_score(X_bar, labels_bar)
    db_bar = davies_bouldin_score(X_bar, labels_bar)
    var_bar = var_acum_8 if k_pca == 8 else var_k
    sizes_bar = pd.Series(labels_bar).value_counts().sort_index().tolist()
    filas_barrido.append({
        "K_pca": k_pca, "varianza_acumulada": var_bar,
        "silhouette": sil_bar, "calinski_harabasz": ch_bar, "davies_bouldin": db_bar,
        "tamanos": str(sizes_bar),
    })
    print(f"  K_pca={k_pca}: var_acum={var_bar:.3f} | Sil={sil_bar:.3f} | "
          f"CH={ch_bar:.2f} | DB={db_bar:.3f} | tamanos={sizes_bar}")

df_barrido = pd.DataFrame(filas_barrido)
df_barrido.to_csv(TABLES_DIR / "sensibilidad_pca_barrido_componentes.csv", index=False)

resumen = pd.DataFrame([{
    "ari_k8_vs_k11": ari, "nmi_k8_vs_k11": nmi,
    "silhouette_k8": silhouette_referencia(), "silhouette_k11": sil_k11,
    "varianza_acumulada_k8": var_acum_8, "varianza_acumulada_k11": var_acum_11,
    "n_hospitales_cambian": n_cambian,
    "tamanos_k11": str(sizes_k11),
}])
resumen.to_csv(TABLES_DIR / "sensibilidad_pca_k8_vs_k11.csv", index=False)
contingencia.to_csv(TABLES_DIR / "sensibilidad_pca_contingencia.csv")
comparacion.to_csv(TABLES_DIR / "sensibilidad_pca_particiones.csv", index=False)

print("\n" + "=" * 80)
print(f"RESUMEN FINAL — K=8 (oficial, {var_acum_8:.1%} var) vs K=11 ({var_acum_11:.1%} var)")
print("=" * 80)
print(f"  ARI: {ari:.3f} | NMI: {nmi:.3f}")
print(f"  Silhouette: {sil_k11:.3f} (K=11) vs {silhouette_referencia():.3f} (K=8)")
print(f"  Varianza acumulada: {var_acum_11:.3f} (K=11) vs {var_acum_8:.3f} (K=8)")
print(f"  Hospitales que cambian de cluster: {n_cambian} de 65")
print("\nAnalisis completo. Tablas persistidas en reports/tables/v5_*.csv")
