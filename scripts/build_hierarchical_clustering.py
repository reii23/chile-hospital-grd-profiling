"""Búsqueda y construcción del agrupamiento jerárquico final.

Evalúa Ward, K-means y GMM para distintos valores de K; luego construye la
partición Ward de Nivel 1 y el subclustering exploratorio del grupo generalista.

Salidas principales:
- `reports/tables/busqueda_balanceada.csv`
- `reports/tables/busqueda_subclustering.csv`
- `reports/tables/asignacion_jerarquica_final.csv`
- `reports/figures/busqueda_k_balanceado.png`
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING as COLS_DROP, LOG1P_COLS

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
RANDOM_STATE = 42
REPORTED_K_SUB = 9

# ---------------------------------------------------------------------------
# 1. Cargar matriz final y escalar
# ---------------------------------------------------------------------------
matriz = pd.read_parquet(PROCESSED / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)

features_clu = matriz.drop(columns=[c for c in COLS_DROP if c in matriz.columns]).copy()
features_clu.index = matriz["COD_HOSPITAL"].values
features_clu = features_clu.fillna(0.0)

for col in LOG1P_COLS:
    if col in features_clu.columns:
        features_clu[col] = np.log1p(features_clu[col].clip(lower=0))

X = RobustScaler().fit_transform(features_clu.values)
ids = features_clu.index.tolist()
n_total = len(ids)


# ---------------------------------------------------------------------------
# 2. Funcion auxiliar: contar clusters por tamanio
# ---------------------------------------------------------------------------
def perfil_tamanios(labels: np.ndarray) -> dict:
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
    return {
        "K_efectivo": len(sizes),
        "n_singletons": sum(1 for s in sizes if s == 1),
        "n_pares": sum(1 for s in sizes if s == 2),
        "n_triadas": sum(1 for s in sizes if s == 3),
        "n_clusters_n4plus": sum(1 for s in sizes if s >= 4),
        "n_clusters_n5plus": sum(1 for s in sizes if s >= 5),
        "mainstream_size": sizes[0] if sizes else 0,
        "tamanos": str(sizes),
    }


# ---------------------------------------------------------------------------
# 3. Barrido principal: directo sobre los 65
# ---------------------------------------------------------------------------
print("=" * 80)
print("BUSQUEDA DE CONFIGURACIONES BALANCEADAS")
print("Criterio: max clusters n>=4, min singletons/pares")
print("=" * 80)

K_GRID = list(range(2, 16))
filas = []
asignaciones = {}

# K-means
print("\n[1/3] K-means K = 2..15")
for k in K_GRID:
    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(X)
    labels = km.labels_
    perfil = perfil_tamanios(labels)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    filas.append({"metodo": "K-means", "K": k, "silhouette": sil, "davies": db, **perfil})
    asignaciones[("K-means", k)] = labels
    print(f"  K={k}: Sil={sil:.3f} | n>=4: {perfil['n_clusters_n4plus']} | "
          f"singletons: {perfil['n_singletons']} | pares: {perfil['n_pares']} | "
          f"main: {perfil['mainstream_size']}")

# Aglomerativo Ward
print("\n[2/3] Aglomerativo Ward K = 2..15")
Z = linkage(X, method="ward")
for k in K_GRID:
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    perfil = perfil_tamanios(labels)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    filas.append({"metodo": "Aglomerativo", "K": k, "silhouette": sil, "davies": db, **perfil})
    asignaciones[("Aglomerativo", k)] = labels
    print(f"  K={k}: Sil={sil:.3f} | n>=4: {perfil['n_clusters_n4plus']} | "
          f"singletons: {perfil['n_singletons']} | pares: {perfil['n_pares']} | "
          f"main: {perfil['mainstream_size']}")

# GMM
print("\n[3/3] GMM K = 2..15")
for k in K_GRID:
    try:
        gmm = GaussianMixture(
            n_components=k, n_init=10, random_state=RANDOM_STATE, max_iter=200,
        ).fit(X)
        labels = gmm.predict(X)
        perfil = perfil_tamanios(labels)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        filas.append({"metodo": "GMM", "K": k, "silhouette": sil, "davies": db,
                      "bic": gmm.bic(X), **perfil})
        asignaciones[("GMM", k)] = labels
        print(f"  K={k}: Sil={sil:.3f} | n>=4: {perfil['n_clusters_n4plus']} | "
              f"singletons: {perfil['n_singletons']} | pares: {perfil['n_pares']} | "
              f"main: {perfil['mainstream_size']}")
    except Exception as exc:
        print(f"  K={k}: error {exc}")


df_res = pd.DataFrame(filas)
df_res.to_csv(TABLES / "busqueda_balanceada.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Encontrar la mejor configuracion
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("MEJORES CONFIGURACIONES (criterio: max n_clusters_n4plus, sin singletons)")
print("=" * 80)

# Score combinado: priorizar n>=4, penalizar singletons
df_res["score"] = (
    df_res["n_clusters_n4plus"] * 10
    - df_res["n_singletons"] * 3
    - df_res["n_pares"] * 1
)

# Top 10 configuraciones por score (con Silhouette > 0.10)
top = df_res[df_res["silhouette"] > 0.10].sort_values(
    ["score", "silhouette"], ascending=[False, False]
).head(10)

print()
cols_show = [
    "metodo", "K", "silhouette", "davies",
    "n_clusters_n4plus", "n_clusters_n5plus",
    "n_singletons", "n_pares", "mainstream_size", "score",
]
print(top[cols_show].to_string(index=False, float_format="%.3f"))


# ---------------------------------------------------------------------------
# 5. Subclustering jerarquico balanceado: K=4 principal + sub del mainstream
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("OPCION JERARQUICA: K=4 principal + subclustering del mainstream")
print("Buscamos K_sub que mantenga clusters mainstream con n>=4")
print("=" * 80)

# Tomar K=4 del nivel 1
labels_K4 = asignaciones[("Aglomerativo", 4)]
sizes_K4 = pd.Series(labels_K4).value_counts()
CLUSTER_MAIN = int(sizes_K4.idxmax())
mask_main = labels_K4 == CLUSTER_MAIN
ids_main = [ids[i] for i in range(n_total) if mask_main[i]]
X_main = X[mask_main]
n_main = len(ids_main)

print(f"\nMainstream: C{CLUSTER_MAIN}, n={n_main}")
Z_main = linkage(X_main, method="ward")

filas_sub = []
asignaciones_sub = {}
for k_sub in range(2, 11):
    labels_sub = fcluster(Z_main, t=k_sub, criterion="maxclust") - 1
    perfil = perfil_tamanios(labels_sub)
    if len(set(labels_sub)) >= 2:
        sil_sub = silhouette_score(X_main, labels_sub)
        db_sub = davies_bouldin_score(X_main, labels_sub)
    else:
        sil_sub = db_sub = np.nan
    filas_sub.append({
        "K_sub": k_sub, "silhouette_sub": sil_sub, "davies_sub": db_sub,
        **{f"sub_{k}": v for k, v in perfil.items()},
    })
    asignaciones_sub[k_sub] = labels_sub
    print(f"  K_sub={k_sub}: Sil={sil_sub:.3f} | n>=4: {perfil['n_clusters_n4plus']} | "
          f"singletons: {perfil['n_singletons']} | pares: {perfil['n_pares']} | "
          f"tamanos={perfil['tamanos']}")

df_sub = pd.DataFrame(filas_sub)
df_sub.to_csv(TABLES / "busqueda_subclustering.csv", index=False)


# La solución suplementaria se fija explícitamente en K_sub=9: maximiza el
# criterio de balance (más grupos con n>=4 y sin pares) dentro del barrido.
df_sub["sub_score"] = (
    df_sub["sub_n_clusters_n4plus"] * 10
    - df_sub["sub_n_singletons"] * 3
    - df_sub["sub_n_pares"] * 1
)
mejor_K_sub = REPORTED_K_SUB
if mejor_K_sub not in asignaciones_sub:
    raise ValueError(f"K_sub reportado no disponible: {mejor_K_sub}")
print(f"\n>>> K_sub suplementario reportado: {mejor_K_sub}")
print(f"    Sil={df_sub.loc[df_sub['K_sub']==mejor_K_sub, 'silhouette_sub'].iloc[0]:.3f}")
print(f"    Tamanos={df_sub.loc[df_sub['K_sub']==mejor_K_sub, 'sub_tamanos'].iloc[0]}")


# ---------------------------------------------------------------------------
# 6. Reportar composicion del K_sub mejor
# ---------------------------------------------------------------------------
labels_sub_best = asignaciones_sub[mejor_K_sub]

# Cargar nombres
df_maestra = pd.read_excel(
    ROOT / "insumos" / "maestras" / "Tablas maestras bases GRD.xlsx",
    sheet_name="Hospitales", header=None, skiprows=1,
    names=["COD_HOSPITAL", "NOMBRE"],
)
df_maestra["COD_HOSPITAL"] = df_maestra["COD_HOSPITAL"].astype(str).str.strip()
nombres = dict(zip(df_maestra["COD_HOSPITAL"], df_maestra["NOMBRE"]))

print()
print("-" * 80)
print(f"COMPOSICION del subclustering del mainstream con K_sub={mejor_K_sub}")
print("-" * 80)

# Perfiles
matriz_main = matriz.set_index("COD_HOSPITAL").loc[ids_main].copy()
matriz_main["sub"] = labels_sub_best

cols_perfil = [
    "egresos_por_anio", "peso_medio_grd", "estancia_media", "severidad_media",
    "tasa_cma", "pct_pediatrico", "pct_geriatrico", "pct_urgencia",
    "pct_programada", "tasa_partos", "pct_uso_pabellon", "cv_estancia",
]
perfiles = matriz_main.groupby("sub")[cols_perfil].mean()
perfiles["n"] = matriz_main.groupby("sub").size()
perfiles.to_csv(TABLES / "perfiles_subclustering.csv")

print()
print("Tamanios y caracteristicas medias por subcluster:")
print(perfiles[["n"] + cols_perfil[:6]].T.to_string(float_format="%.3f"))
print()

# Listado por subcluster
for sc in sorted(set(labels_sub_best)):
    miembros = [ids_main[i] for i in range(len(ids_main)) if labels_sub_best[i] == sc]
    print(f"\nSubcluster S{sc} (n={len(miembros)}):")
    for cod in miembros[:10]:
        print(f"  {cod}  {str(nombres.get(cod, '?'))[:55]}")
    if len(miembros) > 10:
        print(f"  ... y {len(miembros) - 10} mas")


# ---------------------------------------------------------------------------
# 7. Persistir asignacion jerarquica final balanceada
# ---------------------------------------------------------------------------
asig_final = []
for i, cod in enumerate(ids):
    if labels_K4[i] == CLUSTER_MAIN:
        sub_idx = ids_main.index(cod)
        sub_label = int(labels_sub_best[sub_idx])
        nivel1 = "MAINSTREAM"
        nivel2 = f"S{sub_label}"
    else:
        nivel1 = f"C{int(labels_K4[i])}"
        nivel2 = nivel1
    asig_final.append({
        "COD_HOSPITAL": cod,
        "nivel1_K4": int(labels_K4[i]),
        "nivel1_etiqueta": nivel1,
        "nivel2_etiqueta": nivel2,
    })
df_asig = pd.DataFrame(asig_final)
df_asig.to_csv(TABLES / "asignacion_jerarquica_final.csv", index=False)
print(f"\n>>> Persistido: asignacion_jerarquica_final.csv")

# Listado nominal por cluster del Nivel 1. Se persiste aqui porque este script
# es el unico punto del pipeline que fija la particion y dispone de la maestra
# de nombres; los analisis posteriores lo consumen para rotular hospitales.
df_nombres_cluster = (
    pd.DataFrame({
        "cluster": [int(label) for label in labels_K4],
        "COD_HOSPITAL": ids,
        "NOMBRE": [nombres.get(cod, "?") for cod in ids],
    })
    .sort_values(["cluster", "COD_HOSPITAL"], kind="stable")
    .reset_index(drop=True)
)
df_nombres_cluster.to_csv(TABLES / "hospitales_por_cluster_nivel1.csv", index=False)
print(f">>> Persistido: hospitales_por_cluster_nivel1.csv")


# ---------------------------------------------------------------------------
# 8. Resumen final: distribucion en NIVEL 1 + NIVEL 2
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("RESUMEN — ASIGNACION JERARQUICA FINAL (criterio balanceado)")
print("=" * 80)

dist_nivel2 = df_asig["nivel2_etiqueta"].value_counts().sort_index()
print(f"\nClusters totales (nivel 2): {len(dist_nivel2)}")
print()
for lbl, n in dist_nivel2.items():
    rep = "<--- representativo" if n >= 4 else "<--- pequenio (n<4)"
    print(f"  {lbl:<15} n={n}  {rep}")

print(f"\nClusters n>=4: {(dist_nivel2 >= 4).sum()}")
print(f"Clusters n<4 : {(dist_nivel2 < 4).sum()}")
print(f"Singletons   : {(dist_nivel2 == 1).sum()}")
print(f"Pares (n=2)  : {(dist_nivel2 == 2).sum()}")


# ---------------------------------------------------------------------------
# 9. Figuras
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax_left, ax_right = axes

# Figura 1: silhouette por K (universo completo)
for metodo, color in [("K-means", "C0"), ("Aglomerativo", "C1"), ("GMM", "C2")]:
    sub = df_res[df_res["metodo"] == metodo].sort_values("K")
    ax_left.plot(sub["K"], sub["silhouette"], "o-", label=metodo, color=color, linewidth=2, markersize=7)
ax_left.axhline(0.20, color="gray", ls="--", alpha=0.5, label="Umbral 0.20")
ax_left.set_xlabel("K")
ax_left.set_ylabel("Silhouette")
ax_left.set_title("Silhouette por K (universo completo, matriz final)")
ax_left.legend()
ax_left.grid(True, alpha=0.3)

# Figura 2: clusters con n>=4 por K
for metodo, color in [("K-means", "C0"), ("Aglomerativo", "C1"), ("GMM", "C2")]:
    sub = df_res[df_res["metodo"] == metodo].sort_values("K")
    ax_right.plot(sub["K"], sub["n_clusters_n4plus"], "o-", label=f"{metodo} n>=4", color=color, linewidth=2, markersize=7)
    ax_right.plot(sub["K"], sub["n_singletons"], "s--", label=f"{metodo} singletons", color=color, alpha=0.5, linewidth=1.5)
ax_right.set_xlabel("K")
ax_right.set_ylabel("Cantidad")
ax_right.set_title("Clusters n>=4 vs singletons por K")
ax_right.legend(fontsize=8)
ax_right.grid(True, alpha=0.3)

plt.suptitle("Búsqueda de K balanceada sobre la matriz final", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES / "busqueda_k_balanceado.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura: busqueda_k_balanceado.png")
print("Finalizado.")
