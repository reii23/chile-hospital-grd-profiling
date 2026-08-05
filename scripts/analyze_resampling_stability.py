"""Validacion de estabilidad mediante submuestreo (Observacion 15 de
correcciones.txt).

Problema: la metodologia (Seccion sec:bootstrap del Capitulo 3) prometia un
submuestreo de 100 iteraciones evaluando el coeficiente de Silhouette, pero
nunca se ejecuto ni se reporto en resultados. Ademas, recalcular solo
Silhouette en cada submuestra no evalua la estabilidad de la PERTENENCIA a
los grupos: dos submuestras pueden tener Silhouette similar y asignaciones de
hospitales completamente distintas.

Este script implementa y reporta la metodologia completa, agregando las
metricas de estabilidad de asignacion que faltaban:

  1. Submuestreo sin reposicion: 100 iteraciones, 80% de los 65 hospitales
     (n_sub=52), clustering aglomerativo Ward K=4 sobre cada submuestra.
  2. Silhouette de cada submuestra (lo que la metodologia ya prometia) e
     intervalo de confianza empirico al 95% (percentiles 2.5-97.5).
  3. ARI y NMI entre la particion de cada submuestra y la particion original,
     restringida a los hospitales en comun (evalua si la PARTICION es
     estable, no solo su cohesion).
  4. Matriz de coasignacion: para cada par de hospitales, la proporcion de
     submuestras en que ambos aparecen juntos (donde ambos fueron
     muestreados) en las que quedan en el MISMO cluster. Resume si la
     estructura de grupos es consistente hospital por hospital.
  5. Estabilidad individual por hospital: a partir de la matriz de
     coasignacion, se define la estabilidad de un hospital como su
     coasignacion promedio con los demas miembros de SU PROPIO cluster
     original (proporcion de "vecinos" que conserva).
  6. Estabilidad por grupo (C0/C1/C2/C3): promedio de la estabilidad
     individual de los hospitales de cada grupo, y Jaccard por cluster
     entre la particion original y cada submuestra (promediado sobre las
     100 iteraciones, solo cuando el cluster original tiene representantes
     en la submuestra).

Salidas (reports/tables/):
  - estabilidad_submuestreo.csv         (Silhouette y ARI/NMI por iteracion)
  - estabilidad_coasignacion.csv            (65 x 65, proporcion de coasignacion)
  - estabilidad_individual.csv         (estabilidad por hospital)
  - estabilidad_por_grupo.csv          (estabilidad y Jaccard promedio por cluster)
  - estabilidad_remuestreo.txt            (resumen textual con IC 95%)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS

RANDOM_STATE = 42
N_ITERACIONES = int(os.environ.get("N_REP", "100"))
FRAC_SUBMUESTRA = 0.80

rng = np.random.default_rng(RANDOM_STATE)

print("=" * 80)
print("VALIDACION DE ESTABILIDAD MEDIANTE SUBMUESTREO (Observacion 15)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar matriz y particion original (Ward K=4, Nivel 1)
# ---------------------------------------------------------------------------
print("\n[1/6] Cargando matriz institucional y particion original...")
matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
ids_original = matriz["COD_HOSPITAL"].tolist()
n_total = len(ids_original)
n_sub = int(round(n_total * FRAC_SUBMUESTRA))
print(f"  n total = {n_total}, n submuestra (80%) = {n_sub}")

asig_original = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig_original["COD_HOSPITAL"] = asig_original["COD_HOSPITAL"].astype(str)
labels_original = asig_original.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int)
labels_original = labels_original.reindex(ids_original)
clusters_originales = sorted(labels_original.unique())
print(f"  Clusters originales: {clusters_originales}, tamanos: "
      f"{labels_original.value_counts().sort_index().to_dict()}")

feats_clu_cols = [c for c in matriz.columns if c not in COLS_DROP_CLUSTERING]


def preprocesar(matriz_sub: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats = matriz_sub[feats_clu_cols].copy()
    ids = matriz_sub["COD_HOSPITAL"].astype(str).tolist()
    feats.index = ids
    feats = feats.fillna(0.0)
    for col in LOG1P_COLS:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    X = RobustScaler().fit_transform(feats.values)
    return X, ids


# Silhouette y particion de referencia (poblacion completa) para contexto
X_full, ids_full = preprocesar(matriz)
sil_full = silhouette_score(X_full, labels_original.reindex(ids_full).values)
print(f"  Silhouette con los 65 hospitales completos (referencia): {sil_full:.4f}")

# ---------------------------------------------------------------------------
# 2. Submuestreo: 100 iteraciones, 80% sin reposicion, Ward K=4
# ---------------------------------------------------------------------------
print(f"\n[2/6] Ejecutando {N_ITERACIONES} iteraciones de submuestreo (80% sin "
      f"reposicion) + Ward K=4 + ARI/NMI vs. particion original...")

filas_sil = []
# Acumuladores para matriz de coasignacion: (n_veces_juntos, n_veces_mismo_cluster)
coasig_juntos = np.zeros((n_total, n_total), dtype=np.int64)
coasig_mismo = np.zeros((n_total, n_total), dtype=np.int64)
idx_map = {h: i for i, h in enumerate(ids_original)}

# Acumulador de Jaccard por cluster original (solo cuando hay representantes)
jaccard_acumulado = {c: [] for c in clusters_originales}

for rep in range(N_ITERACIONES):
    idx_sub = rng.choice(n_total, size=n_sub, replace=False)
    ids_sub = [ids_original[i] for i in idx_sub]
    matriz_sub = matriz[matriz["COD_HOSPITAL"].isin(ids_sub)].copy()

    X_sub, ids_sub_ord = preprocesar(matriz_sub)
    Z_sub = linkage(X_sub, method="ward")
    labels_sub_arr = fcluster(Z_sub, t=4, criterion="maxclust")
    sil_sub = silhouette_score(X_sub, labels_sub_arr)
    labels_sub = pd.Series(labels_sub_arr, index=ids_sub_ord)

    # ARI / NMI vs particion original, restringido a hospitales en comun
    orig_sub = labels_original.reindex(ids_sub_ord)
    ari = adjusted_rand_score(orig_sub.values, labels_sub.values)
    nmi = normalized_mutual_info_score(orig_sub.values, labels_sub.values)

    filas_sil.append({
        "rep": rep, "n_sub": len(ids_sub_ord), "silhouette": sil_sub,
        "ari_vs_original": ari, "nmi_vs_original": nmi,
    })

    # --- Matriz de coasignacion ---
    for i, h1 in enumerate(ids_sub_ord):
        gi = idx_map[h1]
        for j in range(i + 1, len(ids_sub_ord)):
            h2 = ids_sub_ord[j]
            gj = idx_map[h2]
            coasig_juntos[gi, gj] += 1
            coasig_juntos[gj, gi] += 1
            if labels_sub_arr[i] == labels_sub_arr[j]:
                coasig_mismo[gi, gj] += 1
                coasig_mismo[gj, gi] += 1

    # --- Jaccard por cluster original: compara el conjunto de hospitales
    #     del cluster c en la particion ORIGINAL, restringido a los presentes
    #     en esta submuestra, contra el cluster de la submuestra con el que
    #     tiene mayor solapamiento (mejor emparejamiento) ---
    for c in clusters_originales:
        miembros_c_en_sub = set(
            h for h in ids_sub_ord if labels_original[h] == c
        )
        if len(miembros_c_en_sub) == 0:
            continue
        mejor_jaccard = 0.0
        for c_sub in np.unique(labels_sub_arr):
            miembros_c_sub = set(
                ids_sub_ord[k] for k in range(len(ids_sub_ord)) if labels_sub_arr[k] == c_sub
            )
            inter = len(miembros_c_en_sub & miembros_c_sub)
            union = len(miembros_c_en_sub | miembros_c_sub)
            jac = inter / union if union > 0 else 0.0
            mejor_jaccard = max(mejor_jaccard, jac)
        jaccard_acumulado[c].append(mejor_jaccard)

    if (rep + 1) % 20 == 0:
        print(f"  ... {rep + 1}/{N_ITERACIONES} iteraciones completadas")

df_sil = pd.DataFrame(filas_sil)
df_sil.to_csv(TABLES_DIR / "estabilidad_submuestreo.csv", index=False)
print(f"  Persistido: estabilidad_submuestreo.csv")

# ---------------------------------------------------------------------------
# 3. Intervalos de confianza empiricos (percentiles 2.5-97.5)
# ---------------------------------------------------------------------------
print("\n[3/6] Calculando intervalos de confianza empiricos (percentiles 2.5-97.5)...")
ic_sil = df_sil["silhouette"].quantile([0.025, 0.5, 0.975])
ic_ari = df_sil["ari_vs_original"].quantile([0.025, 0.5, 0.975])
ic_nmi = df_sil["nmi_vs_original"].quantile([0.025, 0.5, 0.975])
print(f"  Silhouette: mediana={ic_sil[0.5]:.4f}, IC95%=[{ic_sil[0.025]:.4f}, {ic_sil[0.975]:.4f}]")
print(f"  ARI vs. original: mediana={ic_ari[0.5]:.4f}, IC95%=[{ic_ari[0.025]:.4f}, {ic_ari[0.975]:.4f}]")
print(f"  NMI vs. original: mediana={ic_nmi[0.5]:.4f}, IC95%=[{ic_nmi[0.025]:.4f}, {ic_nmi[0.975]:.4f}]")

# ---------------------------------------------------------------------------
# 4. Matriz de coasignacion final (proporcion, no conteo)
# ---------------------------------------------------------------------------
print("\n[4/6] Construyendo matriz de coasignacion (proporcion sobre veces muestreados juntos)...")
with np.errstate(invalid="ignore", divide="ignore"):
    prop_coasig = np.where(coasig_juntos > 0, coasig_mismo / coasig_juntos, np.nan)
df_coasig = pd.DataFrame(prop_coasig, index=ids_original, columns=ids_original)
df_coasig.to_csv(TABLES_DIR / "estabilidad_coasignacion.csv")
print(f"  Persistido: estabilidad_coasignacion.csv ({df_coasig.shape})")

# ---------------------------------------------------------------------------
# 5. Estabilidad individual: coasignacion promedio con los "vecinos" del
#    cluster original de cada hospital
# ---------------------------------------------------------------------------
print("\n[5/6] Calculando estabilidad individual por hospital...")
filas_estab = []
for h in ids_original:
    i = idx_map[h]
    c_h = labels_original[h]
    vecinos = [idx_map[h2] for h2 in ids_original if h2 != h and labels_original[h2] == c_h]
    if len(vecinos) == 0:
        estab = np.nan
    else:
        vals = prop_coasig[i, vecinos]
        vals = vals[~np.isnan(vals)]
        estab = float(np.mean(vals)) if len(vals) > 0 else np.nan
    filas_estab.append({
        "COD_HOSPITAL": h, "cluster_original": c_h,
        "n_vecinos_cluster": len(vecinos), "estabilidad_individual": estab,
    })
df_estab = pd.DataFrame(filas_estab)
df_estab.to_csv(TABLES_DIR / "estabilidad_individual.csv", index=False)
print(f"  Persistido: estabilidad_individual.csv")
print(f"  Hospitales con menor estabilidad individual (top 10 mas inestables):")
print(df_estab.sort_values("estabilidad_individual").head(10).to_string(index=False, float_format="%.3f"))

# ---------------------------------------------------------------------------
# 6. Estabilidad y Jaccard por grupo (C0/C1/C2/C3)
# ---------------------------------------------------------------------------
print("\n[6/6] Resumiendo estabilidad y Jaccard promedio por grupo...")
filas_grupo = []
for c in clusters_originales:
    sub = df_estab[df_estab["cluster_original"] == c]
    jac_vals = jaccard_acumulado[c]
    filas_grupo.append({
        "cluster": c,
        "n_hospitales": len(sub),
        "estabilidad_individual_media": float(sub["estabilidad_individual"].mean()),
        "estabilidad_individual_mediana": float(sub["estabilidad_individual"].median()),
        "jaccard_promedio": float(np.mean(jac_vals)) if jac_vals else np.nan,
        "jaccard_mediana": float(np.median(jac_vals)) if jac_vals else np.nan,
        "n_submuestras_con_representantes": len(jac_vals),
    })
df_grupo = pd.DataFrame(filas_grupo)
df_grupo.to_csv(TABLES_DIR / "estabilidad_por_grupo.csv", index=False)
print(df_grupo.to_string(index=False, float_format="%.3f"))
print(f"  Persistido: estabilidad_por_grupo.csv")

# ---------------------------------------------------------------------------
# 7. Resumen textual
# ---------------------------------------------------------------------------
with open(TABLES_DIR / "estabilidad_remuestreo.txt", "w") as f:
    f.write(f"N_ITERACIONES={N_ITERACIONES}\n")
    f.write(f"FRAC_SUBMUESTRA={FRAC_SUBMUESTRA}\n")
    f.write(f"n_sub={n_sub}\n")
    f.write(f"silhouette_mediana={ic_sil[0.5]:.4f}\n")
    f.write(f"silhouette_ic95_lo={ic_sil[0.025]:.4f}\n")
    f.write(f"silhouette_ic95_hi={ic_sil[0.975]:.4f}\n")
    f.write(f"ari_mediana={ic_ari[0.5]:.4f}\n")
    f.write(f"ari_ic95_lo={ic_ari[0.025]:.4f}\n")
    f.write(f"ari_ic95_hi={ic_ari[0.975]:.4f}\n")
    f.write(f"nmi_mediana={ic_nmi[0.5]:.4f}\n")
    f.write(f"nmi_ic95_lo={ic_nmi[0.025]:.4f}\n")
    f.write(f"nmi_ic95_hi={ic_nmi[0.975]:.4f}\n")

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"  Silhouette IC95%: [{ic_sil[0.025]:.4f}, {ic_sil[0.975]:.4f}] (mediana {ic_sil[0.5]:.4f})")
print(f"  ARI vs original IC95%: [{ic_ari[0.025]:.4f}, {ic_ari[0.975]:.4f}] (mediana {ic_ari[0.5]:.4f})")
print(f"  NMI vs original IC95%: [{ic_nmi[0.025]:.4f}, {ic_nmi[0.975]:.4f}] (mediana {ic_nmi[0.5]:.4f})")
print(f"  Estabilidad y Jaccard por grupo:\n{df_grupo.to_string(index=False, float_format='%.3f')}")
print("\nAnalisis completo.")
