# -*- coding: utf-8 -*-
"""Comparacion directa de homogeneidad interna: Ward vs. clasificacion MINSAL
(Observacion 17 de correcciones.txt).

Problema: el contraste previo (Seccion sec:contraste-minsal / sec:res-minsal)
solo reporta ARI y NMI, que cuantifican CONCORDANCIA entre particiones, no
HOMOGENEIDAD relativa. ARI/NMI bajos permiten decir que las particiones son
distintas, pero no que una sea mas homogenea, compacta o "mejor" que la otra.
La hipotesis de trabajo (Capitulo 1) afirma explicitamente que la segmentacion
funcional tiene "mayor homogeneidad interna... que la clasificacion
administrativa vigente", una afirmacion que el pipeline nunca comparo de forma
directa.

Ademas, MINSAL define solo 2 categorias en el universo de 65 hospitales (Alta
y Mediana Complejidad) mientras que Ward K=4 define 4 grupos. Comparar
metricas sensibles al numero de grupos (Silhouette, Calinski-Harabasz) sin
controlar esta diferencia favorece mecanicamente a la particion mas
fragmentada, como ya se demostro en el barrido de K de la Seccion sec:nivel1
(K=2 optimiza las tres metricas internas por encima de K=4).

Este script implementa la comparacion directa que faltaba, en dos niveles:

  A. Comparacion NO controlada por K (Ward K=4 tal como se usa en el trabajo,
     vs. MINSAL de 2 categorias), reportando:
       - Dispersion intragrupo normalizada (WSS/TSS): suma de cuadrados
         intragrupo dividida por la suma de cuadrados total, en la MISMA
         matriz X preprocesada usada para el clustering. Un valor menor
         indica mayor homogeneidad interna, y a diferencia de Silhouette,
         no favorece mecanicamente a mas grupos de la misma forma extrema.
       - MAD (mediana de la desviacion absoluta respecto a la mediana del
         grupo) e IQR por variable, agregados en un promedio normalizado
         sobre las 27 variables directas, para cada particion.
       - Silhouette de MINSAL calculado sobre el mismo X que Ward (etiquetas
         comparables, misma metrica de distancia).
       - Prueba de permutacion: se generan 2000 particiones aleatorias que
         preservan los tamanios de grupo de cada particion real (55/10 para
         MINSAL; 53/7/3/2 para Ward K=4) y se compara la dispersion
         normalizada observada contra esa distribucion nula, obteniendo un
         p-valor para "mas homogenea que el azar bajo el mismo tamanio de
         grupos" en cada particion por separado.

  B. Comparacion CONTROLADA por K=2: se recalcula Ward con K=2 (que ya
     aisla al grupo mas extremo de la red, sizes 62/3, Seccion sec:nivel1) y
     se compara directamente contra MINSAL (55/10), ambas con K=2, sobre las
     mismas variables y las mismas metricas de dispersion. Esto responde a la
     alternativa explicita de la observacion 17 ("una alternativa es comparar
     ... solucion Ward restringida a K=2 ... ambas sobre las mismas variables
     y metricas").

Salidas (reports/tables/):
  - homogeneidad_no_controlada.csv  (Ward K=4 vs MINSAL, metricas)
  - homogeneidad_controlada_k2.csv   (Ward K=2 vs MINSAL, metricas)
  - homogeneidad_por_variable.csv                  (MAD/IQR por variable y particion)
  - permutacion_homogeneidad.csv                (distribucion nula por particion)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS

RANDOM_STATE = 42
N_PERMUTACIONES = int(os.environ.get("N_REP", "2000"))
BASE_EST = ROOT / "info-hospitales" / "Base de Establecimientos 2023.xlsx"

rng = np.random.default_rng(RANDOM_STATE)

print("=" * 80)
print("COMPARACION DIRECTA DE HOMOGENEIDAD: Ward vs. MINSAL (Observacion 17)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar matriz, preprocesar (idéntico al pipeline de clustering)
# ---------------------------------------------------------------------------
print("\n[1/6] Cargando matriz institucional y preprocesando (log1p + RobustScaler)...")
matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
ids = matriz["COD_HOSPITAL"].tolist()
n_total = len(ids)

feats_clu_cols = [c for c in matriz.columns if c not in COLS_DROP_CLUSTERING]
feats = matriz[feats_clu_cols].copy()
for col in LOG1P_COLS:
    if col in feats.columns:
        feats[col] = np.log1p(feats[col].clip(lower=0))
X = RobustScaler().fit_transform(feats.values)
print(f"  X: {X.shape}")

cols_directas = [
    c for c in matriz.columns
    if c not in {"COD_HOSPITAL", "peso_medio_cma_imputado"} and not c.startswith("dim_")
    and pd.api.types.is_numeric_dtype(matriz[c])
]
print(f"  Variables directas para MAD/IQR: {len(cols_directas)}")

# ---------------------------------------------------------------------------
# 2. Cargar particiones: Ward K=4 (original), Ward K=2, MINSAL (2 categorias)
# ---------------------------------------------------------------------------
print("\n[2/6] Cargando particion Ward K=4 y calculando Ward K=2 sobre el mismo X...")
asig_original = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig_original["COD_HOSPITAL"] = asig_original["COD_HOSPITAL"].astype(str)
labels_ward4 = asig_original.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int).reindex(ids).values

Z = linkage(X, method="ward")
labels_ward2 = fcluster(Z, t=2, criterion="maxclust") - 1
print(f"  Ward K=2 tamanios: {pd.Series(labels_ward2).value_counts().sort_index().to_dict()}")
print(f"  Ward K=4 tamanios: {pd.Series(labels_ward4).value_counts().sort_index().to_dict()}")

print("\n  Cargando clasificacion MINSAL (Nivel de Complejidad)...")
base = pd.read_excel(BASE_EST, skiprows=1)
base.columns = [str(c).strip() for c in base.columns]
base["COD_HOSPITAL"] = (
    base["Código Vigente"].astype(str).str.replace(".0", "", regex=False).str.strip()
)
minsal_raw = (
    base.dropna(subset=["Nivel de Complejidad"])
    .drop_duplicates("COD_HOSPITAL")
    .set_index("COD_HOSPITAL")["Nivel de Complejidad"]
)
minsal_str = pd.Series(ids).map(minsal_raw)
labels_minsal = minsal_str.map({"Alta Complejidad": 0, "Mediana Complejidad": 1}).values
print(f"  MINSAL tamanios: {pd.Series(labels_minsal).value_counts().sort_index().to_dict()}")


# ---------------------------------------------------------------------------
# 3. Funciones de dispersion: WSS normalizado, MAD/IQR agregados, Silhouette
# ---------------------------------------------------------------------------
def wss_normalizado(X_mat: np.ndarray, labels: np.ndarray) -> float:
    """Suma de cuadrados intragrupo / suma de cuadrados total (sobre X_mat).

    Menor valor = mayor homogeneidad interna relativa al total de variacion.
    No depende de una escala arbitraria de K de la misma forma que Silhouette,
    aunque particiones con mas grupos SIEMPRE pueden igualar o reducir el WSS
    (nunca lo empeoran), por lo que sigue reportandose junto al control por K.
    """
    centro_global = X_mat.mean(axis=0)
    tss = float(np.sum((X_mat - centro_global) ** 2))
    wss = 0.0
    for g in np.unique(labels):
        Xg = X_mat[labels == g]
        centro_g = Xg.mean(axis=0)
        wss += float(np.sum((Xg - centro_g) ** 2))
    return wss / tss if tss > 0 else np.nan


def mad_iqr_por_variable(df_vars: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """MAD e IQR de cada variable dentro de cada grupo, promediado (ponderado
    por tamanio de grupo) y normalizado por el MAD/IQR global de la variable.
    """
    filas = []
    for col in df_vars.columns:
        vals_global = df_vars[col].astype(float).values
        mad_global = float(np.median(np.abs(vals_global - np.median(vals_global))))
        iqr_global = float(np.percentile(vals_global, 75) - np.percentile(vals_global, 25))

        mad_ponderado = 0.0
        iqr_ponderado = 0.0
        n_total_local = 0
        for g in np.unique(labels):
            vals_g = vals_global[labels == g]
            n_g = len(vals_g)
            if n_g == 0:
                continue
            mad_g = float(np.median(np.abs(vals_g - np.median(vals_g))))
            iqr_g = float(np.percentile(vals_g, 75) - np.percentile(vals_g, 25)) if n_g > 1 else 0.0
            mad_ponderado += mad_g * n_g
            iqr_ponderado += iqr_g * n_g
            n_total_local += n_g
        mad_ponderado /= n_total_local
        iqr_ponderado /= n_total_local

        filas.append({
            "variable": col,
            "mad_intragrupo_ponderado": mad_ponderado,
            "mad_global": mad_global,
            "mad_normalizado": (mad_ponderado / mad_global) if mad_global > 0 else np.nan,
            "iqr_intragrupo_ponderado": iqr_ponderado,
            "iqr_global": iqr_global,
            "iqr_normalizado": (iqr_ponderado / iqr_global) if iqr_global > 0 else np.nan,
        })
    return pd.DataFrame(filas)


def resumen_particion(nombre: str, X_mat: np.ndarray, labels: np.ndarray,
                       df_vars: pd.DataFrame) -> dict:
    sil = silhouette_score(X_mat, labels) if len(np.unique(labels)) > 1 else np.nan
    wss = wss_normalizado(X_mat, labels)
    df_mad = mad_iqr_por_variable(df_vars, labels)
    mad_prom = float(df_mad["mad_normalizado"].mean())
    iqr_prom = float(df_mad["iqr_normalizado"].mean())
    tamanos = pd.Series(labels).value_counts().sort_index().tolist()
    return {
        "particion": nombre, "K": len(np.unique(labels)), "tamanos": str(tamanos),
        "silhouette": sil, "wss_normalizado": wss,
        "mad_normalizado_promedio": mad_prom, "iqr_normalizado_promedio": iqr_prom,
    }, df_mad


# ---------------------------------------------------------------------------
# 4. Comparacion NO controlada: Ward K=4 (original) vs MINSAL (2 categorias)
# ---------------------------------------------------------------------------
print("\n[3/6] Comparacion NO controlada por K: Ward K=4 (real) vs. MINSAL (2 categorias)...")
mask_validos = ~pd.isna(labels_minsal)
X_valid = X[mask_validos.values] if hasattr(mask_validos, "values") else X[mask_validos]
labels_ward4_v = labels_ward4[mask_validos]
labels_minsal_v = labels_minsal[mask_validos].astype(int)
feats_directas_valid = matriz.loc[mask_validos.values if hasattr(mask_validos, "values") else mask_validos,
                                   cols_directas].reset_index(drop=True)

res_ward4, mad_ward4 = resumen_particion("Ward K=4 (real)", X_valid, labels_ward4_v, feats_directas_valid)
res_minsal2, mad_minsal = resumen_particion("MINSAL (2 categorias)", X_valid, labels_minsal_v, feats_directas_valid)

df_no_controlada = pd.DataFrame([res_ward4, res_minsal2])
df_no_controlada.to_csv(TABLES_DIR / "homogeneidad_no_controlada.csv", index=False)
print(df_no_controlada.to_string(index=False, float_format="%.4f"))

# ---------------------------------------------------------------------------
# 5. Comparacion CONTROLADA por K=2: Ward K=2 vs MINSAL (2 categorias)
# ---------------------------------------------------------------------------
print("\n[4/6] Comparacion CONTROLADA por K=2: Ward K=2 vs. MINSAL (2 categorias)...")
labels_ward2_v = labels_ward2[mask_validos.values if hasattr(mask_validos, "values") else mask_validos]

res_ward2, mad_ward2 = resumen_particion("Ward K=2", X_valid, labels_ward2_v, feats_directas_valid)

df_controlada = pd.DataFrame([res_ward2, res_minsal2])
df_controlada.to_csv(TABLES_DIR / "homogeneidad_controlada_k2.csv", index=False)
print(df_controlada.to_string(index=False, float_format="%.4f"))

# Persistir MAD/IQR detallado por variable y particion
for nombre, df_mad in [("Ward_K4", mad_ward4), ("MINSAL_2", mad_minsal), ("Ward_K2", mad_ward2)]:
    df_mad.insert(0, "particion", nombre)
mad_completo = pd.concat([mad_ward4, mad_minsal, mad_ward2], ignore_index=True)
mad_completo.to_csv(TABLES_DIR / "homogeneidad_por_variable.csv", index=False)
print(f"  Persistido: homogeneidad_por_variable.csv")

# ---------------------------------------------------------------------------
# 6. Prueba de permutacion: WSS normalizado observado vs. distribucion nula
#    de particiones aleatorias con los MISMOS tamanios de grupo
# ---------------------------------------------------------------------------
print(f"\n[5/6] Ejecutando {N_PERMUTACIONES} permutaciones por particion "
      f"(mismos tamanios de grupo, orden aleatorio de hospitales)...")


def null_wss(X_mat: np.ndarray, labels_reales: np.ndarray, n_rep: int) -> np.ndarray:
    n = len(labels_reales)
    labels_arr = labels_reales.copy()
    valores = np.empty(n_rep)
    for i in range(n_rep):
        perm = rng.permutation(labels_arr)
        valores[i] = wss_normalizado(X_mat, perm)
    return valores


filas_perm = []
for nombre, labels_v in [
    ("Ward K=4 (real)", labels_ward4_v),
    ("MINSAL (2 categorias)", labels_minsal_v),
    ("Ward K=2", labels_ward2_v),
]:
    wss_obs = wss_normalizado(X_valid, labels_v)
    null_dist = null_wss(X_valid, labels_v, N_PERMUTACIONES)
    p_valor = float((null_dist <= wss_obs).mean())  # proporcion de nulos tan homogeneos o mas
    z_score = float((wss_obs - null_dist.mean()) / null_dist.std())
    filas_perm.append({
        "particion": nombre, "wss_normalizado_observado": wss_obs,
        "wss_normalizado_nulo_media": float(null_dist.mean()),
        "wss_normalizado_nulo_std": float(null_dist.std()),
        "z_score": z_score,
        "p_valor_menor_o_igual": p_valor,
        "n_permutaciones": N_PERMUTACIONES,
    })
    print(f"  {nombre}: WSS_norm observado={wss_obs:.4f} | nulo media={null_dist.mean():.4f} "
          f"(std={null_dist.std():.4f}) | z={z_score:.2f} | p={p_valor:.4f}")

df_perm = pd.DataFrame(filas_perm)
df_perm.to_csv(TABLES_DIR / "permutacion_homogeneidad.csv", index=False)
print(f"  Persistido: permutacion_homogeneidad.csv")

# ---------------------------------------------------------------------------
# 7. Resumen final
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print("\nComparacion NO controlada (Ward K=4 real vs. MINSAL K=2):")
print(df_no_controlada[["particion", "K", "silhouette", "wss_normalizado",
                         "mad_normalizado_promedio", "iqr_normalizado_promedio"]]
      .to_string(index=False, float_format="%.4f"))
print("\nComparacion CONTROLADA por K=2 (Ward K=2 vs. MINSAL K=2):")
print(df_controlada[["particion", "K", "silhouette", "wss_normalizado",
                      "mad_normalizado_promedio", "iqr_normalizado_promedio"]]
      .to_string(index=False, float_format="%.4f"))
print("\nPruebas de permutacion (z-score y p-valor de homogeneidad vs. azar, mismos tamanios):")
print(df_perm[["particion", "wss_normalizado_observado", "z_score", "p_valor_menor_o_igual"]]
      .to_string(index=False, float_format="%.4f"))
print("\nAnalisis completo.")
