"""Prueba de permutacion sobre el flujo completo: cuantifica el problema de
inferencia circular en Kruskal-Wallis (Observacion 14 de correcciones.txt).

Problema: los grupos del clustering se construyen usando las mismas 35
variables de la Hospital_Matrix_Integrada que luego se someten a
Kruskal-Wallis para "validar" la identidad de los grupos. Encontrar diferencias
significativas en esas variables no es evidencia independiente de que la
particion no sea producto del azar: el algoritmo de clustering, por
construccion, busca precisamente separar las observaciones en ese mismo
espacio de variables.

Estrategia: en lugar de asumir que el problema es despreciable o inevitable,
se cuantifica empiricamente cuantas variables "significativas" produciria el
flujo completo (Ward K=4 + Kruskal-Wallis + BH) si NO existiera ninguna
estructura real en los datos.

Para ello se generan datos nulos que preservan la distribucion marginal de
cada variable pero destruyen cualquier estructura de covarianza real entre
hospitales: se permuta independientemente cada columna de la matriz
institucional entre los 65 hospitales (shuffle por columna). Esto simula el
escenario "sin estructura real que separar" mientras mantiene exactamente la
misma escala y forma univariada de cada variable que en los datos reales.

Sobre cada matriz permutada se repite el pipeline completo: preprocesamiento
(log1p + RobustScaler), clustering Ward K=4, y Kruskal-Wallis + BH sobre las
mismas 28 variables interpretables (excluyendo las componentes dim_*). Se
repite 500 veces y se registra cuantas variables resultan "significativas"
(p_BH <= 0.05) en cada repeticion.

Si el flujo completo tiende a marcar muchas variables como significativas
incluso cuando los datos son ruido puro reordenado, eso demuestra
empiricamente la circularidad: el propio acto de agrupar (Ward, sobre esas
variables) ya "fabrica" separacion post-hoc, independientemente de si existe
estructura real.

Salidas (reports/tables/):
  - permutacion_circularidad_resumen.csv     (distribucion del n de variables sig. bajo H0)
  - permutacion_circularidad_detalle.csv     (todas las repeticiones, todas las variables)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import kruskal
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS

RANDOM_STATE = 42
import os
N_REPETICIONES = int(os.environ.get("N_REP", "500"))
ALPHA = 0.05

rng = np.random.default_rng(RANDOM_STATE)

print("=" * 80)
print("PRUEBA DE PERMUTACION: cuantificando la circularidad Ward -> Kruskal-Wallis")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar matriz real y variables numericas interpretables (excluye dim_*)
# ---------------------------------------------------------------------------
print("\n[1/4] Cargando matriz institucional real (65 x 35)...")
matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
ids = matriz["COD_HOSPITAL"].tolist()
n_hosp = len(ids)

EXCLUIR = {"COD_HOSPITAL", "peso_medio_cma_imputado"}
num_cols = [
    c for c in matriz.columns
    if c not in EXCLUIR and not c.startswith("dim_")
    and pd.api.types.is_numeric_dtype(matriz[c])
]
print(f"  Variables interpretables evaluadas con Kruskal-Wallis: {len(num_cols)}")
print(f"  {num_cols}")

feats_clu_cols = [c for c in matriz.columns if c not in COLS_DROP_CLUSTERING]


def preprocesar_y_clusterizar(matriz_sim: pd.DataFrame) -> np.ndarray:
    """Aplica log1p + RobustScaler + Ward K=4 sobre una matriz (real o permutada)."""
    feats = matriz_sim[feats_clu_cols].copy()
    for col in LOG1P_COLS:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    X = RobustScaler().fit_transform(feats.values)
    Z = linkage(X, method="ward")
    labels = fcluster(Z, t=4, criterion="maxclust")
    return labels


def kruskal_bh_n_significativas(matriz_sim: pd.DataFrame, labels: np.ndarray) -> int:
    """Corre Kruskal-Wallis + BH sobre num_cols y devuelve cuantas resultan significativas."""
    p_vals = []
    for col in num_cols:
        muestras = [matriz_sim.loc[labels == g, col].dropna().values for g in np.unique(labels)]
        muestras = [m for m in muestras if len(m) > 0]
        if len(muestras) < 2:
            p_vals.append(np.nan)
            continue
        try:
            _, p = kruskal(*muestras)
        except ValueError:
            p = np.nan
        p_vals.append(p)

    p_arr = np.array(p_vals)
    mask = ~np.isnan(p_arr)
    if mask.sum() == 0:
        return 0
    _, p_adj, _, _ = multipletests(p_arr[mask], alpha=ALPHA, method="fdr_bh")
    return int((p_adj <= ALPHA).sum())


# ---------------------------------------------------------------------------
# 2. Baseline: resultado real, recalculado en cada corrida (no fijado a mano)
# ---------------------------------------------------------------------------
print("\n[2/4] Verificando baseline real (Ward K=4 + Kruskal-Wallis + BH)...")
labels_real = preprocesar_y_clusterizar(matriz)
n_sig_real = kruskal_bh_n_significativas(matriz, labels_real)
print(f"  Variables significativas en los datos reales: {n_sig_real}/{len(num_cols)}")

# ---------------------------------------------------------------------------
# 3. Distribucion nula: permutar cada columna independientemente, repetir
# ---------------------------------------------------------------------------
print(f"\n[3/4] Generando {N_REPETICIONES} matrices nulas (shuffle por columna) "
      f"y repitiendo Ward K=4 + Kruskal-Wallis + BH en cada una...")

filas_resumen = []
filas_detalle = []
for rep in range(N_REPETICIONES):
    matriz_null = matriz.copy()
    for col in feats_clu_cols:
        if col == "COD_HOSPITAL":
            continue
        vals = matriz_null[col].values.copy()
        rng.shuffle(vals)
        matriz_null[col] = vals

    labels_null = preprocesar_y_clusterizar(matriz_null)
    n_clusters_efectivo = len(np.unique(labels_null))

    p_vals = []
    for col in num_cols:
        muestras = [
            matriz_null.loc[labels_null == g, col].dropna().values
            for g in np.unique(labels_null)
        ]
        muestras = [m for m in muestras if len(m) > 0]
        if len(muestras) < 2:
            p_vals.append(np.nan)
            continue
        try:
            _, p = kruskal(*muestras)
        except ValueError:
            p = np.nan
        p_vals.append(p)

    p_arr = np.array(p_vals)
    mask = ~np.isnan(p_arr)
    if mask.sum() > 0:
        _, p_adj_arr, _, _ = multipletests(p_arr[mask], alpha=ALPHA, method="fdr_bh")
        p_adj_full = np.full(len(num_cols), np.nan)
        p_adj_full[mask] = p_adj_arr
    else:
        p_adj_full = np.full(len(num_cols), np.nan)

    n_sig = int(np.nansum(p_adj_full <= ALPHA))
    filas_resumen.append({
        "rep": rep,
        "n_clusters_efectivo": n_clusters_efectivo,
        "n_variables_significativas": n_sig,
    })
    for col, p_adj in zip(num_cols, p_adj_full):
        filas_detalle.append({"rep": rep, "variable": col, "p_valor_ajustado_BH": p_adj})

    if (rep + 1) % 100 == 0:
        print(f"  ... {rep + 1}/{N_REPETICIONES} repeticiones completadas")

df_resumen = pd.DataFrame(filas_resumen)
df_detalle = pd.DataFrame(filas_detalle)
df_resumen.to_csv(TABLES_DIR / "permutacion_circularidad_resumen.csv", index=False)
df_detalle.to_csv(TABLES_DIR / "permutacion_circularidad_detalle.csv", index=False)
print(f"  Persistido: permutacion_circularidad_resumen.csv, permutacion_circularidad_detalle.csv")

# ---------------------------------------------------------------------------
# 4. Resumen final
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESUMEN FINAL — Distribucion nula del numero de variables 'significativas'")
print("=" * 80)
print(f"  Variables significativas en los DATOS REALES: {n_sig_real}/{len(num_cols)}")
print(f"\n  Distribucion bajo H0 (sin estructura real, {N_REPETICIONES} repeticiones):")
print(df_resumen["n_variables_significativas"].describe().to_string())
percentiles = df_resumen["n_variables_significativas"].quantile([0.5, 0.9, 0.95, 0.99])
print(f"\n  Percentiles: \n{percentiles.to_string()}")
prop_igual_o_mayor = (df_resumen["n_variables_significativas"] >= n_sig_real).mean()
print(f"\n  Proporcion de repeticiones NULAS que igualan o superan el resultado real "
      f"({n_sig_real}/{len(num_cols)}): {prop_igual_o_mayor:.3f}")
print(f"  (interpretacion: si esta proporcion es baja, el resultado real supera "
      f"claramente lo esperable por azar bajo este diseno; si es alta, sugiere "
      f"que el procedimiento fabrica significancia post-hoc independientemente "
      f"de la estructura real de los datos)")
print("\nAnalisis completo.")
