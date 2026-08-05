"""Analisis de sensibilidad temporal (Observacion 8 de correcciones.txt).

Evalua si la agregacion 2019-2024 (pool de egresos sin ponderacion por año,
ver Constructor_Features / Constructor_Features_Extendidas) oculta cambios
estructurales asociados a la pandemia de COVID-19 (2020-2021).

Estrategia:
  1. Recalcular la Hospital_Matrix_Integrada EXCLUYENDO 2020-2021, sobre el
     MISMO universo de 65 hospitales elegibles (no se re-deriva elegibilidad,
     para que la comparacion sea limpia y no mezcle un cambio de universo con
     un cambio de periodo).
  2. Re-ejecutar Ward K=4 sobre la matriz sin pandemia y comparar contra la
     particion original (ARI, NMI, cambios de asignacion por hospital).
  3. Cuantificar cuanto cambian los indicadores clave por hospital al excluir
     2020-2021 (sensibilidad por variable).
  4. Comparar hospitales con cobertura completa (6 años) vs incompleta
     (3-5 años) en sus indicadores institucionales.

Salidas (reports/tables/):
  - temporal_matriz_sin_pandemia.csv           (matriz recalculada, 65 x 35)
  - temporal_comparacion_particiones.csv       (ARI, NMI, cambios por hospital)
  - temporal_sensibilidad_por_variable.csv     (delta por variable, pool vs sin 2020-21)
  - temporal_cobertura_completa_vs_incompleta.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.etl.casuistica import Constructor_Features
from src.etl.cie_mappers import Mapeador_CIE10, Mapeador_CIE9
from src.etl.features_extendidas import Constructor_Features_Extendidas
from src.modeling.matriz import Constructor_Matriz
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS, silhouette_referencia
from src.modeling.reducer import Reductor_Dimensional
from src.utils.io import PROCESSED_DIR, TABLES_DIR, cargar_grd_compacto

RANDOM_STATE = 42
ANIOS_EXCLUIDOS = [2020, 2021]

print("=" * 80)
print("ANALISIS DE SENSIBILIDAD TEMPORAL: exclusion de 2020-2021 (pandemia)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar datos originales y universo de 65 hospitales elegibles (FIJO)
# ---------------------------------------------------------------------------
print("\n[1/7] Cargando grd_filtrado.parquet y universo de 65 hospitales...")
df_grd_full = cargar_grd_compacto(PROCESSED_DIR / "grd_filtrado.parquet")
print(f"  Egresos totales (2019-2024): {len(df_grd_full):,}")

matriz_original = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz_original["COD_HOSPITAL"] = matriz_original["COD_HOSPITAL"].astype(str)
universo_65 = pd.Index(sorted(matriz_original["COD_HOSPITAL"].tolist()))
print(f"  Universo fijo (elegibles con 2019-2024 completo): {len(universo_65)} hospitales")

# Particion original (Ward K=4) para comparar despues
asig_original = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig_original["COD_HOSPITAL"] = asig_original["COD_HOSPITAL"].astype(str)
labels_original = asig_original.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int)

# ---------------------------------------------------------------------------
# 2. Filtrar egresos excluyendo 2020-2021, restringido al universo de 65
# ---------------------------------------------------------------------------
print(f"\n[2/7] Filtrando egresos: excluyendo años {ANIOS_EXCLUIDOS}, "
      f"universo fijo de {len(universo_65)} hospitales...")
df_grd_sin_pandemia = df_grd_full[
    (~df_grd_full["anio"].isin(ANIOS_EXCLUIDOS))
    & (df_grd_full["COD_HOSPITAL"].astype(str).isin(universo_65))
].copy()
df_grd_sin_pandemia["COD_HOSPITAL"] = df_grd_sin_pandemia["COD_HOSPITAL"].astype(str)
print(f"  Egresos resultantes (2019, 2022-2024): {len(df_grd_sin_pandemia):,} "
      f"({len(df_grd_sin_pandemia) / len(df_grd_full) * 100:.1f}% del total)")

anios_por_hosp = (
    df_grd_sin_pandemia.groupby("COD_HOSPITAL")["anio"].nunique()
)
sin_datos = universo_65[~universo_65.isin(anios_por_hosp.index)]
if len(sin_datos):
    print(f"  [AVISO] {len(sin_datos)} hospitales sin ningun egreso en el periodo "
          f"reducido (no deberia ocurrir con el universo de 65): {list(sin_datos)}")

# ---------------------------------------------------------------------------
# 3. Reconstruir features SOLO con el periodo reducido, forzando el mismo
#    universo de 65 hospitales (no se re-deriva elegibilidad).
# ---------------------------------------------------------------------------
print("\n[3/7] Reconstruyendo features sobre el periodo reducido (mismo universo)...")

df_cie10 = pd.read_excel(ROOT / "insumos" / "maestras" / "CIE-10.xlsx")
df_cie9 = pd.read_excel(ROOT / "insumos" / "maestras" / "CIE-9 .xlsx")
mapper_cie10 = Mapeador_CIE10(df_maestra=df_cie10)
mapper_cie9 = Mapeador_CIE9(df_maestra=df_cie9)
mapper_cie10.construir()
mapper_cie9.construir()

constructor_feat = Constructor_Features(
    df_grd=df_grd_sin_pandemia, mapper_cie10=mapper_cie10, mapper_cie9=mapper_cie9,
)
# Forzar el universo de elegibles al conjunto fijo de 65 (evita que el
# criterio de >=3 años excluya hospitales solo por haber quitado 2020-2021,
# lo que mezclaria dos efectos distintos).
constructor_feat._hospitales_elegibles = universo_65

f_trad_sp = constructor_feat.features_tradicionales()
f_div_sp = constructor_feat.features_diversidad()
f_cma_sp = constructor_feat.features_cma()
v_cap_sp = constructor_feat.vector_capitulos_ponderado()
v_sec_sp = constructor_feat.vector_secciones()
v_top20_sp = constructor_feat.vector_top20()
print(f"  features_tradicionales: {f_trad_sp.shape} | "
      f"features_diversidad: {f_div_sp.shape} | features_cma: {f_cma_sp.shape}")

constructor_ext = Constructor_Features_Extendidas(
    df_grd=df_grd_sin_pandemia, hospitales_elegibles=universo_65,
)
f_ext_sp = constructor_ext.construir_todas()
print(f"  features_extendidas: {f_ext_sp.shape}")

cas_sp = (
    v_cap_sp.merge(v_sec_sp, on="COD_HOSPITAL", how="inner", suffixes=("", "_sec"))
    .merge(v_top20_sp, on="COD_HOSPITAL", how="inner", suffixes=("", "_top"))
)
red_sp = Reductor_Dimensional(matriz_casuistica=cas_sp)
pca_sp = red_sp.fit_pca()
print(f"  PCA casuistica: K={pca_sp['K']} componentes, "
      f"varianza acumulada={pca_sp['explained_variance_ratio'].sum():.3f}")

matriz_sp = Constructor_Matriz(
    features_tradicionales=f_trad_sp, features_diversidad=f_div_sp,
    features_cma=f_cma_sp, casuistica_reducida=pca_sp["componentes"],
    features_extendidas=f_ext_sp,
).construir()
matriz_sp["COD_HOSPITAL"] = matriz_sp["COD_HOSPITAL"].astype(str)
print(f"  Hospital_Matrix_Integrada (sin pandemia): {matriz_sp.shape}")

faltantes = set(universo_65) - set(matriz_sp["COD_HOSPITAL"])
if faltantes:
    raise RuntimeError(
        f"Universo incompleto tras excluir 2020-2021: faltan {faltantes}. "
        f"Revisar por que estos hospitales quedaron sin datos en el periodo reducido."
    )

matriz_sp.to_csv(TABLES_DIR / "temporal_matriz_sin_pandemia.csv", index=False)
print(f"  Persistido: temporal_matriz_sin_pandemia.csv")

# ---------------------------------------------------------------------------
# 4. Preprocesar (log1p + RobustScaler) y clustering Ward K=4
# ---------------------------------------------------------------------------
print("\n[4/7] Preprocesando y re-ejecutando Ward K=4 sobre la matriz sin pandemia...")


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


X_sp, ids_sp = preprocesar(matriz_sp)
Z_sp = linkage(X_sp, method="ward")
labels_sp_arr = fcluster(Z_sp, t=4, criterion="maxclust") - 1  # 0-indexed
sil_sp = silhouette_score(X_sp, labels_sp_arr)
sizes_sp = pd.Series(labels_sp_arr).value_counts().sort_index().tolist()
print(f"  Ward K=4 (sin pandemia): Silhouette={sil_sp:.3f} | tamanos={sizes_sp}")

labels_sp = pd.Series(labels_sp_arr, index=ids_sp, name="cluster_sin_pandemia")

# ---------------------------------------------------------------------------
# 5. Comparar particiones: original (pool 2019-2024) vs sin pandemia
# ---------------------------------------------------------------------------
print("\n[5/7] Comparando particion original vs particion sin pandemia...")

comparacion = pd.DataFrame({
    "COD_HOSPITAL": ids_sp,
    "cluster_original": [int(labels_original.get(h, -1)) for h in ids_sp],
    "cluster_sin_pandemia": labels_sp.values,
})
mask_validos = comparacion["cluster_original"] != -1
ari = adjusted_rand_score(
    comparacion.loc[mask_validos, "cluster_original"],
    comparacion.loc[mask_validos, "cluster_sin_pandemia"],
)
nmi = normalized_mutual_info_score(
    comparacion.loc[mask_validos, "cluster_original"],
    comparacion.loc[mask_validos, "cluster_sin_pandemia"],
)
print(f"  ARI (original vs sin pandemia): {ari:.3f}")
print(f"  NMI (original vs sin pandemia): {nmi:.3f}")

# Tabla de contingencia (crosstab) para inspeccionar cambios de composicion
contingencia = pd.crosstab(
    comparacion["cluster_original"], comparacion["cluster_sin_pandemia"],
    rownames=["original"], colnames=["sin_pandemia"],
)
print("\n  Tabla de contingencia (original x sin_pandemia):")
print(contingencia.to_string())

comparacion.to_csv(TABLES_DIR / "temporal_comparacion_particiones.csv", index=False)
contingencia.to_csv(TABLES_DIR / "temporal_contingencia_particiones.csv")
with open(TABLES_DIR / "temporal_metricas_concordancia.txt", "w") as f:
    f.write(f"ARI={ari:.4f}\nNMI={nmi:.4f}\n")
    f.write(f"silhouette_sin_pandemia_K4={sil_sp:.4f}\n")
    f.write(f"tamanos_sin_pandemia={sizes_sp}\n")
print(f"  Persistido: temporal_comparacion_particiones.csv, temporal_contingencia_particiones.csv")

# ---------------------------------------------------------------------------
# 6. Sensibilidad por variable: delta entre matriz original y matriz sin pandemia
# ---------------------------------------------------------------------------
print("\n[6/7] Cuantificando sensibilidad por variable (delta relativo por hospital)...")

cols_comunes = [
    c for c in matriz_original.columns
    if c in matriz_sp.columns and c not in ("COD_HOSPITAL", "peso_medio_cma_imputado")
]
orig_idx = matriz_original.set_index("COD_HOSPITAL")[cols_comunes].reindex(universo_65)
sp_idx = matriz_sp.set_index("COD_HOSPITAL")[cols_comunes].reindex(universo_65)

filas_sens = []
for col in cols_comunes:
    orig_vals = orig_idx[col].astype(float)
    sp_vals = sp_idx[col].astype(float)
    delta_abs = (sp_vals - orig_vals).abs()
    denom = orig_vals.abs().replace(0, np.nan)
    delta_rel = (delta_abs / denom).replace([np.inf, -np.inf], np.nan)
    filas_sens.append({
        "variable": col,
        "delta_abs_mediana": float(delta_abs.median()),
        "delta_abs_max": float(delta_abs.max()),
        "delta_rel_mediana_pct": float(delta_rel.median() * 100) if delta_rel.notna().any() else np.nan,
        "delta_rel_max_pct": float(delta_rel.max() * 100) if delta_rel.notna().any() else np.nan,
        "correlacion_orig_vs_sp": float(orig_vals.corr(sp_vals)) if orig_vals.std() > 0 and sp_vals.std() > 0 else np.nan,
    })
df_sens = pd.DataFrame(filas_sens).sort_values("delta_rel_mediana_pct", ascending=False)
df_sens.to_csv(TABLES_DIR / "temporal_sensibilidad_por_variable.csv", index=False)
print(f"  Top 10 variables mas sensibles a excluir 2020-2021 (delta relativo mediano):")
print(df_sens.head(10).to_string(index=False, float_format="%.2f"))
print(f"  Persistido: temporal_sensibilidad_por_variable.csv")

# ---------------------------------------------------------------------------
# 7. Cobertura completa (6 años) vs incompleta (3-5 años)
# ---------------------------------------------------------------------------
print("\n[7/7] Comparando hospitales con cobertura completa vs incompleta...")

anios_por_hosp_full = (
    df_grd_full[df_grd_full["COD_HOSPITAL"].astype(str).isin(universo_65)]
    .assign(COD_HOSPITAL=lambda d: d["COD_HOSPITAL"].astype(str))
    .groupby("COD_HOSPITAL")["anio"].nunique()
    .reindex(universo_65)
)
cobertura_completa = anios_por_hosp_full[anios_por_hosp_full == 6].index
cobertura_incompleta = anios_por_hosp_full[anios_por_hosp_full < 6].index
print(f"  Cobertura completa (6 años): {len(cobertura_completa)} hospitales")
print(f"  Cobertura incompleta (3-5 años): {len(cobertura_incompleta)} hospitales")
print(f"  Distribucion de anios_presentes: "
      f"{anios_por_hosp_full.value_counts().sort_index().to_dict()}")

cols_comparar = [
    "egresos_por_anio", "estancia_media", "peso_medio_grd", "severidad_media",
    "mortalidad_media", "tasa_cma", "pct_pediatrico", "pct_geriatrico",
    "pct_urgencia", "pct_programada", "cv_estancia",
]
cols_comparar = [c for c in cols_comparar if c in matriz_original.columns]
matriz_idx = matriz_original.set_index("COD_HOSPITAL")

filas_cov = []
for col in cols_comparar:
    vals_completa = matriz_idx.loc[matriz_idx.index.isin(cobertura_completa), col].astype(float)
    vals_incompleta = matriz_idx.loc[matriz_idx.index.isin(cobertura_incompleta), col].astype(float)
    try:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(vals_completa, vals_incompleta, alternative="two-sided")
    except ValueError:
        stat, pval = np.nan, np.nan
    filas_cov.append({
        "variable": col,
        "mediana_cobertura_completa": float(vals_completa.median()),
        "mediana_cobertura_incompleta": float(vals_incompleta.median()),
        "n_completa": int(vals_completa.shape[0]),
        "n_incompleta": int(vals_incompleta.shape[0]),
        "mannwhitney_p": float(pval) if pval is not None else np.nan,
    })
df_cov = pd.DataFrame(filas_cov)
df_cov.to_csv(TABLES_DIR / "temporal_cobertura_completa_vs_incompleta.csv", index=False)
print(df_cov.to_string(index=False, float_format="%.3f"))
print(f"  Persistido: temporal_cobertura_completa_vs_incompleta.csv")

print("\n" + "=" * 80)
print("RESUMEN FINAL — Sensibilidad temporal (exclusion 2020-2021)")
print("=" * 80)
print(f"  ARI original vs sin-pandemia: {ari:.3f}")
print(f"  NMI original vs sin-pandemia: {nmi:.3f}")
print(f"  Silhouette sin-pandemia (K=4): {sil_sp:.3f} "
      f"(original: {silhouette_referencia():.3f})")
n_cambian = int((comparacion["cluster_original"] != comparacion["cluster_sin_pandemia"]).sum())
print(f"  Hospitales cuya etiqueta NUMERICA de cluster difiere: {n_cambian} de {len(comparacion)}")
print("  (nota: la etiqueta numerica de cluster es arbitraria entre corridas; "
        "usar la tabla de contingencia para interpretar correspondencias reales)")
print("\nAnalisis completo.")
