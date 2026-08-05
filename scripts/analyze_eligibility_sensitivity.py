"""Evalúa la sensibilidad de la partición Ward al criterio de elegibilidad temporal.

Compara la solución final de 65 hospitales con una reconstrucción que incorpora
los siete establecimientos de cobertura temporal incompleta.

Salidas: `sensibilidad_elegibilidad_matriz_ampliada.csv`,
`sensibilidad_elegibilidad_particiones.csv`,
`sensibilidad_elegibilidad_contingencia.csv` y
`sensibilidad_elegibilidad_hospitales_excluidos.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
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

print("=" * 80)
print("SENSIBILIDAD DEL CRITERIO DE ELEGIBILIDAD: universo ampliado (72 hospitales)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar datos y universos (65 elegibles + 72 ampliado)
# ---------------------------------------------------------------------------
# Se usa el parquet v2, que conserva los 35 diagnósticos y 30 procedimientos:
# reconstruir el universo ampliado sobre la versión truncada compararía dos
# matrices con distinta cobertura de codificación y no aislaría el efecto del
# criterio de elegibilidad, que es lo que esta prueba busca medir.
print("\n[1/6] Cargando grd_filtrado_v2.parquet (72 hospitales, sin filtro de elegibilidad)...")
df_grd_full = cargar_grd_compacto(PROCESSED_DIR / "grd_filtrado.parquet")
df_grd_full["COD_HOSPITAL"] = df_grd_full["COD_HOSPITAL"].astype(str)
universo_72 = pd.Index(sorted(df_grd_full["COD_HOSPITAL"].unique()))
print(f"  Universo ampliado (sin filtro de elegibilidad): {len(universo_72)} hospitales")

matriz_original = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz_original["COD_HOSPITAL"] = matriz_original["COD_HOSPITAL"].astype(str)
universo_65 = pd.Index(sorted(matriz_original["COD_HOSPITAL"].tolist()))
print(f"  Universo elegible original (>=3 años): {len(universo_65)} hospitales")

excluidos_7 = universo_72.difference(universo_65)
print(f"  Hospitales excluidos por el criterio de 3 años (a incorporar ahora): "
      f"{len(excluidos_7)} -> {list(excluidos_7)}")

asig_original = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig_original["COD_HOSPITAL"] = asig_original["COD_HOSPITAL"].astype(str)
labels_original = asig_original.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int)

# ---------------------------------------------------------------------------
# 2. Reconstruir features sobre el universo ampliado de 72 hospitales
# ---------------------------------------------------------------------------
print("\n[2/6] Reconstruyendo features sobre el universo ampliado (72 hospitales)...")

df_cie10 = pd.read_excel(ROOT / "insumos" / "maestras" / "CIE-10.xlsx")
df_cie9 = pd.read_excel(ROOT / "insumos" / "maestras" / "CIE-9 .xlsx")
mapper_cie10 = Mapeador_CIE10(df_maestra=df_cie10)
mapper_cie9 = Mapeador_CIE9(df_maestra=df_cie9)
mapper_cie10.construir()
mapper_cie9.construir()

constructor_feat = Constructor_Features(
    df_grd=df_grd_full, mapper_cie10=mapper_cie10, mapper_cie9=mapper_cie9,
)
# Forzar el universo de elegibles al universo ampliado de 72 (desactiva el
# filtro de >=3 años; incorpora los 7 hospitales con 1-2 años de datos).
constructor_feat._hospitales_elegibles = universo_72

f_trad = constructor_feat.features_tradicionales()
f_div = constructor_feat.features_diversidad()
f_cma = constructor_feat.features_cma()
v_cap = constructor_feat.vector_capitulos_ponderado()
v_sec = constructor_feat.vector_secciones()
v_top20 = constructor_feat.vector_top20()
print(f"  features_tradicionales: {f_trad.shape} | features_diversidad: {f_div.shape} | "
      f"features_cma: {f_cma.shape}")

constructor_ext = Constructor_Features_Extendidas(
    df_grd=df_grd_full, hospitales_elegibles=universo_72,
)
f_ext = constructor_ext.construir_todas()
print(f"  features_extendidas: {f_ext.shape}")

cas = (
    v_cap.merge(v_sec, on="COD_HOSPITAL", how="inner", suffixes=("", "_sec"))
    .merge(v_top20, on="COD_HOSPITAL", how="inner", suffixes=("", "_top"))
)
red = Reductor_Dimensional(matriz_casuistica=cas)
pca = red.fit_pca()
print(f"  PCA casuistica: K={pca['K']} componentes, "
      f"varianza acumulada={pca['explained_variance_ratio'].sum():.3f}")

matriz_ampliada = Constructor_Matriz(
    features_tradicionales=f_trad, features_diversidad=f_div,
    features_cma=f_cma, casuistica_reducida=pca["componentes"],
    features_extendidas=f_ext,
).construir()
matriz_ampliada["COD_HOSPITAL"] = matriz_ampliada["COD_HOSPITAL"].astype(str)
print(f"  Hospital_Matrix_Integrada (universo ampliado): {matriz_ampliada.shape}")

faltantes = set(universo_72) - set(matriz_ampliada["COD_HOSPITAL"])
if faltantes:
    print(f"  [AVISO] {len(faltantes)} hospitales del universo ampliado quedaron sin "
          f"fila en la matriz (posible por falta de datos suficientes para algun "
          f"bloque de features): {faltantes}")

matriz_ampliada.to_csv(TABLES_DIR / "sensibilidad_elegibilidad_matriz_ampliada.csv", index=False)
print(f"  Persistido: v5_matriz_universo_ampliado.csv")

# ---------------------------------------------------------------------------
# 3. Preprocesar y clustering Ward K=4 sobre el universo ampliado
# ---------------------------------------------------------------------------
print("\n[3/6] Preprocesando y ejecutando Ward K=4 sobre el universo ampliado...")


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


X_amp, ids_amp = preprocesar(matriz_ampliada)
Z_amp = linkage(X_amp, method="ward")
labels_amp_arr = fcluster(Z_amp, t=4, criterion="maxclust") - 1
sil_amp = silhouette_score(X_amp, labels_amp_arr)
sizes_amp = pd.Series(labels_amp_arr).value_counts().sort_index().tolist()
print(f"  Ward K=4 (universo ampliado, n={len(ids_amp)}): Silhouette={sil_amp:.3f} | "
      f"tamanos={sizes_amp}")

labels_amp = pd.Series(labels_amp_arr, index=ids_amp, name="cluster_ampliado")

# ---------------------------------------------------------------------------
# 4. Comparar particiones: original (65) vs ampliada restringida a los 65 comunes
# ---------------------------------------------------------------------------
print("\n[4/6] Comparando particion original (65) vs ampliada (restringida a los 65 comunes)...")

comparacion = pd.DataFrame({
    "COD_HOSPITAL": [h for h in ids_amp if h in universo_65],
})
comparacion["cluster_original"] = comparacion["COD_HOSPITAL"].map(labels_original).astype(int)
comparacion["cluster_ampliado"] = comparacion["COD_HOSPITAL"].map(labels_amp).astype(int)

ari = adjusted_rand_score(comparacion["cluster_original"], comparacion["cluster_ampliado"])
nmi = normalized_mutual_info_score(comparacion["cluster_original"], comparacion["cluster_ampliado"])
print(f"  ARI (original vs ampliado, sobre los 65 hospitales comunes): {ari:.3f}")
print(f"  NMI (original vs ampliado, sobre los 65 hospitales comunes): {nmi:.3f}")

contingencia = pd.crosstab(
    comparacion["cluster_original"], comparacion["cluster_ampliado"],
    rownames=["original_65"], colnames=["ampliado_72_restringido_a_65"],
)
print("\n  Tabla de contingencia (original_65 x ampliado_restringido):")
print(contingencia.to_string())

comparacion.to_csv(TABLES_DIR / "sensibilidad_elegibilidad_particiones.csv", index=False)
contingencia.to_csv(TABLES_DIR / "sensibilidad_elegibilidad_contingencia.csv")
n_cambian = int((comparacion["cluster_original"] != comparacion["cluster_ampliado"]).sum())
print(f"  Persistido: v5_comparacion_particiones.csv, v5_contingencia_particiones.csv")
print(f"  Hospitales cuya etiqueta cambia respecto de la particion original: "
      f"{n_cambian} de {len(comparacion)} (nota: etiquetas numericas son arbitrarias "
      f"entre corridas, usar tabla de contingencia)")

# ---------------------------------------------------------------------------
# 5. Donde caen los 7 hospitales excluidos, en la particion ampliada
# ---------------------------------------------------------------------------
print("\n[5/6] Ubicacion de los 7 hospitales de baja cobertura en la particion ampliada...")

anios_excluidos = (
    df_grd_full[df_grd_full["COD_HOSPITAL"].isin(excluidos_7)]
    .groupby("COD_HOSPITAL")["anio"].nunique()
)
tabla_excluidos = pd.DataFrame({
    "COD_HOSPITAL": list(excluidos_7),
})
tabla_excluidos["anios_presentes"] = tabla_excluidos["COD_HOSPITAL"].map(anios_excluidos)
tabla_excluidos["cluster_ampliado"] = tabla_excluidos["COD_HOSPITAL"].map(labels_amp)

cols_indicadores = ["egresos_por_anio", "estancia_media", "peso_medio_grd",
                     "severidad_media", "mortalidad_media", "tasa_cma"]
matriz_amp_idx = matriz_ampliada.set_index("COD_HOSPITAL")
for col in cols_indicadores:
    if col in matriz_amp_idx.columns:
        tabla_excluidos[col] = tabla_excluidos["COD_HOSPITAL"].map(matriz_amp_idx[col])

print(tabla_excluidos.to_string(index=False, float_format="%.2f"))
tabla_excluidos.to_csv(TABLES_DIR / "sensibilidad_elegibilidad_hospitales_excluidos.csv", index=False)
print(f"  Persistido: v5_indicadores_hospitales_excluidos.csv")

# ---------------------------------------------------------------------------
# 6. Resumen
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESUMEN FINAL — Sensibilidad del criterio de elegibilidad")
print("=" * 80)
print(f"  Universo original (>=3 años): {len(universo_65)} hospitales")
print(f"  Universo ampliado (sin filtro de presencia temporal): {len(universo_72)} hospitales")
print(f"  ARI (65 comunes, original vs ampliado): {ari:.3f}")
print(f"  NMI (65 comunes, original vs ampliado): {nmi:.3f}")
print(f"  Silhouette ampliado (K=4, n=72): {sil_amp:.3f} "
      f"(original, n=65: {silhouette_referencia():.3f})")
print(f"  Hospitales de los 65 originales que cambian de cluster al incorporar "
      f"los 7 excluidos: {n_cambian}")
print(f"  Distribucion de los 7 hospitales excluidos en la particion ampliada: "
      f"{tabla_excluidos['cluster_ampliado'].value_counts().sort_index().to_dict()}")
print("\nAnalisis completo.")
