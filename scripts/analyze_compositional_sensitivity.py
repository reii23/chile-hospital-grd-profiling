"""Analisis de sensibilidad composicional (Observacion 11 de correcciones.txt).

Evalua si aplicar PCA + distancia euclidiana directamente sobre las
proporciones de casuistica (capitulos CIE-10, secciones CIE-9-MC, Top-20 GRD)
-- todas composiciones que suman ~1 -- produce artefactos de clausura
composicional respecto de aplicar primero una transformacion CLR
(centered log-ratio), que es la alternativa estandar para datos
composicionales.

Ademas, corrige la circularidad en la seleccion de variante de capitulos
(principal vs. ponderado): en el pipeline original, la variante se elegia
maximizando el Silhouette del clustering final sobre la MISMA matriz
candidata que luego se usa como solucion definitiva. Aqui se selecciona en
cambio con un criterio interno al PCA (varianza explicada acumulada a K fijo),
independiente de cualquier clustering posterior.

Estrategia:
  1. Cargar los tres vectores composicionales ya persistidos.
  2. Aplicar CLR con pseudo-conteo (manejo explicito de ceros) a cada vector.
  3. Ejecutar PCA sobre la representacion CLR concatenada (mismo K=8 que el
     pipeline original, para comparacion directa).
  4. Construir la matriz candidata CLR+PCA analoga a la Hospital_Matrix_Integrada
     y correr el mismo clustering Ward K=4.
  5. Comparar (ARI, NMI, Silhouette, tamanos) contra la particion original.
  6. Seleccionar la variante de capitulos (principal/ponderado) con un
     criterio no circular: la variante cuyo PCA (K=8 fijo) explica mayor
     varianza acumulada, evaluado independientemente en PCA convencional y
     en PCA sobre CLR.

Salidas (reports/tables/):
  - sensibilidad_composicional.csv
  - seleccion_variante_no_circular.csv
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
)
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS, silhouette_referencia
from src.utils.io import PROCESSED_DIR, TABLES_DIR

RANDOM_STATE = 42
K_PCA = 8

print("=" * 80)
print("ANALISIS DE SENSIBILIDAD COMPOSICIONAL: PCA convencional vs. CLR+PCA")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar vectores composicionales y matriz original
# ---------------------------------------------------------------------------
print("\n[1/6] Cargando vectores de casuistica y matriz institucional original...")
v_caps = pd.read_parquet(PROCESSED_DIR / "casuistica_capitulos.parquet")
v_secs = pd.read_parquet(PROCESSED_DIR / "casuistica_procedimientos.parquet")
v_top20 = pd.read_parquet(PROCESSED_DIR / "casuistica_top20_grds.parquet")

v_cap_principal = v_caps[v_caps["variante"] == "principal"].drop(columns="variante").reset_index(drop=True)
v_cap_ponderado = v_caps[v_caps["variante"] == "ponderado"].drop(columns="variante").reset_index(drop=True)

matriz_original = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz_original["COD_HOSPITAL"] = matriz_original["COD_HOSPITAL"].astype(str)

asig_original = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig_original["COD_HOSPITAL"] = asig_original["COD_HOSPITAL"].astype(str)
labels_original = asig_original.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int)

print(f"  v_cap_principal: {v_cap_principal.shape} | v_cap_ponderado: {v_cap_ponderado.shape}")
print(f"  v_secs: {v_secs.shape} | v_top20: {v_top20.shape}")


def concatenar(v_cap: pd.DataFrame) -> pd.DataFrame:
    return (
        v_cap.merge(v_secs, on="COD_HOSPITAL", how="inner", suffixes=("", "_sec"))
        .merge(v_top20, on="COD_HOSPITAL", how="inner", suffixes=("", "_top"))
    )


# ---------------------------------------------------------------------------
# 2. Transformacion CLR con manejo explicito de ceros
# ---------------------------------------------------------------------------
def clr_transform(df_prop: pd.DataFrame, id_col: str = "COD_HOSPITAL", delta: float = None) -> pd.DataFrame:
    """Aplica CLR (centered log-ratio) a una matriz de proporciones por fila.

    Maneja ceros mediante un pseudo-conteo multiplicativo estandar en
    analisis composicional (Aitchison): delta = 1 / (2 * D), donde D es el
    numero de partes de la composicion, aplicado solo a los ceros y
    renormalizando la fila para que siga sumando 1 antes de tomar logaritmos.
    """
    ids = df_prop[id_col]
    X = df_prop.drop(columns=[id_col]).astype(float).values
    D = X.shape[1]
    if delta is None:
        delta = 1.0 / (2.0 * D)

    # Pseudo-conteo multiplicativo solo sobre los ceros (Martin-Fernandez et al.)
    X_adj = X.copy()
    n_zeros = (X_adj == 0).sum(axis=1)
    for i in range(X_adj.shape[0]):
        if n_zeros[i] == 0:
            continue
        mask_zero = X_adj[i] == 0
        n_z = mask_zero.sum()
        # partes con cero reciben delta; partes no-cero se contraen proporcionalmente
        X_adj[i, mask_zero] = delta
        X_adj[i, ~mask_zero] = X_adj[i, ~mask_zero] * (1 - n_z * delta)

    # Evitar log(0) residual por error numerico
    X_adj = np.clip(X_adj, 1e-12, None)
    log_X = np.log(X_adj)
    geo_mean_log = log_X.mean(axis=1, keepdims=True)
    clr = log_X - geo_mean_log

    cols = [f"clr_{c}" for c in df_prop.drop(columns=[id_col]).columns]
    out = pd.DataFrame(clr, columns=cols)
    out.insert(0, id_col, ids.values)
    return out


print("\n[2/6] Aplicando transformacion CLR a los tres vectores composicionales...")
cas_ponderado = concatenar(v_cap_ponderado)
clr_cas = clr_transform(cas_ponderado)
print(f"  Matriz CLR (ponderado + secciones + top20): {clr_cas.shape}")

# ---------------------------------------------------------------------------
# 3. PCA sobre CLR (K=8 fijo, igual al pipeline original, para comparar en
#    igualdad de condiciones) y PCA convencional (referencia)
# ---------------------------------------------------------------------------
print(f"\n[3/6] PCA sobre representacion CLR (K={K_PCA} fijo, comparable al original)...")


def pca_fijo(df_num: pd.DataFrame, id_col: str, k: int, scaler_cls=StandardScaler):
    ids = df_num[id_col].values
    X = df_num.drop(columns=[id_col]).values
    scaler = scaler_cls()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    scores = pca.fit_transform(Xs)
    var_acum = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    cols = [f"dim_{i+1:02d}" for i in range(k)]
    out = pd.DataFrame(scores, columns=cols)
    out.insert(0, id_col, ids)
    return out, var_acum


pca_clr_componentes, var_clr = pca_fijo(clr_cas, "COD_HOSPITAL", K_PCA)
print(f"  Varianza acumulada (CLR + PCA, K={K_PCA}): {var_clr:.3f}")
# La referencia del pipeline se recalcula sobre los mismos vectores de casuística.
pca_ref = PCA(n_components=K_PCA, random_state=RANDOM_STATE).fit(
    StandardScaler().fit_transform(cas_ponderado.drop(columns=["COD_HOSPITAL"]).values)
)
var_pca_convencional = float(np.cumsum(pca_ref.explained_variance_ratio_)[-1])
print(f"  Varianza acumulada (PCA convencional, K={K_PCA}, referencia del pipeline): "
      f"{var_pca_convencional:.4f}")

# ---------------------------------------------------------------------------
# 4. Construir matriz candidata CLR+PCA analoga a Hospital_Matrix_Integrada
# ---------------------------------------------------------------------------
print("\n[4/6] Ensamblando matriz candidata con componentes CLR+PCA...")

cols_directas = [
    c for c in matriz_original.columns
    if c not in [f"dim_{i+1:02d}" for i in range(K_PCA)] and c != "COD_HOSPITAL"
]
matriz_clr = matriz_original[["COD_HOSPITAL"] + cols_directas].merge(
    pca_clr_componentes, on="COD_HOSPITAL", how="inner", validate="1:1",
)
print(f"  Matriz candidata CLR+PCA: {matriz_clr.shape}")


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


X_clr, ids_clr = preprocesar(matriz_clr)
Z_clr = linkage(X_clr, method="ward")
labels_clr_arr = fcluster(Z_clr, t=4, criterion="maxclust") - 1
sil_clr = silhouette_score(X_clr, labels_clr_arr)
sizes_clr = pd.Series(labels_clr_arr).value_counts().sort_index().tolist()
print(f"  Ward K=4 (CLR+PCA): Silhouette={sil_clr:.3f} | tamanos={sizes_clr}")

# ---------------------------------------------------------------------------
# 5. Comparar particion CLR+PCA vs particion original (PCA convencional)
# ---------------------------------------------------------------------------
print("\n[5/6] Comparando particion CLR+PCA vs particion original (PCA convencional)...")

comparacion = pd.DataFrame({
    "COD_HOSPITAL": ids_clr,
    "cluster_original_pca": [int(labels_original.get(h, -1)) for h in ids_clr],
    "cluster_clr_pca": labels_clr_arr,
})
mask_validos = comparacion["cluster_original_pca"] != -1
ari = adjusted_rand_score(
    comparacion.loc[mask_validos, "cluster_original_pca"],
    comparacion.loc[mask_validos, "cluster_clr_pca"],
)
nmi = normalized_mutual_info_score(
    comparacion.loc[mask_validos, "cluster_original_pca"],
    comparacion.loc[mask_validos, "cluster_clr_pca"],
)
print(f"  ARI (PCA convencional vs CLR+PCA): {ari:.3f}")
print(f"  NMI (PCA convencional vs CLR+PCA): {nmi:.3f}")

contingencia = pd.crosstab(
    comparacion["cluster_original_pca"], comparacion["cluster_clr_pca"],
    rownames=["pca_convencional"], colnames=["clr_pca"],
)
print("\n  Tabla de contingencia (pca_convencional x clr_pca):")
print(contingencia.to_string())

n_cambian = int((comparacion["cluster_original_pca"] != comparacion["cluster_clr_pca"]).sum())
resumen = pd.DataFrame([{
    "ari": ari, "nmi": nmi,
    "silhouette_pca_convencional": silhouette_referencia(),
    "silhouette_clr_pca": sil_clr,
    "varianza_acumulada_pca_convencional": var_pca_convencional,
    "varianza_acumulada_clr_pca": var_clr,
    "tamanos_clr_pca": str(sizes_clr),
    "n_hospitales_cambian_cluster": n_cambian,
}])
resumen.to_csv(TABLES_DIR / "sensibilidad_composicional.csv", index=False)
comparacion.to_csv(TABLES_DIR / "comparacion_particiones_composicional.csv", index=False)
contingencia.to_csv(TABLES_DIR / "contingencia_composicional.csv")
print(f"  Persistido: v4_comparacion_pca_vs_clr.csv, comparacion_particiones_composicional.csv")

# ---------------------------------------------------------------------------
# 6. Seleccion de variante de capitulos SIN circularidad
#    (criterio: varianza acumulada del PCA a K=8 fijo, tanto en PCA
#    convencional como en CLR+PCA; no se usa Silhouette del clustering final)
# ---------------------------------------------------------------------------
print("\n[6/6] Seleccion de variante de capitulos con criterio no circular...")

filas_variante = []
for nombre, v_cap in [("principal", v_cap_principal), ("ponderado", v_cap_ponderado)]:
    cas = concatenar(v_cap)

    # PCA convencional (estandarizacion + PCA sobre proporciones brutas)
    _, var_conv = pca_fijo(cas, "COD_HOSPITAL", K_PCA, scaler_cls=StandardScaler)

    # CLR + PCA
    clr_v = clr_transform(cas)
    _, var_clr_v = pca_fijo(clr_v, "COD_HOSPITAL", K_PCA, scaler_cls=StandardScaler)

    filas_variante.append({
        "variante_capitulos": nombre,
        "varianza_acumulada_pca_convencional_K8": var_conv,
        "varianza_acumulada_clr_pca_K8": var_clr_v,
    })
    print(f"  {nombre}: var_acum PCA convencional={var_conv:.4f} | var_acum CLR+PCA={var_clr_v:.4f}")

df_variante = pd.DataFrame(filas_variante)
ganador_conv = df_variante.loc[df_variante["varianza_acumulada_pca_convencional_K8"].idxmax(), "variante_capitulos"]
ganador_clr = df_variante.loc[df_variante["varianza_acumulada_clr_pca_K8"].idxmax(), "variante_capitulos"]
df_variante.to_csv(TABLES_DIR / "seleccion_variante_no_circular.csv", index=False)
print(f"\n  Variante ganadora (criterio no circular, PCA convencional): {ganador_conv}")
print(f"  Variante ganadora (criterio no circular, CLR+PCA): {ganador_clr}")
print(f"  Variante seleccionada originalmente en el pipeline (via Silhouette circular): ponderado")
print(f"  Persistido: seleccion_variante_no_circular.csv")

print("\n" + "=" * 80)
print("RESUMEN FINAL — Sensibilidad composicional (PCA convencional vs CLR+PCA)")
print("=" * 80)
print(f"  ARI: {ari:.3f} | NMI: {nmi:.3f}")
print(f"  Silhouette: {sil_clr:.3f} (CLR+PCA) vs "
      f"{silhouette_referencia():.3f} (PCA convencional)")
print(f"  Varianza acumulada K=8: {var_clr:.3f} (CLR+PCA) vs "
      f"{var_pca_convencional:.3f} (PCA convencional)")
print(f"  Hospitales que cambian de cluster: {n_cambian} de {len(comparacion)}")
print(f"  Variante de capitulos ganadora sin circularidad: {ganador_conv} (PCA) / {ganador_clr} (CLR+PCA)")
print("\nAnalisis completo.")
