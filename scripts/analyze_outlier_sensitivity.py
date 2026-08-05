"""Sensibilidad de la deteccion de hospitales atipicos (Observacion 19 de
correcciones.txt).

Problemas identificados en el pipeline original (notebooks/04_xai_outliers.ipynb):
  1. Isolation Forest se ajusta con contamination=0.10 sin justificar ese valor
     ni evaluar sensibilidad a otros valores. Con contamination=0.10 sobre 65
     hospitales, el algoritmo esta obligado a marcar ~6-7 observaciones como
     atipicas independientemente de la estructura real de los datos.
  2. Isolation Forest se ajusta GLOBALMENTE sobre los 65 hospitales (una sola
     llamada a IsolationForest().fit(X), sin agrupar por cluster), por lo que
     no es correcto describir sus resultados como deteccion "dentro de su
     propio cluster".
  3. El criterio de distancia al centroide usa desviacion estandar (ddof=0)
     calculada DENTRO de cada cluster para normalizar (z-score). Esto es
     estadisticamente fragil para los clusters pequenios C0 (n=3) y C3
     (n=3): con 3 observaciones, la desviacion estandar intra-cluster es una
     estimacion de altisima varianza y el z-score resultante no es confiable.
  4. No se reporta explicitamente cuantos y cuales hospitales son "atipicos
     robustos" (>=2 criterios) vs. detectados por un solo criterio.

Este script:
  A. Ejecuta Isolation Forest global (correcto, sin agrupar por cluster) con
     contamination in {0.05, 0.10, 0.15, 0.20} y reporta el conjunto de
     hospitales marcados en cada caso, cuantificando la sensibilidad al
     parametro y cuales hospitales son detectados de forma consistente en
     TODOS los niveles de contaminacion explorados (evidencia mas fuerte
     de atipicidad genuina, independiente del parametro elegido).
  B. Recalcula el criterio de distancia al centroide reemplazando la
     desviacion estandar intra-cluster por la desviacion absoluta mediana
     (MAD) GLOBAL de las distancias al centroide (no calculada por cluster),
     evitando el problema de estimar dispersion dentro de grupos de tamano
     2 o 3.
  C. Recalcula silhouette individual (sin cambios, ya es un criterio valido
     por si mismo) y combina los tres criterios corregidos, reportando
     explicitamente cuantos hospitales satisfacen >=2 criterios (atipicos
     robustos) vs. exactamente 1 (atipicos por un solo criterio).

Salidas (reports/tables/):
  - sensibilidad_contaminacion_atipicos.csv
  - atipicos_consistentes.csv
  - atipicos_corregidos.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS

RANDOM_STATE = 42
NIVELES_CONTAMINACION = [0.05, 0.10, 0.15, 0.20]

print("=" * 80)
print("SENSIBILIDAD DE LA DETECCION DE HOSPITALES ATIPICOS (Observacion 19)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar matriz, preprocesar y particion Ward K=4
# ---------------------------------------------------------------------------
print("\n[1/5] Cargando matriz institucional y particion Ward K=4...")
matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
ids = matriz["COD_HOSPITAL"].tolist()

feats_clu_cols = [c for c in matriz.columns if c not in COLS_DROP_CLUSTERING]
feats = matriz[feats_clu_cols].copy()
for col in LOG1P_COLS:
    if col in feats.columns:
        feats[col] = np.log1p(feats[col].clip(lower=0))
X = RobustScaler().fit_transform(feats.values)

asig = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
asig["COD_HOSPITAL"] = asig["COD_HOSPITAL"].astype(str)
labels = asig.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int).reindex(ids).values
tamanos_cluster = pd.Series(labels).value_counts().to_dict()
print(f"  X: {X.shape} | tamanos de cluster: {tamanos_cluster}")

nombres_df = pd.read_csv(TABLES_DIR / "hospitales_por_cluster_nivel1.csv", dtype=str)
nombres_df["COD_HOSPITAL"] = nombres_df["COD_HOSPITAL"].astype(str)
nombres_map = nombres_df.set_index("COD_HOSPITAL")["NOMBRE"].to_dict()

# ---------------------------------------------------------------------------
# 2. Isolation Forest global: sensibilidad a la contaminacion
# ---------------------------------------------------------------------------
print(f"\n[2/5] Ejecutando Isolation Forest GLOBAL (ajustado una sola vez sobre "
      f"los 65 hospitales, sin agrupar por cluster) con contamination in "
      f"{NIVELES_CONTAMINACION}...")

resultados_contaminacion = {}
filas_contam = []
for c in NIVELES_CONTAMINACION:
    iso = IsolationForest(contamination=c, random_state=RANDOM_STATE, n_estimators=300)
    pred = iso.fit_predict(X)
    scores = iso.score_samples(X)
    marcados = set(np.array(ids)[pred == -1])
    resultados_contaminacion[c] = marcados
    n_marcados = len(marcados)
    filas_contam.append({
        "contamination": c, "n_marcados_esperado": int(round(c * len(ids))),
        "n_marcados_real": n_marcados,
        "hospitales": ",".join(sorted(marcados)),
    })
    print(f"  contamination={c:.2f}: {n_marcados} hospitales marcados "
          f"(esperado ~{c * len(ids):.1f})")

df_contam = pd.DataFrame(filas_contam)
df_contam.to_csv(TABLES_DIR / "sensibilidad_contaminacion_atipicos.csv", index=False)
print(f"  Persistido: sensibilidad_contaminacion_atipicos.csv")

# Hospitales detectados en TODOS los niveles de contaminacion explorados
interseccion = set.intersection(*resultados_contaminacion.values())
print(f"\n  Hospitales detectados en TODOS los niveles de contaminacion "
      f"({NIVELES_CONTAMINACION}): {len(interseccion)}")
for h in sorted(interseccion):
    print(f"    {h}  {nombres_map.get(h, '?')}")

df_consistentes = pd.DataFrame({"COD_HOSPITAL": sorted(interseccion)})
df_consistentes["NOMBRE"] = df_consistentes["COD_HOSPITAL"].map(nombres_map)
df_consistentes.to_csv(TABLES_DIR / "atipicos_consistentes.csv", index=False)
print(f"  Persistido: atipicos_consistentes.csv")

# ---------------------------------------------------------------------------
# 3. Isolation Forest final con contamination=0.10 (valor usado en el
#    pipeline principal), reportando explicitamente que el ajuste es GLOBAL
# ---------------------------------------------------------------------------
print("\n[3/5] Isolation Forest final (contamination=0.10, ajuste GLOBAL)...")
iso_final = IsolationForest(contamination=0.10, random_state=RANDOM_STATE, n_estimators=300)
iso_pred = iso_final.fit_predict(X)
iso_score = iso_final.score_samples(X)
iso_outlier = iso_pred == -1
print(f"  Marcados (contamination=0.10): {int(iso_outlier.sum())} de {len(ids)}")

# ---------------------------------------------------------------------------
# 4. Criterio de distancia al centroide CORREGIDO: MAD global, no SD por
#    cluster (evita el problema de estimar dispersion en C0 n=3, C3 n=3)
# ---------------------------------------------------------------------------
print("\n[4/5] Recalculando distancia al centroide con MAD global "
      "(no desviacion estandar intra-cluster)...")

dist_centroide = np.zeros(len(ids))
for c in np.unique(labels):
    m = labels == c
    centroide = X[m].mean(axis=0)
    dist_centroide[m] = np.linalg.norm(X[m] - centroide, axis=1)

# MAD GLOBAL de las distancias al centroide (sobre las 65 observaciones, no
# por cluster), para evitar estimar dispersion dentro de grupos de n=3.
mediana_global = np.median(dist_centroide)
mad_global = np.median(np.abs(dist_centroide - mediana_global))
# Factor de consistencia para aproximar la escala de una desviacion estandar
# bajo normalidad (1/Phi^-1(0.75) ~ 1.4826), practica estandar del MAD robusto.
mad_global_escalado = mad_global * 1.4826
z_robusto = (dist_centroide - mediana_global) / mad_global_escalado if mad_global_escalado > 0 else np.zeros(len(ids))
dist_outlier_corregido = z_robusto > 3

print(f"  Mediana global de distancia al centroide: {mediana_global:.3f}")
print(f"  MAD global (escalado): {mad_global_escalado:.3f}")
print(f"  Marcados por distancia (z-robusto > 3, MAD global): "
      f"{int(dist_outlier_corregido.sum())} de {len(ids)}")
for i, h in enumerate(ids):
    if dist_outlier_corregido[i]:
        print(f"    {h}  {nombres_map.get(h, '?')}  cluster={labels[i]}  z_robusto={z_robusto[i]:.2f}")

clusters_pequenios = sorted(
    (int(c), int(n)) for c, n in pd.Series(labels).value_counts().items() if n < 5
)
detalle_pequenios = ", ".join(f"C{c} (n={n})" for c, n in clusters_pequenios)
print(f"\n  [Nota] El criterio original usaba SD intra-cluster: para "
      f"{detalle_pequenios}, esa SD se estima con muy pocas observaciones, "
      f"una base insuficiente. El criterio corregido usa la MAD de las "
      f"{len(ids)} distancias al centroide en conjunto, evitando ese problema.")

# ---------------------------------------------------------------------------
# 5. Silhouette individual + combinacion de los 3 criterios corregidos
# ---------------------------------------------------------------------------
print("\n[5/5] Combinando los tres criterios corregidos y reportando "
      "atipicos robustos (>=2 criterios) vs. por un solo criterio...")

sil_samples = silhouette_samples(X, labels)
sil_negativo = sil_samples < 0

out = pd.DataFrame({
    "COD_HOSPITAL": ids,
    "NOMBRE": [nombres_map.get(h, "?") for h in ids],
    "cluster": labels,
    "n_cluster": [tamanos_cluster[c] for c in labels],
    "silhouette": sil_samples,
    "sil_negativo": sil_negativo,
    "iso_outlier": iso_outlier,
    "iso_score": iso_score,
    "dist_centroide": dist_centroide,
    "z_robusto_mad_global": z_robusto,
    "dist_outlier_corregido": dist_outlier_corregido,
})
out["n_criterios"] = out[["sil_negativo", "iso_outlier", "dist_outlier_corregido"]].sum(axis=1)
out["outlier_robusto"] = out["n_criterios"] >= 2
out["outlier_union"] = out["n_criterios"] >= 1
out = out.sort_values(["n_criterios", "iso_score"], ascending=[False, True])
out.to_csv(TABLES_DIR / "atipicos_corregidos.csv", index=False)
print(f"  Persistido: atipicos_corregidos.csv")

n_robustos = int(out["outlier_robusto"].sum())
n_union = int(out["outlier_union"].sum())
print(f"\n  Atipicos por >=2 criterios (robustos): {n_robustos}")
print(f"  Atipicos por >=1 criterio (union): {n_union}")
if n_robustos > 0:
    print("\n  Hospitales atipicos ROBUSTOS (>=2 criterios):")
    for _, r in out[out["outlier_robusto"]].iterrows():
        crit = []
        if r["sil_negativo"]:
            crit.append("Silhouette<0")
        if r["iso_outlier"]:
            crit.append("IsolationForest")
        if r["dist_outlier_corregido"]:
            crit.append("Dist.Centroide(MAD)")
        print(f"    {r['COD_HOSPITAL']}  {r['NOMBRE']}  [{', '.join(crit)}]")
else:
    print("\n  Ningun hospital satisface 2 o mas criterios simultaneamente con "
          "esta configuracion (los tres criterios capturan facetas distintas "
          "de la atipicidad y rara vez coinciden en el mismo establecimiento).")

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"  Sensibilidad a contaminacion (Isolation Forest global):")
print(df_contam[["contamination", "n_marcados_real"]].to_string(index=False))
print(f"\n  Hospitales detectados en TODOS los niveles de contaminacion: {len(interseccion)}")
print(f"  Atipicos robustos (>=2 criterios corregidos): {n_robustos}")
print(f"  Atipicos por union (>=1 criterio corregido): {n_union}")
print("\nAnalisis completo.")
