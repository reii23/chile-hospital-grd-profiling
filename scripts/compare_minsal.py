"""Compara el clustering Ward final con la clasificación de complejidad MINSAL.

Compara la particion obtenida (clustering no supervisado sobre datos GRD)
contra la etiqueta administrativa de Nivel de Complejidad del MINSAL
(Alta / Mediana / Baja), que es la referencia externa.

Metricas:
  - ARI (Adjusted Rand Index): acuerdo global ajustado al azar.
  - NMI (Normalized Mutual Information).
  - Jaccard cluster-a-cluster + tabla de contingencia.

Se reporta para el corte principal Ward K=4. El clustering tiene mas grupos
que MINSAL (3 niveles), por lo que la lectura correcta es: cuanto de la
estructura MINSAL recupera el clustering y como se reparte cada nivel.

Fuente MINSAL: info-hospitales/Base de Establecimientos 2023.xlsx
  columna 'Codigo Vigente' (cruza 65/65) + 'Nivel de Complejidad'.

Outputs:
- reports/tables/comparacion_minsal_contingencia.csv
- reports/tables/comparacion_minsal_jaccard.csv
- reports/tables/comparacion_minsal_metricas.csv
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

from src.modeling.referencia import COLS_DROP_CLUSTERING as COLS_DROP

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
BASE_EST = ROOT / "info-hospitales" / "Base de Establecimientos 2023.xlsx"

# ---------------------------------------------------------------------------
# 1. Reconstruir particion Ward K=4 (mismo preprocesamiento que build_dataset.py)
# ---------------------------------------------------------------------------
matriz = pd.read_parquet(PROCESSED / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)

feats = matriz.drop(columns=[c for c in COLS_DROP if c in matriz.columns]).copy()
feats.index = matriz["COD_HOSPITAL"].values
feats = feats.fillna(0.0)
for col in ["egresos_por_anio", "pabellones_promedio"]:
    if col in feats.columns:
        feats[col] = np.log1p(feats[col].clip(lower=0))

X = RobustScaler().fit_transform(feats.values)
Z = linkage(X, method="ward")
labels_k4 = fcluster(Z, t=4, criterion="maxclust") - 1

clusters = pd.DataFrame({
    "COD_HOSPITAL": feats.index.astype(str),
    "cluster": [f"C{c}" for c in labels_k4],
})

# ---------------------------------------------------------------------------
# 2. Cargar etiqueta MINSAL
# ---------------------------------------------------------------------------
base = pd.read_excel(BASE_EST, skiprows=1)
base.columns = [str(c).strip() for c in base.columns]
base["COD_HOSPITAL"] = (
    base["Código Vigente"].astype(str).str.replace(".0", "", regex=False).str.strip()
)
minsal = (
    base.dropna(subset=["Nivel de Complejidad"])
    .drop_duplicates("COD_HOSPITAL")
    .set_index("COD_HOSPITAL")["Nivel de Complejidad"]
)

df = clusters.copy()
df["minsal"] = df["COD_HOSPITAL"].map(minsal)
sin_etiqueta = df["minsal"].isna().sum()
if sin_etiqueta:
    print(f"[aviso] {sin_etiqueta} hospitales sin etiqueta MINSAL (se excluyen del calculo)")
df = df.dropna(subset=["minsal"]).copy()

print(f"Hospitales comparados: {len(df)}")
print("Distribucion MINSAL:", df["minsal"].value_counts().to_dict())
print("Distribucion clusters:", df["cluster"].value_counts().sort_index().to_dict())
print()

# ---------------------------------------------------------------------------
# 3. Tabla de contingencia cluster x MINSAL
# ---------------------------------------------------------------------------
cont = pd.crosstab(df["cluster"], df["minsal"])
cont.to_csv(TABLES / "comparacion_minsal_contingencia.csv")
print("=" * 70)
print("Tabla de contingencia: cluster (filas) x Nivel MINSAL (columnas)")
print("=" * 70)
print(cont.to_string())
print()

# ---------------------------------------------------------------------------
# 4. Jaccard cluster-a-nivel
# ---------------------------------------------------------------------------
ga = sorted(df["cluster"].unique())
gb = sorted(df["minsal"].unique())
J = pd.DataFrame(index=ga, columns=gb, dtype=float)
a = df["cluster"].values
b = df["minsal"].values
for ca in ga:
    sa = set(np.where(a == ca)[0])
    for cb in gb:
        sb = set(np.where(b == cb)[0])
        union = len(sa | sb)
        J.loc[ca, cb] = (len(sa & sb) / union) if union else 0.0
J.to_csv(TABLES / "comparacion_minsal_jaccard.csv")
print("=" * 70)
print("Jaccard cluster-a-nivel MINSAL")
print("=" * 70)
print(J.to_string(float_format="%.2f"))
print()

# ---------------------------------------------------------------------------
# 5. Metricas globales (ARI, NMI)
# ---------------------------------------------------------------------------
ari = adjusted_rand_score(df["minsal"], df["cluster"])
nmi = normalized_mutual_info_score(df["minsal"], df["cluster"])

# Jaccard medio: para cada nivel MINSAL, mejor cluster solapado
jacc_por_nivel = {cb: J[cb].max() for cb in gb}
jacc_medio = float(np.mean(list(jacc_por_nivel.values())))

met = pd.DataFrame([{
    "comparacion": "Ward K=4 vs MINSAL (Alta/Mediana/Baja)",
    "n": len(df),
    "ARI": ari,
    "NMI": nmi,
    "jaccard_medio_por_nivel": jacc_medio,
}])
met.to_csv(TABLES / "comparacion_minsal_metricas.csv", index=False)

print("=" * 70)
print("Metricas globales clustering vs MINSAL")
print("=" * 70)
print(f"  ARI = {ari:.3f}")
print(f"  NMI = {nmi:.3f}")
print(f"  Jaccard medio por nivel MINSAL = {jacc_medio:.3f}")
print("  Jaccard por nivel:", {k: round(v, 2) for k, v in jacc_por_nivel.items()})
print()
print(">>> Tablas guardadas en reports/tables/")
