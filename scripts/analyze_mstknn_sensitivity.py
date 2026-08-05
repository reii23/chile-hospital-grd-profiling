"""Barrido de sensibilidad MST-kNN en modo mutuo sobre la matriz institucional.

Reemplaza al antiguo `aplicar_mstknn.py`, que leia artefactos de una generacion
previa del pipeline (`hospital_matrix_integrada.parquet` y
`subcluster_sin_K_ari_cruzado.csv`) que ya no existen.

MST-kNN no fija el numero de grupos: la granularidad depende de la definicion
local de vecindad, controlada por `k`. Este script recorre k = 3..8 en modo
mutuo sobre la MISMA matriz y el MISMO preprocesamiento que usa Ward
(log1p en las variables de escala + RobustScaler), de modo que la comparacion
con la particion reportada sea directa.

Salidas
-------
  - reports/tables/mstknn_sensibilidad.csv   (una fila por valor de k)

El Silhouette solo se calcula cuando la solucion tiene al menos dos grupos y
no es degenerada; con muchos grupos unitarios su interpretacion es limitada,
por lo que se reporta como referencia y no como criterio de seleccion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.mstknn import MSTkNN
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import PROCESSED_DIR, TABLES_DIR

VALORES_K = range(3, 9)

print("=" * 80)
print("SENSIBILIDAD MST-kNN (modo mutuo) SOBRE LA MATRIZ INSTITUCIONAL")
print("=" * 80)

matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
features = matriz.drop(
    columns=[c for c in COLS_DROP_CLUSTERING if c in matriz.columns]
).fillna(0.0)
for col in LOG1P_COLS:
    if col in features.columns:
        features[col] = np.log1p(features[col].clip(lower=0))
X = RobustScaler().fit_transform(features)
print(f"\nMatriz preprocesada: {X.shape[0]} hospitales x {X.shape[1]} variables")

filas = []
for k in VALORES_K:
    labels, info = MSTkNN(k=k, metric="euclidean", knn_simetrico=True).fit_predict(X)
    tamanos = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
    grandes = [t for t in tamanos if t >= 3]
    pares = [t for t in tamanos if t == 2]
    unitarios = [t for t in tamanos if t == 1]

    sil = np.nan
    if 1 < info["K_descubierto"] < X.shape[0]:
        sil = float(silhouette_score(X, labels))

    filas.append({
        "k": k,
        "K_total": info["K_descubierto"],
        "n_grupos_ge3": len(grandes),
        "n_pares": len(pares),
        "n_unitarios": len(unitarios),
        "tamano_bloque_mayor": tamanos[0],
        "tamanos_grupos_ge3": str(grandes),
        "silhouette": sil,
        "n_aristas_interseccion": info["n_aristas_interseccion"],
        "ratio_aristas": round(info["ratio_aristas"], 4),
    })
    print(f"  k={k}: K_total={info['K_descubierto']:>3}  grupos n>=3={len(grandes)}  "
          f"pares={len(pares)}  unitarios={len(unitarios)}  "
          f"bloque_mayor={tamanos[0]:>2}  silhouette={sil:.4f}")

df = pd.DataFrame(filas)
salida = TABLES_DIR / "mstknn_sensibilidad.csv"
df.to_csv(salida, index=False)
print(f"\n>>> Persistido: {salida.name}")

mejor = df.loc[df["silhouette"].idxmax()]
print(f"\nMayor Silhouette del barrido: k={int(mejor['k'])} "
      f"(silhouette={mejor['silhouette']:.4f}, K_total={int(mejor['K_total'])}, "
      f"unitarios={int(mejor['n_unitarios'])})")
print("\nNinguna configuracion reproduce una particion compacta de cuatro grupos "
      "comparable directamente con Ward.")
print("\nAnalisis completo.")
