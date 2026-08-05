"""Estadistica descriptiva de las variables clinico-operativas (n=65).

Alimenta la Tabla `tab:descriptiva-variables` del capitulo de resultados. Se
genera desde la Hospital_Matrix_Integrada para que la tabla acompañe cualquier
recalculo del pipeline en lugar de quedar fijada de una corrida anterior.

Salida: `reports/tables/estadisticas_descriptivas.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR

# Las componentes PCA y los auxiliares de imputacion no son interpretables aqui.
EXCLUIR = {"COD_HOSPITAL", "peso_medio_cma_imputado"}

matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
cols = [c for c in matriz.columns if c not in EXCLUIR and not c.startswith("dim_")]

desc = matriz[sorted(cols)].describe().T[["mean", "std", "min", "50%", "max"]]

# Orden estable: primero las variables de gestion y diversidad, luego el resto.
ORDEN = [
    "egresos_por_anio", "estancia_media", "estancia_mediana", "peso_medio_grd",
    "severidad_media", "mortalidad_media", "entropia_grd", "comorbilidades_promedio",
    "tasa_cma", "peso_medio_cma", "cv_estancia", "edad_mediana", "pabellones_promedio",
]
resto = [c for c in desc.index if c not in ORDEN]
desc = desc.loc[[c for c in ORDEN if c in desc.index] + sorted(resto)]

desc.to_csv(TABLES_DIR / "estadisticas_descriptivas.csv")
print(f"n hospitales: {len(matriz)} | variables descritas: {len(desc)}")
print(desc.to_string(float_format="%.3f"))
print(f"\n>>> Persistido: {(TABLES_DIR / 'estadisticas_descriptivas.csv').name}")
