"""Simula el error de indicadores según el umbral de cobertura temporal.

Evalúa la justificación del umbral de elegibilidad de al menos tres años:
calcula, para los hospitales con seis años completos, el error relativo de seis
indicadores al estimarlos con subconjuntos de uno a cinco años.

Salidas en ``reports/tables``:
- ``eligibilidad_distribucion_anios.csv``
- ``eligibilidad_error_por_umbral.csv``
- ``eligibilidad_error_detalle.csv``
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR

RANDOM_STATE = 42
N_REPETICIONES_MAX = 200
K_VALORES = [1, 2, 3, 4, 5]
ANIOS_TOTALES = [2019, 2020, 2021, 2022, 2023, 2024]
VARIABLES_CLAVE = [
    "egresos_por_anio",
    "estancia_media",
    "peso_medio_grd",
    "severidad_media",
    "mortalidad_media",
    "tasa_cma",
]
rng = np.random.default_rng(RANDOM_STATE)

print("=" * 80)
print("SIMULACIÓN: error de estimación según años de cobertura")
print("=" * 80)

print("\n[1/5] Cargando base pre-elegibilidad...")
df_full = pd.read_parquet(
    PROCESSED_DIR / "grd_filtrado.parquet",
    columns=[
        "COD_HOSPITAL", "anio", "MODALIDAD", "DIAS_ESTADA", "IR_29301_PESO",
        "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD",
    ],
)
df_full["COD_HOSPITAL"] = df_full["COD_HOSPITAL"].astype(str)
df_hosp_full = df_full[df_full["MODALIDAD"] == "HOSPITALIZACION"]
anios_por_hosp = df_hosp_full.groupby("COD_HOSPITAL")["anio"].nunique()
distribucion = anios_por_hosp.value_counts().sort_index()
distribucion.rename_axis("anios_presentes").reset_index(name="n_hospitales").to_csv(
    TABLES_DIR / "eligibilidad_distribucion_anios.csv", index=False
)
hospitales_completos = sorted(anios_por_hosp[anios_por_hosp == len(ANIOS_TOTALES)].index)
print(distribucion.to_string())
print(f"Hospitales con cobertura completa: {len(hospitales_completos)}")

print("\n[2/5] Pre-agregando estadísticos por hospital y año...")
df_completo = df_hosp_full[df_hosp_full["COD_HOSPITAL"].isin(hospitales_completos)]
agg_hosp = df_completo.groupby(["COD_HOSPITAL", "anio"]).agg(
    n=("DIAS_ESTADA", "size"),
    suma_estancia=("DIAS_ESTADA", "sum"),
    suma_peso=("IR_29301_PESO", "sum"),
    suma_severidad=("IR_29301_SEVERIDAD", "sum"),
    suma_mortalidad=("IR_29301_MORTALIDAD", "sum"),
)
agg_cma = (
    df_full[df_full["COD_HOSPITAL"].isin(hospitales_completos)]
    .groupby(["COD_HOSPITAL", "anio", "MODALIDAD"])
    .size()
    .unstack(fill_value=0)
)
for modalidad in ("HOSPITALIZACION", "CMA"):
    if modalidad not in agg_cma.columns:
        agg_cma[modalidad] = 0
agg_cma = agg_cma.rename(columns={"HOSPITALIZACION": "n_hosp", "CMA": "n_cma"})[["n_hosp", "n_cma"]]


def indicadores(codigo_hospital: str, anios: tuple[int, ...]) -> dict[str, float]:
    indices = [(codigo_hospital, anio) for anio in anios]
    disponibles = [indice for indice in indices if indice in agg_hosp.index]
    if not disponibles:
        return {variable: np.nan for variable in VARIABLES_CLAVE}
    hospital = agg_hosp.loc[disponibles]
    total = hospital["n"].sum()
    cma_disponibles = [indice for indice in indices if indice in agg_cma.index]
    cma = agg_cma.loc[cma_disponibles] if cma_disponibles else pd.DataFrame(columns=["n_hosp", "n_cma"])
    n_hosp = cma["n_hosp"].sum() if not cma.empty else 0
    n_cma = cma["n_cma"].sum() if not cma.empty else 0
    return {
        "egresos_por_anio": total / len(anios),
        "estancia_media": hospital["suma_estancia"].sum() / total,
        "peso_medio_grd": hospital["suma_peso"].sum() / total,
        "severidad_media": hospital["suma_severidad"].sum() / total,
        "mortalidad_media": hospital["suma_mortalidad"].sum() / total,
        "tasa_cma": n_cma / (n_hosp + n_cma) if (n_hosp + n_cma) else 0.0,
    }

print("[3/5] Calculando referencia con los seis años completos...")
referencia = pd.DataFrame(
    {hospital: indicadores(hospital, tuple(ANIOS_TOTALES)) for hospital in hospitales_completos}
).T
referencia.index.name = "COD_HOSPITAL"

print("[4/5] Calculando error relativo por subconjunto anual...")
filas_detalle: list[tuple[object, ...]] = []
for k in K_VALORES:
    combinaciones = list(itertools.combinations(ANIOS_TOTALES, k))
    if len(combinaciones) > N_REPETICIONES_MAX:
        seleccion = rng.choice(len(combinaciones), size=N_REPETICIONES_MAX, replace=False)
        combinaciones = [combinaciones[indice] for indice in seleccion]
    for rep_id, anios in enumerate(combinaciones):
        for hospital in hospitales_completos:
            estimado = indicadores(hospital, anios)
            for variable in VARIABLES_CLAVE:
                valor_referencia = referencia.loc[hospital, variable]
                valor_estimado = estimado[variable]
                if pd.isna(valor_referencia) or pd.isna(valor_estimado) or valor_referencia == 0:
                    continue
                filas_detalle.append((k, rep_id, hospital, variable, abs(valor_estimado - valor_referencia) / abs(valor_referencia)))
    print(f"  k={k}: {len(combinaciones)} combinaciones evaluadas")

detalle = pd.DataFrame(
    filas_detalle,
    columns=["k_anios", "rep_id", "COD_HOSPITAL", "variable", "error_relativo"],
)
detalle.to_csv(TABLES_DIR / "eligibilidad_error_detalle.csv", index=False)

print("[5/5] Resumiendo error mediano e intervalos bootstrap...")
def intervalo_bootstrap(valores: np.ndarray, n_bootstrap: int = 2000) -> tuple[float, float]:
    indices = rng.integers(0, len(valores), size=(n_bootstrap, len(valores)))
    medianas = np.median(valores[indices], axis=1)
    return float(np.percentile(medianas, 2.5)), float(np.percentile(medianas, 97.5))

filas_resumen: list[dict[str, object]] = []
for k in K_VALORES:
    for variable in VARIABLES_CLAVE:
        valores = detalle.loc[(detalle["k_anios"] == k) & (detalle["variable"] == variable), "error_relativo"].to_numpy()
        if not len(valores):
            continue
        ic_bajo, ic_alto = intervalo_bootstrap(valores)
        filas_resumen.append({
            "k_anios": k,
            "variable": variable,
            "error_rel_mediano_pct": float(np.median(valores) * 100),
            "ic95_lo_pct": ic_bajo * 100,
            "ic95_hi_pct": ic_alto * 100,
            "n_observaciones": len(valores),
        })
resumen = pd.DataFrame(filas_resumen)
resumen.to_csv(TABLES_DIR / "eligibilidad_error_por_umbral.csv", index=False)

print("\nError relativo mediano promedio por umbral:")
print(resumen.groupby("k_anios")["error_rel_mediano_pct"].mean().round(2).to_string())
print("\nAnálisis completo.")
