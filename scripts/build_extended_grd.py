"""ETL extendido: re-lee raw GRD con columnas adicionales y persiste grd_filtrado.parquet enriquecido.

Variables nuevas que se incorporan al `grd_filtrado.parquet`:

  Demograficas del paciente:
    - SEXO
    - FECHA_NACIMIENTO -> EDAD (al ingreso)

  Origen y modalidad del egreso:
    - TIPO_PROCEDENCIA (origen del paciente)
    - TIPO_INGRESO (urgencia / programada / otro)
    - TIPOALTA (domicilio / traslado / fallecido / otro)
    - ESPECIALIDAD_MEDICA

  Atencion obstetrica / neonatal:
    - CONDICIONDEALTANEONATO1 (presencia de parto)
    - PESORN1 (peso recien nacido)

  Pabellon y procedimientos:
    - USOSPABELLON

Conserva: COD_HOSPITAL, SERVICIO_SALUD, IR_29301_*, FECHA_INGRESO, FECHAALTA,
TIPO_ACTIVIDAD, DIAGNOSTICO1..35, PROCEDIMIENTO1..30, anio.

Los campos de codificación se cargan como ``string[pyarrow]`` para conservar
ceros a la izquierda y contener el consumo de memoria al procesar los 65 campos.

El archivo se sobrescribe al reconstruir el conjunto GRD enriquecido.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Encodings y separadores por anio (verificados en notebook 01)
YEAR_CONFIGS = {
    2019: {"encoding": "utf-8", "sep": "|"},
    2020: {"encoding": "utf-8", "sep": "|"},
    2021: {"encoding": "utf-8", "sep": "|"},
    2022: {"encoding": "utf-16", "sep": "|"},
    2023: {"encoding": "utf-16", "sep": "|"},
    2024: {"encoding": "iso-8859-1", "sep": "|"},
}

# Columnas a cargar del raw GRD (extendido)
USECOLS_EXTENDIDO = [
    # Identificacion
    "COD_HOSPITAL", "SERVICIO_SALUD",
    # GRD
    "IR_29301_COD_GRD", "IR_29301_PESO", "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD",
    # Fechas y modalidad
    "FECHA_INGRESO", "FECHAALTA", "TIPO_ACTIVIDAD",
    # Demograficas
    "SEXO", "FECHA_NACIMIENTO",
    # Origen y modalidad del egreso
    "TIPO_PROCEDENCIA", "TIPO_INGRESO", "TIPOALTA", "ESPECIALIDAD_MEDICA",
    # Atencion obstetrica / neonatal
    "CONDICIONDEALTANEONATO1", "PESORN1",
    # Pabellon
    "USOSPABELLON",
    # Codificación clínica completa: 35 diagnósticos CIE-10 y 30 procedimientos CIE-9-MC.
    *[f"DIAGNOSTICO{i}" for i in range(1, 36)],
    *[f"PROCEDIMIENTO{i}" for i in range(1, 31)],
]


def load_year(year: int, raw_dir: Path, config: dict, usecols: list[str]) -> pd.DataFrame:
    """Carga el archivo GRD de un anio con su configuracion."""
    candidates = list(raw_dir.glob(f"*{year}*.txt"))
    if not candidates:
        print(f"[AVISO] no se encontro archivo para {year}")
        return pd.DataFrame()
    filepath = candidates[0]
    print(f"  Cargando {year}: {filepath.name}", flush=True)

    # Verificar columnas presentes
    header = pd.read_csv(
        filepath, sep=config["sep"], encoding=config["encoding"], nrows=0,
    ).columns.str.strip().str.upper().str.replace(" ", "_").tolist()

    cols_to_load = [c for c in usecols if c in header]
    missing = sorted(set(usecols) - set(cols_to_load))
    if missing:
        print(f"    cols ausentes en {year}: {missing}", flush=True)

    # Arrow conserva los códigos como texto y reduce el costo de memoria de los
    # 65 campos clínicos frente a columnas ``object`` de Python.
    df = pd.read_csv(
        filepath, sep=config["sep"], encoding=config["encoding"],
        low_memory=False,
        dtype={col: "string[pyarrow]" for col in cols_to_load},
        usecols=cols_to_load,
    )
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    # Homologar identificador de paciente del 2024 (no lo necesitamos pero por consistencia)
    if year == 2024 and "ID_BENEFICIARIO" in df.columns:
        df.rename(columns={"ID_BENEFICIARIO": "CIP_ENCRIPTADO"}, inplace=True)

    df["anio"] = year
    return df


# ---------------------------------------------------------------------------
# 1. Cargar todos los anios
# ---------------------------------------------------------------------------
print("=" * 70)
print("ETL EXTENDIDO: re-lectura raw con variables adicionales")
print("=" * 70)

t0 = time.perf_counter()
dfs = {}
for year, cfg in YEAR_CONFIGS.items():
    df = load_year(year, RAW_DIR, cfg, USECOLS_EXTENDIDO)
    if not df.empty:
        dfs[year] = df
        print(f"    -> {len(df):,} egresos | {df.shape[1]} cols | "
              f"{df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

print(f"\nLectura total: {time.perf_counter() - t0:.1f}s")

grd_raw = pd.concat(dfs.values(), ignore_index=True, sort=False)
print(f"Concatenado: {len(grd_raw):,} egresos | {grd_raw.shape[1]} cols")
del dfs


# ---------------------------------------------------------------------------
# 2. Filtrar por TIPO_ACTIVIDAD (mismo criterio que ETL original)
# ---------------------------------------------------------------------------
TIPOS_HOSP = [
    "HOSPITALIZACIÓN",
    "HOSPITALIZACIÓN EN URGENCIA",
    "HOSPITALIZACIÓN DIURNA",
]
TIPO_CMA = "CIRUGÍA MAYOR AMBULATORIA (CMA)"
is_hosp = grd_raw["TIPO_ACTIVIDAD"].isin(TIPOS_HOSP)
is_cma = grd_raw["TIPO_ACTIVIDAD"] == TIPO_CMA
mask = is_hosp | is_cma

grd = grd_raw[mask].copy()
grd["MODALIDAD"] = np.where(is_cma[mask], "CMA", "HOSPITALIZACION")
print(f"\nFiltrado por TIPO_ACTIVIDAD: {len(grd):,} egresos "
      f"(HOSP={(grd['MODALIDAD']=='HOSPITALIZACION').sum():,}, "
      f"CMA={(grd['MODALIDAD']=='CMA').sum():,})")
del grd_raw

# Eliminar registros sin hospital
grd = grd[grd["COD_HOSPITAL"].notna() & (grd["COD_HOSPITAL"].astype(str).str.strip() != "")]


# ---------------------------------------------------------------------------
# 3. Parsear fechas (manejo formato 2023 DD-MM-YYYY)
# ---------------------------------------------------------------------------
print("\nParseando fechas...")
mask_2023 = grd["anio"] == 2023
for col_raw, col_dt in [
    ("FECHA_INGRESO", "FECHA_INGRESO_dt"),
    ("FECHAALTA", "FECHAALTA_dt"),
    ("FECHA_NACIMIENTO", "FECHA_NACIMIENTO_dt"),
]:
    if col_raw not in grd.columns:
        grd[col_dt] = pd.NaT
        continue
    grd[col_dt] = pd.NaT
    if mask_2023.any():
        grd.loc[mask_2023, col_dt] = pd.to_datetime(
            grd.loc[mask_2023, col_raw], format="mixed", dayfirst=True, errors="coerce",
        )
    grd.loc[~mask_2023, col_dt] = pd.to_datetime(
        grd.loc[~mask_2023, col_raw], format="mixed", dayfirst=False, errors="coerce",
    )

grd["DIAS_ESTADA"] = (grd["FECHAALTA_dt"] - grd["FECHA_INGRESO_dt"]).dt.days.clip(lower=0)


# ---------------------------------------------------------------------------
# 4. Calcular EDAD al ingreso
# ---------------------------------------------------------------------------
print("Calculando edad al ingreso...")
grd["EDAD"] = (
    (grd["FECHA_INGRESO_dt"] - grd["FECHA_NACIMIENTO_dt"]).dt.days / 365.25
).clip(lower=0, upper=120)


# ---------------------------------------------------------------------------
# 5. Convertir numericas (peso, severidad, mortalidad, peso RN)
# ---------------------------------------------------------------------------
print("Convirtiendo numericas...")
for col in ["IR_29301_PESO", "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD"]:
    grd[col] = pd.to_numeric(
        grd[col].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )
if "PESORN1" in grd.columns:
    grd["PESORN1"] = pd.to_numeric(
        grd["PESORN1"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )


# ---------------------------------------------------------------------------
# 6. Normalizar TIPO_INGRESO y TIPOALTA (categorias clave)
# ---------------------------------------------------------------------------
print("Normalizando categoricas...")
def norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().str.replace(" ", "_")

if "TIPO_INGRESO" in grd.columns:
    grd["TIPO_INGRESO_N"] = norm_str(grd["TIPO_INGRESO"])
if "TIPOALTA" in grd.columns:
    grd["TIPOALTA_N"] = norm_str(grd["TIPOALTA"])
if "TIPO_PROCEDENCIA" in grd.columns:
    grd["TIPO_PROCEDENCIA_N"] = norm_str(grd["TIPO_PROCEDENCIA"])

# USOSPABELLON: "1" o "0" ; convertir a numerico
if "USOSPABELLON" in grd.columns:
    grd["USOSPABELLON_N"] = pd.to_numeric(grd["USOSPABELLON"], errors="coerce").fillna(0).astype(int)


# ---------------------------------------------------------------------------
# 7. Persistir
# ---------------------------------------------------------------------------
out_path = PROCESSED_DIR / "grd_filtrado.parquet"
print(f"\nPersistiendo en {out_path.name} ...")

# Drop de columnas raw redundantes para reducir tamanio
cols_drop = ["FECHA_INGRESO", "FECHAALTA", "FECHA_NACIMIENTO"]
grd_persist = grd.drop(columns=[c for c in cols_drop if c in grd.columns])

grd_persist.to_parquet(out_path, index=False, compression="zstd")
print(f"Tamanio: {out_path.stat().st_size / 1e6:.0f} MB")
print(f"Shape final: {grd_persist.shape}")
print(f"Columnas: {sorted(grd_persist.columns.tolist())}")
print(f"\nETL extendido completo en {time.perf_counter() - t0:.1f}s")
