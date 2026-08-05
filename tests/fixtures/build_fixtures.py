"""Generador determinista de fixtures sintéticas para tests.

Ejecutar una sola vez (o cuando cambien los esquemas):
    .venv/bin/python tests/fixtures/build_fixtures.py

Genera:
- tests/fixtures/grd_sample.csv          — 1.000 filas (250/año × 4 hospitales sintéticos)
- tests/fixtures/cie10_master_mini.xlsx  — 50 códigos repartidos en 5 capítulos
- tests/fixtures/cie9_master_mini.xlsx   — 30 códigos repartidos en 4 secciones

Las fixtures son DETERMINISTAS (semilla fija = 42) para reproducibilidad bit-a-bit
en CI (Req 13.5, Property 9).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent
SEED = 42


# ------------------------------------------------------------------ CIE-10 mini
def build_cie10_master_mini() -> pd.DataFrame:
    """50 códigos CIE-10 repartidos en 5 capítulos (10 códigos cada uno).

    Replicamos exactamente el esquema de la tabla maestra real:
    columnas = ['Versión', 'Código', 'Descripción', 'Categoría', 'Sección', 'Capítulo']
    """
    capitulos = [
        ("Cap.01  CIERTAS ENFERMEDADES INFECCIOSAS Y PARASITARIAS (A00-B99)",
         [f"A{i:02d}" for i in range(0, 10)],
         "INFECCIONES INTESTINALES"),
        ("Cap.02  NEOPLASIAS (C00-D49)",
         [f"C{i:02d}" for i in range(0, 10)],
         "NEOPLASIAS MALIGNAS"),
        ("Cap.09  ENFERMEDADES DEL APARATO CIRCULATORIO (I00-I99)",
         [f"I{i:02d}" for i in range(0, 10)],
         "ENFERMEDADES HIPERTENSIVAS"),
        ("Cap.15  EMBARAZO, PARTO Y PUERPERIO (O00-O9A)",
         [f"O{i:02d}" for i in range(0, 10)],
         "EMBARAZO TERMINADO EN ABORTO"),
        ("Cap.19  LESIONES TRAUMÁTICAS, ENVENENAMIENTOS Y OTRAS CONSECUENCIAS DE CAUSAS EXTERNAS (S00-T88)",
         [f"S{i:02d}" for i in range(0, 10)],
         "TRAUMATISMOS DE LA CABEZA"),
    ]

    rows = []
    for capitulo, codigos, seccion_label in capitulos:
        for cod in codigos:
            rows.append({
                "Versión": "CIE-v2013",
                "Código": cod,
                "Descripción": f"Descripción de {cod}",
                "Categoría": f"{cod} CATEGORIA EJEMPLO",
                "Sección": f"{cod[0]}00-{cod[0]}09  {seccion_label}",
                "Capítulo": capitulo,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ CIE-9 mini
def build_cie9_master_mini() -> pd.DataFrame:
    """30 códigos CIE-9-MC repartidos en 4 secciones.

    Esquema real: ['Código', 'Descripción', 'Categoría', 'Sección', 'Capítulo']
    En la tabla real, Sección y Capítulo coinciden — replicamos esa convención.
    """
    secciones = [
        ("00. (00 00) PROCEDIMIENTOS E INTERVENCIONES NO CLASIFICADOS BAJO OTROS CONCEPTOS",
         [(f"00.{i:02d}", "ULTRASONIDO TERAPEUTICO") for i in range(1, 9)]),
        ("03. (08 16) OPERACIONES SOBRE EL OJO",
         [(f"{i}.{j:02d}", "PROCEDIMIENTO OFTALMOLOGICO")
          for i in range(8, 17) for j in range(0, 1)]),
        ("07. (35 39) OPERACIONES SOBRE EL APARATO CARDIOVASCULAR",
         [(f"{i}.{j:02d}", "PROCEDIMIENTO CARDIOVASCULAR")
          for i in range(35, 40) for j in range(0, 1)]),
        ("13. (72 75) PROCEDIMIENTOS OBSTETRICOS",
         [(f"{i}.{j:02d}", "PROCEDIMIENTO OBSTETRICO")
          for i in range(72, 76) for j in range(0, 2)]),
    ]

    rows = []
    for seccion, codigos in secciones:
        for cod, desc in codigos:
            rows.append({
                "Código": cod,
                "Descripción": desc,
                "Categoría": f"{cod[:2]} CATEGORIA EJEMPLO",
                "Sección": seccion,
                "Capítulo": seccion,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ GRD raw
def build_grd_sample(rng: np.random.Generator) -> pd.DataFrame:
    """1.000 filas: 4 hospitales × 6 años × ~42 egresos/año, mezcla
    HOSPITALIZACIÓN/CMA/HOSPITALIZACIÓN EN URGENCIA/HOSPITALIZACIÓN DIURNA.

    Distribuciones:
    - HOSP_001: predominio circulatorio (cap_09)
    - HOSP_002: predominio obstétrico (cap_15)
    - HOSP_003: predominio infeccioso (cap_01)
    - HOSP_004: hospital con alta tasa de CMA + perfil diverso
    """
    hospitales = ["HOSP_001", "HOSP_002", "HOSP_003", "HOSP_004"]
    anios = [2019, 2020, 2021, 2022, 2023, 2024]

    # Códigos CIE-10 por capítulo (deben coincidir con build_cie10_master_mini)
    codigos_por_cap = {
        "01": [f"A{i:02d}" for i in range(0, 10)],
        "02": [f"C{i:02d}" for i in range(0, 10)],
        "09": [f"I{i:02d}" for i in range(0, 10)],
        "15": [f"O{i:02d}" for i in range(0, 10)],
        "19": [f"S{i:02d}" for i in range(0, 10)],
    }

    # Códigos CIE-9-MC por sección (deben coincidir con build_cie9_master_mini)
    codigos_proc_por_sec = {
        "00": [f"00.{i:02d}" for i in range(1, 9)],
        "ojo": [f"{i}.00" for i in range(8, 17)],
        "cv": [f"{i}.00" for i in range(35, 40)],
        "obst": [f"{i}.{j:02d}" for i in range(72, 76) for j in range(0, 2)],
    }

    # Perfil por hospital: probabilidades de cada capítulo en DIAGNOSTICO1
    perfiles = {
        "HOSP_001": {"01": 0.10, "02": 0.10, "09": 0.55, "15": 0.05, "19": 0.20},
        "HOSP_002": {"01": 0.05, "02": 0.05, "09": 0.10, "15": 0.65, "19": 0.15},
        "HOSP_003": {"01": 0.55, "02": 0.10, "09": 0.10, "15": 0.10, "19": 0.15},
        "HOSP_004": {"01": 0.20, "02": 0.20, "09": 0.20, "15": 0.20, "19": 0.20},
    }

    # Tasa de CMA por hospital
    tasa_cma = {"HOSP_001": 0.10, "HOSP_002": 0.05, "HOSP_003": 0.08, "HOSP_004": 0.30}

    grds_top = ["540001", "540002", "540003", "1234", "5678", "9012", "3456", "7890"]

    rows = []
    egresos_por_hosp_anio = 42  # ~42 × 6 años × 4 hospitales = 1008 ≈ 1000

    for hosp in hospitales:
        cma_rate = tasa_cma[hosp]
        for anio in anios:
            for _ in range(egresos_por_hosp_anio):
                # Decidir modalidad
                es_cma = rng.random() < cma_rate
                if es_cma:
                    tipo = "CIRUGÍA MAYOR AMBULATORIA (CMA)"
                else:
                    tipo = rng.choice(
                        ["HOSPITALIZACIÓN", "HOSPITALIZACIÓN EN URGENCIA",
                         "HOSPITALIZACIÓN DIURNA"],
                        p=[0.85, 0.10, 0.05],
                    )

                # Diagnóstico principal según perfil
                cap = rng.choice(
                    list(perfiles[hosp].keys()),
                    p=list(perfiles[hosp].values()),
                )
                diag1 = rng.choice(codigos_por_cap[cap])

                # Diagnósticos secundarios (cobertura decreciente)
                diags = [diag1]
                for k in range(2, 6):
                    cobertura = {2: 0.85, 3: 0.65, 4: 0.50, 5: 0.35}[k]
                    if rng.random() < cobertura:
                        cap_k = rng.choice(list(perfiles[hosp].keys()))
                        diags.append(rng.choice(codigos_por_cap[cap_k]))
                    else:
                        diags.append(None)

                # Procedimientos según modalidad
                procs = [None] * 5
                if es_cma:
                    procs[0] = rng.choice(codigos_proc_por_sec["ojo"])
                else:
                    # ~70% de hospitalizaciones tienen al menos un procedimiento
                    if rng.random() < 0.70:
                        if cap == "09":
                            procs[0] = rng.choice(codigos_proc_por_sec["cv"])
                        elif cap == "15":
                            procs[0] = rng.choice(codigos_proc_por_sec["obst"])
                        else:
                            procs[0] = rng.choice(codigos_proc_por_sec["00"])
                        for k in range(1, 5):
                            if rng.random() < 0.30:
                                procs[k] = rng.choice(codigos_proc_por_sec["00"])

                # Fechas
                mes = rng.integers(1, 13)
                dia = rng.integers(1, 28)
                fecha_ing = f"{anio}-{mes:02d}-{dia:02d}"
                if es_cma:
                    fecha_alta = fecha_ing  # CMA: mismo día
                    dias_estada = 0
                else:
                    dias_estada = int(rng.poisson(5) + 1)
                    fa_dt = pd.Timestamp(fecha_ing) + pd.Timedelta(days=dias_estada)
                    fecha_alta = fa_dt.strftime("%Y-%m-%d")

                # Métricas GRD
                if es_cma:
                    peso = round(rng.uniform(0.3, 0.7), 4)
                    sev = 0
                    mort = 0
                else:
                    peso = round(rng.uniform(0.5, 2.5), 4)
                    sev = int(rng.integers(1, 5))
                    mort = int(rng.integers(1, 5))

                row = {
                    "COD_HOSPITAL": hosp,
                    "SERVICIO_SALUD": f"SS_{hosp[-1]}",
                    "IR_29301_COD_GRD": rng.choice(grds_top),
                    "IR_29301_PESO": str(peso).replace(".", ","),  # coma decimal
                    "IR_29301_SEVERIDAD": str(sev),
                    "IR_29301_MORTALIDAD": str(mort),
                    "FECHA_INGRESO": fecha_ing,
                    "FECHAALTA": fecha_alta,
                    "TIPO_ACTIVIDAD": tipo,
                    "anio": anio,
                }
                for k, diag in enumerate(diags, start=1):
                    row[f"DIAGNOSTICO{k}"] = diag
                for k, proc in enumerate(procs, start=1):
                    row[f"PROCEDIMIENTO{k}"] = proc

                rows.append(row)

    df = pd.DataFrame(rows)
    # Garantizar orden de columnas estable (necesario para tests determinísticos)
    cols_orden = [
        "COD_HOSPITAL", "SERVICIO_SALUD", "IR_29301_COD_GRD",
        "IR_29301_PESO", "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD",
        "FECHA_INGRESO", "FECHAALTA", "TIPO_ACTIVIDAD", "anio",
    ]
    cols_orden += [f"DIAGNOSTICO{k}" for k in range(1, 6)]
    cols_orden += [f"PROCEDIMIENTO{k}" for k in range(1, 6)]
    return df[cols_orden]


# ------------------------------------------------------------------ Main
def main() -> None:
    rng = np.random.default_rng(SEED)

    print("→ Construyendo CIE-10 mini master...")
    df_cie10 = build_cie10_master_mini()
    df_cie10.to_excel(FIXTURES_DIR / "cie10_master_mini.xlsx", index=False)
    print(f"  {FIXTURES_DIR / 'cie10_master_mini.xlsx'} ({len(df_cie10)} filas)")

    print("→ Construyendo CIE-9 mini master...")
    df_cie9 = build_cie9_master_mini()
    df_cie9.to_excel(FIXTURES_DIR / "cie9_master_mini.xlsx", index=False)
    print(f"  {FIXTURES_DIR / 'cie9_master_mini.xlsx'} ({len(df_cie9)} filas)")

    print("→ Construyendo GRD sample...")
    df_grd = build_grd_sample(rng)
    df_grd.to_csv(FIXTURES_DIR / "grd_sample.csv", index=False, sep="|")
    print(f"  {FIXTURES_DIR / 'grd_sample.csv'} ({len(df_grd)} filas)")

    print("\n✓ Fixtures generadas correctamente.")
    print(f"  Hospitales: {df_grd['COD_HOSPITAL'].nunique()}")
    print(f"  Años: {sorted(df_grd['anio'].unique())}")
    print(f"  Tipos actividad:\n{df_grd['TIPO_ACTIVIDAD'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
