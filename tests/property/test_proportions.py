"""Tests property-based para los vectores de proporciones del Constructor_Features.

# Feature: integracion-clinica-pipeline-grd, Property 1: Los vectores de proporciones suman 1.0
# Feature: integracion-clinica-pipeline-grd, Property 5: tasa_cma ∈ [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from src.etl.casuistica import (
    CAP_DESCONOCIDO,
    COLUMNAS_CAPITULOS,
    Constructor_Features,
)


# ---------------------------------------------------------------------------
# Estrategias compartidas
# ---------------------------------------------------------------------------
# Hospitales sintéticos
hospitales_st = st.sampled_from(["HOSP_001", "HOSP_002", "HOSP_003", "HOSP_004"])
anios_st = st.sampled_from([2019, 2020, 2021, 2022, 2023, 2024])

# Diagnósticos: una letra del alfabeto cubierto en la fixture mini + dígitos
codigos_diag_validos_mini = st.sampled_from([
    f"{letra}{n:02d}"
    for letra in "ACIOS"
    for n in range(0, 10)
])
codigos_diag_o_nulo = st.one_of(st.none(), codigos_diag_validos_mini)

# Procedimientos: enteros 0-99 con sub-código opcional
codigos_proc_validos = st.builds(
    lambda ent, sub: f"{ent:02d}.{sub:02d}" if sub is not None else f"{ent:02d}",
    ent=st.sampled_from([0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 35, 36, 37, 38, 72, 73, 74, 75]),
    sub=st.one_of(st.none(), st.integers(min_value=0, max_value=99)),
)
codigos_proc_o_nulo = st.one_of(st.none(), codigos_proc_validos)


def gen_egreso(modalidad="HOSPITALIZACION"):
    """Generador de un egreso sintético con todas las columnas requeridas."""
    return st.builds(
        lambda hosp, anio, grd, peso, sev, mort, dias, d1, d2, d3, d4, d5, p1, p2, p3, p4, p5: {
            "COD_HOSPITAL": hosp,
            "anio": anio,
            "MODALIDAD": modalidad,
            "IR_29301_COD_GRD": grd,
            "IR_29301_PESO": peso,
            "IR_29301_SEVERIDAD": sev,
            "IR_29301_MORTALIDAD": mort,
            "DIAS_ESTADA": dias,
            "DIAGNOSTICO1": d1,
            "DIAGNOSTICO2": d2,
            "DIAGNOSTICO3": d3,
            "DIAGNOSTICO4": d4,
            "DIAGNOSTICO5": d5,
            "PROCEDIMIENTO1": p1,
            "PROCEDIMIENTO2": p2,
            "PROCEDIMIENTO3": p3,
            "PROCEDIMIENTO4": p4,
            "PROCEDIMIENTO5": p5,
        },
        hosp=hospitales_st,
        anio=anios_st,
        grd=st.sampled_from(["540001", "540002", "540003", "1234", "5678", "9012"]),
        peso=st.floats(min_value=0.3, max_value=3.0, allow_nan=False),
        sev=st.integers(min_value=0, max_value=4),
        mort=st.integers(min_value=0, max_value=4),
        dias=st.integers(min_value=0, max_value=30),
        d1=codigos_diag_validos_mini,  # principal siempre presente
        d2=codigos_diag_o_nulo,
        d3=codigos_diag_o_nulo,
        d4=codigos_diag_o_nulo,
        d5=codigos_diag_o_nulo,
        p1=codigos_proc_o_nulo,
        p2=codigos_proc_o_nulo,
        p3=codigos_proc_o_nulo,
        p4=codigos_proc_o_nulo,
        p5=codigos_proc_o_nulo,
    )


# Generador eficiente: parámetros pequeños → DataFrame grande (sin overhead de Hypothesis).
def _build_df_grande(rng_seed: int, n_egresos_por_hosp_anio: int = 100,
                     tasa_cma: float = 0.10) -> pd.DataFrame:
    """Construye determinísticamente un DataFrame de egresos sintético.

    4 hospitales × 6 años × n_egresos_por_hosp_anio = ~2400 filas para n=100.
    """
    rng = np.random.default_rng(rng_seed)
    hospitales = ["HOSP_001", "HOSPITAL_002", "HOSP_003", "HOSP_004"]
    anios = [2019, 2020, 2021, 2022, 2023, 2024]
    diags_pool = [f"{l}{n:02d}" for l in "ACIOS" for n in range(0, 10)]
    procs_pool = [f"{i:02d}.{j:02d}" for i in [0, 8, 13, 35, 72] for j in [0, 5, 10]]
    grds_pool = ["540001", "540002", "540003", "1234", "5678", "9012"]

    rows = []
    for hosp in hospitales:
        for anio in anios:
            for _ in range(n_egresos_por_hosp_anio):
                es_cma = rng.random() < tasa_cma
                modalidad = "CMA" if es_cma else "HOSPITALIZACION"
                row = {
                    "COD_HOSPITAL": hosp,
                    "anio": anio,
                    "MODALIDAD": modalidad,
                    "IR_29301_COD_GRD": rng.choice(grds_pool),
                    "IR_29301_PESO": float(rng.uniform(0.3, 3.0)),
                    "IR_29301_SEVERIDAD": int(rng.integers(0, 5)),
                    "IR_29301_MORTALIDAD": int(rng.integers(0, 5)),
                    "DIAS_ESTADA": int(rng.integers(0, 30)),
                    "DIAGNOSTICO1": rng.choice(diags_pool),
                }
                # Diagnósticos secundarios con cobertura decreciente
                for k, cobertura in enumerate([0.85, 0.65, 0.50, 0.35], start=2):
                    row[f"DIAGNOSTICO{k}"] = (
                        rng.choice(diags_pool) if rng.random() < cobertura else None
                    )
                # Procedimientos: ~70% tienen al menos PROC1
                tiene_proc = rng.random() < 0.70
                row["PROCEDIMIENTO1"] = rng.choice(procs_pool) if tiene_proc else None
                for k, cobertura in enumerate([0.40, 0.25, 0.15, 0.05], start=2):
                    row[f"PROCEDIMIENTO{k}"] = (
                        rng.choice(procs_pool)
                        if (tiene_proc and rng.random() < cobertura)
                        else None
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# Reemplazamos el generador: solo genera una semilla pequeña
def gen_dataset_grande():
    """Genera un dataset sintético grande parametrizado por una semilla.

    Esto evita el overhead de Hypothesis de generar listas de miles de dicts.
    El dataset siempre tiene 4 hospitales × 6 años × 100 egresos = 2400 filas
    (suficiente para que los 4 hospitales sean elegibles).
    """
    return st.builds(_build_df_grande, rng_seed=st.integers(min_value=0, max_value=1000))


# Settings reutilizable: tests sobre datasets grandes
SETTINGS_GRANDE = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.large_base_example,
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)


# Dataset pequeño para tests donde no necesitamos elegibilidad
def gen_dataset_pequeño():
    return st.builds(
        lambda egresos: pd.DataFrame(egresos),
        egresos=st.lists(gen_egreso("HOSPITALIZACION"), min_size=10, max_size=100),
    )


@pytest.fixture(scope="module")
def constructor_factory_local(mapper_cie10_mini, mapper_cie9_mini):
    """Versión local del factory (no usar; preferir el de conftest.py)."""

    def _make(df: pd.DataFrame) -> Constructor_Features:
        # Asegurar orden de columnas y tipos esperados
        c = Constructor_Features(
            df_grd=df,
            mapper_cie10=mapper_cie10_mini,
            mapper_cie9=mapper_cie9_mini,
        )
        return c

    return _make


# ---------------------------------------------------------------------------
# Property 1: Vectores de proporciones suman 1.0
# ---------------------------------------------------------------------------
class TestVectoresProporcionesSumanUno:
    """# Feature: integracion-clinica-pipeline-grd, Property 1"""

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_vector_capitulos_principal_suma_uno(self, df, constructor_factory):
        c = constructor_factory(df)
        if len(c.hospitales_elegibles()) == 0:
            return  # caso degenerado
        v = c.vector_capitulos_principal()
        cols_prop = [col for col in v.columns if col != "COD_HOSPITAL"]
        sumas = v[cols_prop].sum(axis=1)
        np.testing.assert_allclose(sumas, 1.0, atol=1e-6)

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_vector_capitulos_ponderado_suma_uno(self, df, constructor_factory):
        c = constructor_factory(df)
        if len(c.hospitales_elegibles()) == 0:
            return
        v = c.vector_capitulos_ponderado()
        cols_prop = [col for col in v.columns if col != "COD_HOSPITAL"]
        sumas = v[cols_prop].sum(axis=1)
        np.testing.assert_allclose(sumas, 1.0, atol=1e-6)

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_vector_top20_suma_uno(self, df, constructor_factory):
        c = constructor_factory(df)
        if len(c.hospitales_elegibles()) == 0:
            return
        v = c.vector_top20()
        cols_prop = [col for col in v.columns if col != "COD_HOSPITAL"]
        sumas = v[cols_prop].sum(axis=1)
        np.testing.assert_allclose(sumas, 1.0, atol=1e-6)

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_vector_secciones_suma_uno_o_vacio(self, df, constructor_factory):
        """Si hay egresos con procedimientos: suma=1; si no, dataframe vacío."""
        c = constructor_factory(df)
        if len(c.hospitales_elegibles()) == 0:
            return
        v = c.vector_secciones()
        if len(v) == 0:
            return  # ningún egreso con PROC válido (caso edge)
        cols_prop = [col for col in v.columns if col != "COD_HOSPITAL"]
        sumas = v[cols_prop].sum(axis=1)
        np.testing.assert_allclose(sumas, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Property 5: tasa_cma ∈ [0, 1]
# ---------------------------------------------------------------------------
class TestTasaCmaDominio:
    """# Feature: integracion-clinica-pipeline-grd, Property 5: tasa_cma ∈ [0, 1]"""

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_tasa_cma_en_rango(self, df, constructor_factory):
        c = constructor_factory(df)
        if len(c.hospitales_elegibles()) == 0:
            return
        cma_df = c.features_cma()
        assert (cma_df["tasa_cma"] >= 0.0).all()
        assert (cma_df["tasa_cma"] <= 1.0).all()

    def test_tasa_cma_es_cero_si_no_hay_cma(self, mapper_cie10_mini, mapper_cie9_mini):
        """Edge case: hospital sin egresos CMA → tasa_cma=0, peso_medio_cma=NaN (Req 2.7)."""
        # Generar egresos solo HOSPITALIZACION para 4 hospitales en 6 años, 600/año
        rng = np.random.default_rng(42)
        rows = []
        for hosp in ["HOSP_001", "HOSP_002", "HOSP_003", "HOSP_004"]:
            for anio in range(2019, 2025):
                for _ in range(600):
                    rows.append({
                        "COD_HOSPITAL": hosp,
                        "anio": anio,
                        "MODALIDAD": "HOSPITALIZACION",
                        "IR_29301_COD_GRD": "X",
                        "IR_29301_PESO": 1.0,
                        "IR_29301_SEVERIDAD": 1,
                        "IR_29301_MORTALIDAD": 1,
                        "DIAS_ESTADA": 5,
                        "DIAGNOSTICO1": "A09",
                        "DIAGNOSTICO2": None, "DIAGNOSTICO3": None,
                        "DIAGNOSTICO4": None, "DIAGNOSTICO5": None,
                        "PROCEDIMIENTO1": None, "PROCEDIMIENTO2": None,
                        "PROCEDIMIENTO3": None, "PROCEDIMIENTO4": None,
                        "PROCEDIMIENTO5": None,
                    })
        df = pd.DataFrame(rows)
        c = Constructor_Features(df, mapper_cie10_mini, mapper_cie9_mini)
        cma = c.features_cma()
        assert (cma["tasa_cma"] == 0.0).all()
        assert cma["peso_medio_cma"].isna().all()
