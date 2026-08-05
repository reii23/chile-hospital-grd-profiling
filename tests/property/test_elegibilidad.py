"""Tests property-based para la elegibilidad de hospitales.

# Feature: integracion-clinica-pipeline-grd, Property 11: La elegibilidad depende
# exclusivamente de la presencia en >=3 de los 6 anios del periodo.
"""
from __future__ import annotations

import pandas as pd
from hypothesis import given, settings, strategies as st

from src.etl.casuistica import (
    MIN_ANIOS_PRESENTES,
    Constructor_Features,
)


# Estrategia: matriz hospital × año × n_egresos
def _hospital_anio_egresos():
    """Genera (hospital, anio, n_egresos) con n_egresos potencialmente bajo o alto."""
    return st.tuples(
        st.sampled_from(["H1", "H2", "H3", "H4", "H5", "H6"]),
        st.sampled_from([2019, 2020, 2021, 2022, 2023, 2024]),
        st.integers(min_value=0, max_value=2000),
    )


def construir_df_desde_matriz(matriz: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Convierte una lista de (hospital, anio, n_egresos) en un DataFrame de egresos."""
    rows = []
    for hosp, anio, n in matriz:
        for _ in range(n):
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
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Property 11
# ---------------------------------------------------------------------------
class TestElegibilidadPresenciaTemporal:
    """# Feature: integracion-clinica-pipeline-grd, Property 11"""

    @given(matriz=st.lists(_hospital_anio_egresos(), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_elegible_tiene_al_menos_tres_anios_presentes(self, matriz, constructor_factory):
        df = construir_df_desde_matriz(matriz)
        if df.empty:
            return
        c = constructor_factory(df)
        elegibles = c.hospitales_elegibles()

        # Para cada elegible: verificar presencia en >=3 años (cualquier n>0 cuenta)
        for hosp in elegibles:
            sub = df[df["COD_HOSPITAL"] == hosp]
            anios_presentes = sub["anio"].nunique()
            assert anios_presentes >= MIN_ANIOS_PRESENTES, (
                f"Hospital {hosp} elegible pero presente solo en "
                f"{anios_presentes} años"
            )

    @given(matriz=st.lists(_hospital_anio_egresos(), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_descartado_tiene_menos_de_tres_anios_presentes(self, matriz, constructor_factory):
        df = construir_df_desde_matriz(matriz)
        if df.empty:
            return
        c = constructor_factory(df)
        elegibles = set(c.hospitales_elegibles().tolist())
        todos_hosp = set(df["COD_HOSPITAL"].unique().tolist())

        for hosp in todos_hosp - elegibles:
            sub = df[df["COD_HOSPITAL"] == hosp]
            anios_presentes = sub["anio"].nunique()
            assert anios_presentes < MIN_ANIOS_PRESENTES, (
                f"Hospital {hosp} descartado pero presente en "
                f"{anios_presentes} años (debería ser < {MIN_ANIOS_PRESENTES})"
            )
