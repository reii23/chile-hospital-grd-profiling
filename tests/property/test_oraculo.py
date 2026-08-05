"""Tests property-based: oráculo de referencia para los vectores de Constructor_Features.

# Feature: integracion-clinica-pipeline-grd, Property 4: Los vectores calculados coinciden con su oráculo de referencia
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.etl.casuistica import (
    CAP_DESCONOCIDO,
    Constructor_Features,
    columnas_codificadas,
)
from src.etl.cie_mappers import SEC_DESCONOCIDA
from tests.property.test_proportions import (
    SETTINGS_GRANDE,
    gen_dataset_grande,
)


# ---------------------------------------------------------------------------
# Oráculos: implementación naïve loop-based para comparar
# ---------------------------------------------------------------------------
def oraculo_capitulos_principal(
    df: pd.DataFrame, mapper_cie10, hospitales: list[str]
) -> dict[str, dict[str, float]]:
    """Conteo manual loop-by-row de capítulos sobre DIAGNOSTICO1."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for _, row in df.iterrows():
        if row["MODALIDAD"] != "HOSPITALIZACION":
            continue
        if row["COD_HOSPITAL"] not in hospitales:
            continue
        cap = mapper_cie10.mapear(pd.Series([row["DIAGNOSTICO1"]])).iloc[0]
        counts[row["COD_HOSPITAL"]][cap] += 1

    proporciones: dict[str, dict[str, float]] = {}
    for hosp, cnt in counts.items():
        total = sum(cnt.values())
        proporciones[hosp] = {cap: c / total for cap, c in cnt.items()} if total else {}
    return proporciones


def oraculo_capitulos_ponderado(
    df: pd.DataFrame, mapper_cie10, hospitales: list[str]
) -> dict[str, dict[str, float]]:
    """Suma ponderada loop-by-row: peso 1.0 a DIAG1, 0.5 a cada secundario.

    Recorre todas las columnas de diagnóstico presentes en el DataFrame, de modo
    que el oráculo acompaña al esquema real en lugar de fijar cinco campos.
    """
    cols_diagnostico = columnas_codificadas(df, "DIAGNOSTICO")
    pesos: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in df.iterrows():
        if row["MODALIDAD"] != "HOSPITALIZACION":
            continue
        if row["COD_HOSPITAL"] not in hospitales:
            continue
        for col in cols_diagnostico:
            valor = row[col]
            if pd.isna(valor) or str(valor).strip() == "":
                continue
            cap = mapper_cie10.mapear(pd.Series([valor])).iloc[0]
            peso = 1.0 if col == "DIAGNOSTICO1" else 0.5
            pesos[row["COD_HOSPITAL"]][cap] += peso

    proporciones: dict[str, dict[str, float]] = {}
    for hosp, cnt in pesos.items():
        total = sum(cnt.values())
        proporciones[hosp] = {cap: c / total for cap, c in cnt.items()} if total else {}
    return proporciones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def asserts_diccionarios_coinciden(
    actual_df: pd.DataFrame, oraculo: dict[str, dict[str, float]], col_id: str = "COD_HOSPITAL"
):
    """Verifica que las proporciones del DataFrame == oráculo (tolerancia 1e-9)."""
    for _, row in actual_df.iterrows():
        hosp = row[col_id]
        actual = {col: row[col] for col in actual_df.columns if col != col_id and row[col] > 0}
        oraculo_hosp = {k: v for k, v in oraculo.get(hosp, {}).items() if v > 0}
        # Mismas claves no nulas
        assert set(actual.keys()) == set(oraculo_hosp.keys()), (
            f"Hospital {hosp}: claves divergentes\n"
            f"  actual: {sorted(actual.keys())}\n"
            f"  oráculo: {sorted(oraculo_hosp.keys())}"
        )
        for cap in actual:
            np.testing.assert_allclose(
                actual[cap], oraculo_hosp[cap], atol=1e-9,
                err_msg=f"Hospital {hosp}, capítulo {cap}",
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestOraculoCapitulos:
    """# Feature: integracion-clinica-pipeline-grd, Property 4"""

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_vector_capitulos_principal_coincide_con_oraculo(
        self, df, constructor_factory
    ):
        c = constructor_factory(df)
        elegibles = c.hospitales_elegibles().tolist()
        if not elegibles:
            return
        actual = c.vector_capitulos_principal()
        oraculo = oraculo_capitulos_principal(df, c.mapper_cie10, elegibles)
        asserts_diccionarios_coinciden(actual, oraculo)

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_vector_capitulos_ponderado_coincide_con_oraculo(
        self, df, constructor_factory
    ):
        c = constructor_factory(df)
        elegibles = c.hospitales_elegibles().tolist()
        if not elegibles:
            return
        actual = c.vector_capitulos_ponderado()
        oraculo = oraculo_capitulos_ponderado(df, c.mapper_cie10, elegibles)
        asserts_diccionarios_coinciden(actual, oraculo)
