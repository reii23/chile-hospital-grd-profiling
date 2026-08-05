"""Tests property-based: invariancia frente a registros del subconjunto irrelevante.

# Feature: integracion-clinica-pipeline-grd, Property 6: Las features son invariantes frente a registros del subconjunto irrelevante
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from src.etl.casuistica import Constructor_Features
from tests.property.test_proportions import (
    SETTINGS_GRANDE,
    gen_dataset_grande,
    gen_egreso,
)


cma_extra = st.lists(gen_egreso("CMA"), min_size=0, max_size=100)


# ---------------------------------------------------------------------------
# Property 6
# ---------------------------------------------------------------------------
class TestInvarianciaSubconjunto:
    """# Feature: integracion-clinica-pipeline-grd, Property 6"""

    @given(df=gen_dataset_grande(), extra=cma_extra)
    @SETTINGS_GRANDE
    def test_features_tradicionales_ignoran_cma_extra(
        self, df, extra, constructor_factory
    ):
        """features_tradicionales(df) == features_tradicionales(df ∪ CMA_extra)."""
        c1 = constructor_factory(df)
        if len(c1.hospitales_elegibles()) == 0:
            return
        df2 = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        c2 = constructor_factory(df2)

        f1 = c1.features_tradicionales().set_index("COD_HOSPITAL")
        f2 = c2.features_tradicionales().set_index("COD_HOSPITAL")
        # Mismos hospitales (porque CMA no afecta elegibilidad)
        assert set(f1.index) == set(f2.index)
        pd.testing.assert_frame_equal(
            f1.sort_index(), f2.sort_index(), check_dtype=False, atol=1e-9
        )

    @given(df=gen_dataset_grande(), extra=cma_extra)
    @SETTINGS_GRANDE
    def test_elegibilidad_independiente_de_cma(self, df, extra, constructor_factory):
        """Property 11 + 6: la elegibilidad solo depende de hospitalización."""
        c1 = constructor_factory(df)
        elegibles_1 = set(c1.hospitales_elegibles().tolist())

        df2 = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        c2 = constructor_factory(df2)
        elegibles_2 = set(c2.hospitales_elegibles().tolist())

        assert elegibles_1 == elegibles_2

    @given(df=gen_dataset_grande())
    @SETTINGS_GRANDE
    def test_diversidad_ignora_cma(self, df, constructor_factory):
        """features_diversidad solo depende de egresos hospitalización."""
        c1 = constructor_factory(df)
        if len(c1.hospitales_elegibles()) == 0:
            return
        # df sin CMA = df solo HOSPITALIZACION
        df_sin_cma = df[df["MODALIDAD"] == "HOSPITALIZACION"].copy()
        c2 = constructor_factory(df_sin_cma)

        d1 = c1.features_diversidad().set_index("COD_HOSPITAL")
        d2 = c2.features_diversidad().set_index("COD_HOSPITAL")
        pd.testing.assert_frame_equal(
            d1.sort_index(), d2.sort_index(), check_dtype=False, atol=1e-9
        )
