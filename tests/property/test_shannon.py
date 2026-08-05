"""Property-based tests para src/etl/diversity.py.

# Feature: integracion-clinica-pipeline-grd, Property 5: entropia_grd ∈ [0, 1]
# Feature: integracion-clinica-pipeline-grd, Property 5: comorbilidades_promedio ∈ [0, 4]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from src.etl.diversity import comorbilidades_por_egreso, shannon_normalizada


# ----------------------------------------------------------------------------
# Estrategias
# ----------------------------------------------------------------------------
# Series de conteos enteros ≥ 0, longitud variable
series_conteos = st.builds(
    lambda lst: pd.Series(lst),
    lst=st.lists(
        st.integers(min_value=0, max_value=10_000),
        min_size=0,
        max_size=50,
    ),
)


# Generador de un valor de diagnóstico arbitrario (string|None|whitespace)
diag_value = st.one_of(
    st.none(),
    st.just(""),
    st.just("  "),  # whitespace
    st.from_regex(r"[A-Z]\d{2}(\.\d)?", fullmatch=True),
)


# DataFrame de 4 columnas DIAG2..5, n filas variable.
# Construimos con st.lists de 4-tuplas y luego convertimos a DataFrame, evitando
# las restricciones de hypothesis.extra.pandas.data_frames con rows+columns.
df_diag2_5 = st.builds(
    lambda rows: pd.DataFrame(
        rows, columns=["DIAGNOSTICO2", "DIAGNOSTICO3", "DIAGNOSTICO4", "DIAGNOSTICO5"]
    ),
    rows=st.lists(
        st.tuples(diag_value, diag_value, diag_value, diag_value),
        min_size=0,
        max_size=20,
    ),
)


# ----------------------------------------------------------------------------
# Property 5: entropia_grd ∈ [0, 1]
# ----------------------------------------------------------------------------
class TestShannonDominio:
    """# Feature: integracion-clinica-pipeline-grd, Property 5: entropia ∈ [0, 1]"""

    @given(s=series_conteos)
    @settings(max_examples=100)
    def test_shannon_en_rango_0_1(self, s):
        """Para toda serie de conteos, shannon_normalizada(s) ∈ [0, 1]."""
        h = shannon_normalizada(s)
        assert 0.0 <= h <= 1.0, f"Entropía fuera de rango: {h}"

    def test_shannon_serie_vacia_es_cero(self):
        """Convención: serie vacía → 0.0."""
        assert shannon_normalizada(pd.Series([], dtype=float)) == 0.0

    def test_shannon_singleton_es_cero(self):
        """K = 1 → entropía 0 (Req 4.2)."""
        assert shannon_normalizada(pd.Series([100])) == 0.0
        assert shannon_normalizada(pd.Series([1])) == 0.0

    def test_shannon_uniforme_es_uno(self):
        """K elementos con frecuencia igual → entropía 1.0."""
        for k in [2, 5, 10, 100]:
            s = pd.Series([10] * k)
            h = shannon_normalizada(s)
            assert abs(h - 1.0) < 1e-12, f"K={k}: esperado 1.0, obtenido {h}"

    def test_shannon_ignora_ceros(self):
        """Req 4.3: una serie con ceros adicionales tiene la misma entropía
        que la serie sin ceros."""
        s_sin_ceros = pd.Series([5, 3, 2])
        s_con_ceros = pd.Series([5, 3, 2, 0, 0, 0])
        h1 = shannon_normalizada(s_sin_ceros)
        h2 = shannon_normalizada(s_con_ceros)
        assert abs(h1 - h2) < 1e-12

    def test_shannon_concentrada_es_bajo(self):
        """Distribución muy desbalanceada → entropía baja."""
        # 99% en 1 GRD, 1% en otro
        s = pd.Series([990, 10])
        h = shannon_normalizada(s)
        assert 0.0 < h < 0.15

    def test_shannon_total_cero_es_cero(self):
        """Serie con todos ceros → 0.0."""
        assert shannon_normalizada(pd.Series([0, 0, 0])) == 0.0

    @given(s=series_conteos)
    @settings(max_examples=100)
    def test_shannon_invariante_a_escala(self, s):
        """Multiplicar todos los conteos por una constante > 0 no cambia la
        entropía (Property derivada: la entropía mide proporciones)."""
        if s.sum() == 0:
            return  # caso degenerado
        h1 = shannon_normalizada(s)
        h2 = shannon_normalizada(s * 7)
        assert abs(h1 - h2) < 1e-12


# ----------------------------------------------------------------------------
# Property 5: comorbilidades_promedio ∈ [0, 4]
# ----------------------------------------------------------------------------
class TestComorbilidadesDominio:
    """# Feature: integracion-clinica-pipeline-grd, Property 5: comorbilidades ∈ [0, 4]"""

    @given(df=df_diag2_5)
    @settings(max_examples=100)
    def test_comorbilidades_en_rango_0_4(self, df):
        """Para todo df con 4 columnas DIAG2..5, count(no_nulos) ∈ [0, 4]."""
        result = comorbilidades_por_egreso(df)
        assert (result >= 0).all()
        assert (result <= 4).all()

    def test_comorbilidades_caso_canonico(self):
        df = pd.DataFrame({
            "DIAGNOSTICO2": ["A09", None, "I21", ""],
            "DIAGNOSTICO3": ["B05", "C50", None, "  "],
            "DIAGNOSTICO4": [None, None, None, "S00"],
            "DIAGNOSTICO5": [None, None, None, None],
        })
        # Fila 0: 2 no nulos. Fila 1: 1. Fila 2: 1. Fila 3: 1 (S00; "" y "  " son nulos).
        assert comorbilidades_por_egreso(df).tolist() == [2, 1, 1, 1]

    def test_comorbilidades_df_vacio(self):
        df = pd.DataFrame(columns=["DIAGNOSTICO2", "DIAGNOSTICO3"])
        result = comorbilidades_por_egreso(df)
        assert len(result) == 0

    def test_comorbilidades_todas_nulas(self):
        df = pd.DataFrame({
            "DIAGNOSTICO2": [None, None, None],
            "DIAGNOSTICO3": [None, None, None],
            "DIAGNOSTICO4": [None, None, None],
            "DIAGNOSTICO5": [None, None, None],
        })
        assert (comorbilidades_por_egreso(df) == 0).all()

    def test_comorbilidades_todas_llenas(self):
        df = pd.DataFrame({
            "DIAGNOSTICO2": ["A", "B"],
            "DIAGNOSTICO3": ["C", "D"],
            "DIAGNOSTICO4": ["E", "F"],
            "DIAGNOSTICO5": ["G", "H"],
        })
        assert (comorbilidades_por_egreso(df) == 4).all()
