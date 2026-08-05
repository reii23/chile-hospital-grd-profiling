"""Property-based tests para la clasificación TIPO_ACTIVIDAD → MODALIDAD.

# Feature: integracion-clinica-pipeline-grd, Property derivada: clasificación TIPO_ACTIVIDAD → MODALIDAD
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st


# ----------------------------------------------------------------------------
# Mapeo de referencia (oráculo)
# ----------------------------------------------------------------------------
TIPOS_HOSP = frozenset({
    "HOSPITALIZACIÓN",
    "HOSPITALIZACIÓN EN URGENCIA",
    "HOSPITALIZACIÓN DIURNA",
})
TIPO_CMA = "CIRUGÍA MAYOR AMBULATORIA (CMA)"

# Valores que pueden aparecer en TIPO_ACTIVIDAD (los 4 válidos + ruido)
tipo_actividad_valores = st.sampled_from([
    "HOSPITALIZACIÓN",
    "HOSPITALIZACIÓN EN URGENCIA",
    "HOSPITALIZACIÓN DIURNA",
    "CIRUGÍA MAYOR AMBULATORIA (CMA)",
    "DESCONOCIDO",
    "NO IDENTIFICADO",
    "",
    "HOSPITALIZACION",  # sin tilde — debe excluirse
    "OTRO VALOR ARBITRARIO",
])


def clasificar_modalidad(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reproduce la lógica del notebook 01 (Req 2.1, 2.2, 2.3).

    Returns
    -------
    (mask, modalidad) : tuple[pd.Series, pd.Series]
        - `mask`: True si el registro debe persistirse (Hospitalización ∪ CMA).
        - `modalidad`: 'HOSPITALIZACION' o 'CMA' para los registros que pasan.
    """
    is_hosp = serie.isin(TIPOS_HOSP)
    is_cma = serie == TIPO_CMA
    mask = is_hosp | is_cma
    modalidad = pd.Series(
        np.where(serie == TIPO_CMA, "CMA", "HOSPITALIZACION"),
        index=serie.index,
    )
    return mask, modalidad


# ----------------------------------------------------------------------------
# Property tests
# ----------------------------------------------------------------------------
class TestModalidadDeterminista:
    """# Feature: integracion-clinica-pipeline-grd, Property derivada"""

    @given(valores=st.lists(tipo_actividad_valores, min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_modalidad_es_hosp_sii_tipo_en_lista(self, valores):
        """MODALIDAD == HOSPITALIZACION ⟺ TIPO_ACTIVIDAD ∈ TIPOS_HOSP"""
        s = pd.Series(valores)
        mask, modalidad = clasificar_modalidad(s)
        # En registros que pasan el mask, modalidad correcta
        for tipo, m, mod in zip(valores, mask, modalidad):
            if not m:
                continue
            if mod == "HOSPITALIZACION":
                assert tipo in TIPOS_HOSP, (
                    f"{tipo!r} clasificado HOSPITALIZACION pero no está en TIPOS_HOSP"
                )
            else:
                assert mod == "CMA"
                assert tipo == TIPO_CMA, (
                    f"{tipo!r} clasificado CMA pero no es '{TIPO_CMA}'"
                )

    @given(valores=st.lists(tipo_actividad_valores, min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_registros_invalidos_excluidos(self, valores):
        """Req 2.2: registros con TIPO_ACTIVIDAD inválido NO deben pasar el mask."""
        s = pd.Series(valores)
        mask, _ = clasificar_modalidad(s)
        for tipo, m in zip(valores, mask):
            if tipo not in TIPOS_HOSP and tipo != TIPO_CMA:
                assert not m, (
                    f"{tipo!r} no es válido pero pasó el mask (Req 2.2)"
                )

    @given(valores=st.lists(tipo_actividad_valores, min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_modalidad_dominio_estricto(self, valores):
        """MODALIDAD ∈ {HOSPITALIZACION, CMA} sin valores extraños."""
        s = pd.Series(valores)
        _, modalidad = clasificar_modalidad(s)
        valores_unicos = set(modalidad.unique())
        assert valores_unicos.issubset({"HOSPITALIZACION", "CMA"})

    def test_caso_canonico_completo(self):
        """Smoke test con los 4 valores válidos y 2 inválidos."""
        s = pd.Series([
            "HOSPITALIZACIÓN",
            "HOSPITALIZACIÓN EN URGENCIA",
            "HOSPITALIZACIÓN DIURNA",
            "CIRUGÍA MAYOR AMBULATORIA (CMA)",
            "DESCONOCIDO",
            "",
        ])
        mask, modalidad = clasificar_modalidad(s)
        assert mask.tolist() == [True, True, True, True, False, False]
        # Para los que pasan el mask:
        modalidad_validas = modalidad[mask].tolist()
        assert modalidad_validas == [
            "HOSPITALIZACION",
            "HOSPITALIZACION",
            "HOSPITALIZACION",
            "CMA",
        ]
