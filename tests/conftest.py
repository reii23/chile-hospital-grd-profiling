"""Configuración de pytest y fixtures compartidas.

Las fixtures se generan una vez con `python tests/fixtures/build_fixtures.py`
y se cargan aquí para ser inyectadas en los tests.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from hypothesis import settings, Verbosity

# Perfil determinístico de Hypothesis para CI
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,  # los tests sobre dataframes pueden tardar
    derandomize=True,
)
settings.register_profile(
    "dev",
    max_examples=50,
    verbosity=Verbosity.verbose,
)
settings.load_profile("ci")


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------------------
# Fixtures de archivos
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def df_grd_mini() -> pd.DataFrame:
    """1.000 filas sintéticas con todas las columnas del Pipeline_Integrado."""
    path = FIXTURES_DIR / "grd_sample.csv"
    if not path.exists():
        pytest.skip(
            f"Fixture {path} no existe. Ejecutar:\n"
            f"  python tests/fixtures/build_fixtures.py"
        )
    return pd.read_csv(path, sep="|", dtype=str)


@pytest.fixture(scope="session")
def df_cie10_mini() -> pd.DataFrame:
    """Tabla maestra CIE-10 mini (50 códigos × 5 capítulos)."""
    path = FIXTURES_DIR / "cie10_master_mini.xlsx"
    if not path.exists():
        pytest.skip(
            f"Fixture {path} no existe. Ejecutar:\n"
            f"  python tests/fixtures/build_fixtures.py"
        )
    return pd.read_excel(path)


@pytest.fixture(scope="session")
def df_cie9_mini() -> pd.DataFrame:
    """Tabla maestra CIE-9-MC mini (30 códigos × 4 secciones)."""
    path = FIXTURES_DIR / "cie9_master_mini.xlsx"
    if not path.exists():
        pytest.skip(
            f"Fixture {path} no existe. Ejecutar:\n"
            f"  python tests/fixtures/build_fixtures.py"
        )
    return pd.read_excel(path)


# ------------------------------------------------------------------------------
# Fixtures de mappers (lazy: solo se construyen cuando los pide un test)
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mapper_cie10_mini(df_cie10_mini):
    """Instancia de Mapeador_CIE10 con la tabla mini ya construida."""
    from src.etl.cie_mappers import Mapeador_CIE10

    mapper = Mapeador_CIE10(df_maestra=df_cie10_mini)
    mapper.construir()
    return mapper


@pytest.fixture(scope="session")
def mapper_cie9_mini(df_cie9_mini):
    """Instancia de Mapeador_CIE9 con la tabla mini ya construida."""
    from src.etl.cie_mappers import Mapeador_CIE9

    mapper = Mapeador_CIE9(df_maestra=df_cie9_mini)
    mapper.construir()
    return mapper


@pytest.fixture(scope="session")
def constructor_factory(mapper_cie10_mini, mapper_cie9_mini):
    """Factoría que retorna un Constructor_Features dado un DataFrame de egresos.

    Disponible para todos los tests de la suite (compartida en conftest.py).
    """
    from src.etl.casuistica import Constructor_Features

    def _make(df):
        return Constructor_Features(
            df_grd=df,
            mapper_cie10=mapper_cie10_mini,
            mapper_cie9=mapper_cie9_mini,
        )

    return _make
