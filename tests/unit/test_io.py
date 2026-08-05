"""Tests unitarios para src/utils/io.py.

Cubre:
- write_parquet falla cuando faltan expected_cols (Req 13.5).
- El orden de columnas y filas en el archivo persistido es determinista.
- Dos ejecuciones consecutivas con mismo input producen el mismo archivo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.io import (
    ARTIFACTS,
    artifact_exists,
    get_path,
    read_parquet,
    write_parquet,
)


# ----------------------------------------------------------------------------
# Setup: redirigir ARTIFACTS a un tmp_path por cada test
# ----------------------------------------------------------------------------
@pytest.fixture
def tmp_artifact(tmp_path, monkeypatch):
    """Redirige una clave de ARTIFACTS a un archivo temporal por test."""
    key = "casuistica_capitulos"
    tmp_file = tmp_path / "test_output.parquet"
    monkeypatch.setitem(ARTIFACTS, key, tmp_file)
    return key, tmp_file


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
class TestWriteParquetSchema:
    """Validación de esquema obligatorio."""

    def test_falla_cuando_faltan_columnas_obligatorias(self, tmp_artifact):
        key, _ = tmp_artifact
        df = pd.DataFrame({"COD_HOSPITAL": ["A", "B"], "valor": [1, 2]})
        with pytest.raises(ValueError, match="columnas obligatorias ausentes"):
            write_parquet(df, key, expected_cols=["COD_HOSPITAL", "cap_01"])

    def test_falla_si_expected_cols_vacio(self, tmp_artifact):
        key, _ = tmp_artifact
        df = pd.DataFrame({"COD_HOSPITAL": ["A"]})
        with pytest.raises(ValueError, match="expected_cols no puede estar vacío"):
            write_parquet(df, key, expected_cols=[])

    def test_falla_con_key_desconocida(self, tmp_path):
        df = pd.DataFrame({"COD_HOSPITAL": ["A"]})
        with pytest.raises(KeyError, match="Artefacto desconocido"):
            write_parquet(df, "key_inexistente", expected_cols=["COD_HOSPITAL"])


class TestWriteParquetOrden:
    """Determinismo de orden de columnas y filas."""

    def test_orden_columnas_obligatorias_primero(self, tmp_artifact):
        key, _ = tmp_artifact
        # Crear df con columnas en orden "incorrecto"
        df = pd.DataFrame({
            "extra_z": [1, 2],
            "cap_01": [0.5, 0.3],
            "COD_HOSPITAL": ["B", "A"],
            "extra_a": [10, 20],
        })
        write_parquet(df, key, expected_cols=["COD_HOSPITAL", "cap_01"])

        df_out = read_parquet(key)
        # Obligatorias primero, en orden indicado
        assert list(df_out.columns[:2]) == ["COD_HOSPITAL", "cap_01"]
        # Extras después, ordenadas alfabéticamente
        assert list(df_out.columns[2:]) == ["extra_a", "extra_z"]

    def test_orden_filas_por_primera_obligatoria(self, tmp_artifact):
        key, _ = tmp_artifact
        df = pd.DataFrame({
            "COD_HOSPITAL": ["C", "A", "B"],
            "valor": [3, 1, 2],
        })
        write_parquet(df, key, expected_cols=["COD_HOSPITAL"])

        df_out = read_parquet(key)
        # Filas en orden alfabético por COD_HOSPITAL
        assert df_out["COD_HOSPITAL"].tolist() == ["A", "B", "C"]
        assert df_out["valor"].tolist() == [1, 2, 3]

    def test_dos_ejecuciones_producen_archivos_identicos(self, tmp_artifact):
        """Property 9 (determinismo): mismo input → mismo output bit-a-bit."""
        key, tmp_file = tmp_artifact
        df = pd.DataFrame({
            "COD_HOSPITAL": ["B", "A", "C"],
            "valor": [2, 1, 3],
        })

        write_parquet(df, key, expected_cols=["COD_HOSPITAL"])
        bytes_1 = tmp_file.read_bytes()

        write_parquet(df, key, expected_cols=["COD_HOSPITAL"])
        bytes_2 = tmp_file.read_bytes()

        assert bytes_1 == bytes_2, (
            "Dos ejecuciones consecutivas con el mismo input deben producir "
            "archivos parquet idénticos bit-a-bit (Req 13.5)."
        )


class TestRoundtrip:
    """Lectura tras escritura preserva los datos."""

    def test_roundtrip_basico(self, tmp_artifact):
        key, _ = tmp_artifact
        df_in = pd.DataFrame({
            "COD_HOSPITAL": ["A", "B"],
            "cap_01": [0.5, 0.3],
            "cap_02": [0.5, 0.7],
        })
        write_parquet(df_in, key, expected_cols=["COD_HOSPITAL"])
        df_out = read_parquet(key)

        # Mismo contenido (ordenado por COD_HOSPITAL)
        pd.testing.assert_frame_equal(
            df_out.sort_values("COD_HOSPITAL").reset_index(drop=True),
            df_in.sort_values("COD_HOSPITAL").reset_index(drop=True),
            check_like=True,
        )


class TestArtifactExists:
    def test_retorna_false_para_artefacto_inexistente(self, tmp_artifact):
        key, _ = tmp_artifact
        assert artifact_exists(key) is False

    def test_retorna_true_tras_persistir(self, tmp_artifact):
        key, _ = tmp_artifact
        df = pd.DataFrame({"COD_HOSPITAL": ["A"]})
        write_parquet(df, key, expected_cols=["COD_HOSPITAL"])
        assert artifact_exists(key) is True


class TestGetPath:
    def test_get_path_retorna_path_registrado(self):
        path = get_path("hospital_matrix")
        assert isinstance(path, Path)
        assert path.name == "hospital_matrix.parquet"

    def test_get_path_falla_con_key_invalida(self):
        with pytest.raises(KeyError, match="Artefacto desconocido"):
            get_path("clave_inventada_xyz")
