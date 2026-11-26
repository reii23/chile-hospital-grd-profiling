"""Pruebas de regresión: el pipeline reproduce las cifras reportadas en la tesis.

Operan sobre los artefactos versionados en ``data/processed/`` y
``reports/tables/``. Si un cambio en el código altera la matriz, la reducción
de casuística o la partición de referencia, estas pruebas fallan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.modeling.referencia import particion_referencia, silhouette_referencia
from src.utils.io import PROCESSED_DIR, TABLES_DIR

MATRIZ = PROCESSED_DIR / "hospital_matrix.parquet"
pytestmark = pytest.mark.skipif(not MATRIZ.exists(), reason="requiere data/processed/hospital_matrix.parquet")


def test_matriz_institucional_tiene_65_hospitales_y_35_variables():
    X, _, ids = particion_referencia()
    assert X.shape == (65, 35)
    assert len(set(ids)) == 65


DIMS = [f"dim_0{i}" for i in range(1, 9)]


def test_componentes_de_casuistica_son_8_ortogonales_y_ordenadas():
    matriz = pd.read_parquet(MATRIZ)
    dims = matriz[DIMS].to_numpy()
    varianzas = dims.var(axis=0, ddof=1)
    assert np.all(np.diff(varianzas) < 0)
    assert np.abs(np.corrcoef(dims.T) - np.eye(8)).max() < 1e-8
    assert np.abs(dims.mean(axis=0)).max() < 1e-8


def test_componentes_de_la_matriz_coinciden_con_el_artefacto_de_reduccion():
    matriz = pd.read_parquet(MATRIZ).assign(COD_HOSPITAL=lambda d: d["COD_HOSPITAL"].astype(str)).set_index("COD_HOSPITAL")
    reducida = pd.read_parquet(PROCESSED_DIR / "casuistica_reducida.parquet")
    reducida = reducida.assign(COD_HOSPITAL=reducida["COD_HOSPITAL"].astype(str)).set_index("COD_HOSPITAL")
    np.testing.assert_allclose(matriz.loc[reducida.index, DIMS].to_numpy(), reducida[DIMS].to_numpy(), atol=1e-10)


def test_particion_ward_reproduce_tamanos_y_silhouette():
    _, labels, _ = particion_referencia()
    assert sorted(np.bincount(labels).tolist(), reverse=True) == [54, 5, 3, 3]
    assert silhouette_referencia() == pytest.approx(0.398, abs=0.0005)


def test_particion_coincide_con_la_asignacion_reportada():
    _, labels, ids = particion_referencia()
    reportada = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype={"COD_HOSPITAL": str}).set_index("COD_HOSPITAL")
    assert adjusted_rand_score(reportada.loc[ids, "nivel1_K4"], labels) == pytest.approx(1.0)
