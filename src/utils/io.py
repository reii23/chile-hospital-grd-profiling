"""I/O sobre parquet con validación de esquema y orden determinista.

Este módulo centraliza el mapeo `nombre_artefacto → ruta` para evitar strings
dispersos en notebooks, y aplica:

1. Orden de columnas determinista (Req 13.5: reproducibilidad bit-a-bit).
2. Orden de filas determinista (sort por la primera columna obligatoria).
3. Validación de esquema mínimo (raise si faltan columnas obligatorias).
4. Compresión zstd por defecto (~3× más compacto que snappy, lectura rápida).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# Rutas canónicas
# ----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

REPORTS_DIR = _ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


ARTIFACTS: dict[str, Path] = {
    # Datos crudos procesados
    "grd_filtrado": PROCESSED_DIR / "grd_filtrado.parquet",
    # Vectores de casuística
    "casuistica_capitulos": PROCESSED_DIR / "casuistica_capitulos.parquet",
    "casuistica_procedimientos": PROCESSED_DIR / "casuistica_procedimientos.parquet",
    "casuistica_top20_grds": PROCESSED_DIR / "casuistica_top20_grds.parquet",
    "casuistica_reducida": PROCESSED_DIR / "casuistica_reducida.parquet",
    # Features intermedias
    "features_tradicionales": PROCESSED_DIR / "features_tradicionales.parquet",
    "features_diversidad": PROCESSED_DIR / "features_diversidad.parquet",
    "features_cma": PROCESSED_DIR / "features_cma.parquet",
    "features_extendidas": PROCESSED_DIR / "features_extendidas.parquet",
    # Matriz institucional y clusters
    "hospital_matrix": PROCESSED_DIR / "hospital_matrix.parquet",
    "hospital_matrix_scaled": PROCESSED_DIR / "hospital_matrix_scaled.parquet",
    "hospital_clusters_integrado": PROCESSED_DIR / "hospital_clusters_integrado.parquet",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
PREFIJOS_CLINICOS = ("DIAGNOSTICO", "PROCEDIMIENTO")


def cargar_grd_compacto(
    path: Path, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Carga el parquet de egresos acotando el uso de memoria.

    Leer el archivo completo de una vez es inviable: pandas materializa cada
    código clínico como un objeto Python independiente y, con 5,8 millones de
    registros por 65 campos de codificación, la conversión supera la memoria
    disponible del equipo.

    Para evitarlo, las columnas de diagnósticos y procedimientos se leen de a
    una y se convierten de inmediato a `category`, de modo que cada columna
    queda representada por códigos enteros sobre un vocabulario compartido.

    Parameters
    ----------
    path : Path
        Ruta al parquet de egresos.
    columns : Iterable[str] | None
        Subconjunto de columnas a cargar. Por defecto, todas.

    Returns
    -------
    pd.DataFrame
        Egresos con las columnas clínicas en dtype `category`.
    """
    disponibles = [n for n in pq.read_schema(path).names if not n.startswith("__index")]
    pedidas = disponibles if columns is None else [c for c in columns if c in disponibles]

    clinicas = [c for c in pedidas if c.startswith(PREFIJOS_CLINICOS)]
    restantes = [c for c in pedidas if c not in clinicas]

    df = pd.read_parquet(path, columns=restantes)
    for col in clinicas:
        df[col] = pd.read_parquet(path, columns=[col])[col].astype("category")
    return df


def get_path(key: str) -> Path:
    """Retorna la ruta canónica de un artefacto registrado.

    Raises
    ------
    KeyError
        Si `key` no está en ARTIFACTS.
    """
    if key not in ARTIFACTS:
        valid = ", ".join(sorted(ARTIFACTS))
        raise KeyError(f"Artefacto desconocido: {key!r}. Válidos: {valid}")
    return ARTIFACTS[key]


def write_parquet(
    df: pd.DataFrame,
    key: str,
    expected_cols: Iterable[str],
    *,
    compression: str = "zstd",
) -> Path:
    """Persiste df como parquet con orden determinista y validación de esquema.

    El parquet resultante satisface Req 13.5 (reproducibilidad bit-a-bit):

    - Columnas: primero `expected_cols` en el orden indicado, luego el resto
      ordenado alfabéticamente.
    - Filas: ordenadas por `expected_cols[0]` ascendente.
    - Compresión: zstd (override con `compression`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a persistir.
    key : str
        Clave del artefacto en `ARTIFACTS`.
    expected_cols : Iterable[str]
        Columnas obligatorias. La primera se usa como sort key de filas.
    compression : str, default="zstd"
        Codec de compresión (`zstd`, `snappy`, `gzip`, etc.).

    Returns
    -------
    Path
        Ruta donde se escribió el archivo.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias o `expected_cols` está vacío.
    KeyError
        Si `key` no está registrado en ARTIFACTS.
    """
    expected = list(expected_cols)
    if not expected:
        raise ValueError("expected_cols no puede estar vacío")

    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(
            f"{key}: columnas obligatorias ausentes: {sorted(missing)}. "
            f"Disponibles: {sorted(df.columns)}"
        )

    # Orden determinista de columnas: obligatorias primero, resto alfabético
    extra_cols = sorted(set(df.columns) - set(expected))
    df_ordered = df[expected + extra_cols]

    # Orden determinista de filas por la primera columna obligatoria.
    # `kind='stable'` (mergesort) garantiza orden estable entre ejecuciones.
    df_ordered = df_ordered.sort_values(
        expected[0], kind="stable"
    ).reset_index(drop=True)

    path = get_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_ordered.to_parquet(path, index=False, compression=compression)
    return path


def read_parquet(key: str) -> pd.DataFrame:
    """Lee un artefacto registrado.

    Raises
    ------
    KeyError
        Si `key` no está registrado.
    FileNotFoundError
        Si el archivo no existe en disco.
    """
    path = get_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"Artefacto {key!r} no encontrado en {path}. "
            f"Ejecutar el notebook que lo genera."
        )
    return pd.read_parquet(path)


def artifact_exists(key: str) -> bool:
    """True si el artefacto existe en disco."""
    return get_path(key).exists()
