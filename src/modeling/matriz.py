"""Constructor de la Hospital_Matrix_Integrada (Req 9)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import write_parquet


@dataclass
class Constructor_Matriz:
    features_tradicionales: pd.DataFrame
    features_diversidad: pd.DataFrame
    features_cma: pd.DataFrame
    casuistica_reducida: pd.DataFrame
    features_extendidas: pd.DataFrame | None = None

    COLS_TRADICIONALES: tuple[str, ...] = (
        "egresos_por_anio", "estancia_media", "estancia_mediana",
        "peso_medio_grd", "severidad_media", "mortalidad_media",
    )
    COLS_DIVERSIDAD: tuple[str, ...] = ("entropia_grd", "comorbilidades_promedio")
    COLS_CMA: tuple[str, ...] = ("tasa_cma", "peso_medio_cma")

    def construir(self) -> pd.DataFrame:
        df = self.features_tradicionales.copy()
        df = df.merge(self.features_diversidad, on="COD_HOSPITAL", how="left", validate="1:1")
        df = df.merge(self.features_cma, on="COD_HOSPITAL", how="left", validate="1:1")
        df = df.merge(self.casuistica_reducida, on="COD_HOSPITAL", how="left", validate="1:1")
        if self.features_extendidas is not None:
            df = df.merge(
                self.features_extendidas, on="COD_HOSPITAL", how="left", validate="1:1"
            )

        df["peso_medio_cma_imputado"] = df["peso_medio_cma"].isna()
        if df["peso_medio_cma_imputado"].any():
            con_cma = df[df["tasa_cma"] > 0]
            mediana = (
                con_cma["peso_medio_cma"].median()
                if not con_cma["peso_medio_cma"].dropna().empty
                else 1.0
            )
            df.loc[df["peso_medio_cma_imputado"], "peso_medio_cma"] = mediana

        # Features extendidas pueden tener NaN residual (hospitales sin partos,
        # sin procedimientos, etc.): se imputan a 0 de forma explicita.
        if self.features_extendidas is not None:
            cols_ext = [c for c in self.features_extendidas.columns if c != "COD_HOSPITAL"]
            df[cols_ext] = df[cols_ext].fillna(0.0)

        cols_obligatorias = [
            "COD_HOSPITAL", *self.COLS_TRADICIONALES,
            *self.COLS_DIVERSIDAD, *self.COLS_CMA,
        ]
        nulos = df[cols_obligatorias].isna().any(axis=0)
        if nulos.any():
            cols_con_nulos = nulos[nulos].index.tolist()
            raise ValueError(
                f"Hospital_Matrix_Integrada con nulos en columnas obligatorias: {cols_con_nulos}"
            )

        return df.sort_values("COD_HOSPITAL").reset_index(drop=True)

    def construir_y_persistir(self) -> Path:
        matriz = self.construir()
        expected = [
            "COD_HOSPITAL", *self.COLS_TRADICIONALES,
            *self.COLS_DIVERSIDAD, *self.COLS_CMA,
            "peso_medio_cma_imputado",
        ]
        return write_parquet(matriz, "hospital_matrix", expected_cols=expected)
