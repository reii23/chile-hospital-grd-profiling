"""Validadores de datos y mapeos clínicos.

Niveles de severidad (ver design.md §"Error Handling"):
- Nivel 1 (fatal): elevan `CoberturaInsuficienteError`. Detienen el pipeline.
- Nivel 2 (advertencia): retornan strings/flags, no detienen ejecución.
- Nivel 3 (omisión registrada): el flujo natural los maneja; aquí solo se reporta.

Implementaciones:
- `Validador_CIE10` (Req 5.6, 15.1): cobertura DIAGNOSTICO1 → cap_NN.
- `Validador_CIE9`  (Req 6.6, 15.2): cobertura PROCEDIMIENTO1 → sec_NN + flag por hospital.
- `Validador_GRD`   (Req 15.3): Top-20 GRDs vs maestra MINSAL.
- `Validador_Correlaciones` (Req 4.6, 4.7): Spearman entropia_grd vs egresos_por_anio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:  # evitar import circular en runtime
    from src.etl.cie_mappers import Mapeador_CIE10, Mapeador_CIE9


# ----------------------------------------------------------------------------
# Excepciones
# ----------------------------------------------------------------------------
class CoberturaInsuficienteError(RuntimeError):
    """Se eleva cuando la cobertura CIE no alcanza el umbral mínimo (Req 15.1, 15.2)."""


# ----------------------------------------------------------------------------
# Validador_CIE10 (Req 5.6, 15.1)
# ----------------------------------------------------------------------------
@dataclass
class Validador_CIE10:
    """Valida cobertura del mapeo CIE-10 sobre los egresos.

    Atributos
    ---------
    df_grd : pd.DataFrame
        DataFrame con columnas `anio`, `DIAGNOSTICO1`, `MODALIDAD`.
    mapper : Mapeador_CIE10
        Mapeador CIE-10 ya construido.
    umbral_critico : float, default=0.95
        Cobertura mínima por año para no detener (Req 15.1).
    umbral_advertencia_desconocido : float, default=0.05
        Proporción máxima nacional de `CAP_DESCONOCIDO` antes de advertir (Req 5.6).
    """

    df_grd: pd.DataFrame
    mapper: "Mapeador_CIE10"
    umbral_critico: float = 0.95
    umbral_advertencia_desconocido: float = 0.05

    def validar_por_anio(self) -> pd.DataFrame:
        """Cobertura DIAGNOSTICO1 → cap_NN agrupado por año.

        Eleva `CoberturaInsuficienteError` si algún año < umbral_critico (Req 15.1).
        """
        from src.etl.cie_mappers import CAP_DESCONOCIDO

        df = self.df_grd[self.df_grd["MODALIDAD"] == "HOSPITALIZACION"]
        mask_no_nulo = df["DIAGNOSTICO1"].notna() & (
            df["DIAGNOSTICO1"].astype(str).str.strip() != ""
        )
        df = df[mask_no_nulo]

        if df.empty:
            return pd.DataFrame(columns=["anio", "n_total", "n_mapeados", "cobertura"])

        capitulos = self.mapper.mapear(df["DIAGNOSTICO1"])
        df_eval = df.assign(_cap=capitulos.values)
        df_eval["_mapeado"] = (df_eval["_cap"] != CAP_DESCONOCIDO).astype("int64")

        agg = df_eval.groupby("anio").agg(
            n_total=("_mapeado", "size"),
            n_mapeados=("_mapeado", "sum"),
        )
        agg["cobertura"] = agg["n_mapeados"] / agg["n_total"]
        agg = agg.reset_index().sort_values("anio").reset_index(drop=True)

        bajo_umbral = agg[agg["cobertura"] < self.umbral_critico]
        if not bajo_umbral.empty:
            primer = bajo_umbral.iloc[0]
            raise CoberturaInsuficienteError(
                f"Cobertura CIE-10 insuficiente en año {int(primer['anio'])}: "
                f"{primer['cobertura']:.1%} < {self.umbral_critico:.1%} "
                f"(Req 15.1)"
            )
        return agg

    def emitir_advertencia_desconocido(
        self, vector_chapters: pd.DataFrame
    ) -> str | None:
        """Si CAP_DESCONOCIDO > umbral nacional, retorna mensaje de advertencia."""
        from src.etl.cie_mappers import CAP_DESCONOCIDO

        if CAP_DESCONOCIDO not in vector_chapters.columns:
            return None
        prop_nacional = float(vector_chapters[CAP_DESCONOCIDO].mean())
        if prop_nacional > self.umbral_advertencia_desconocido:
            return (
                f"[WARN] CAP_DESCONOCIDO = {prop_nacional:.1%} a nivel nacional "
                f"> umbral {self.umbral_advertencia_desconocido:.1%} (Req 5.6). "
                f"Inspeccionar reports/tables/cie10_codigos_no_mapeados.csv."
            )
        return None


# ----------------------------------------------------------------------------
# Validador_CIE9 (Req 6.6, 15.2)
# ----------------------------------------------------------------------------
@dataclass
class Validador_CIE9:
    """Valida cobertura del mapeo CIE-9-MC sobre los egresos.

    La cobertura se evalúa **solo sobre egresos con `PROCEDIMIENTO1` no nulo**.
    """

    df_grd: pd.DataFrame
    mapper: "Mapeador_CIE9"
    umbral_critico: float = 0.90
    umbral_cobertura_baja_hospital: float = 0.50

    def validar_por_anio(self) -> pd.DataFrame:
        """Cobertura PROC1 → sec_NN agrupado por año, sobre PROC1 no nulo.

        Eleva `CoberturaInsuficienteError` si algún año < umbral_critico.
        """
        from src.etl.cie_mappers import SEC_DESCONOCIDA

        df = self.df_grd[self.df_grd["MODALIDAD"] == "HOSPITALIZACION"]
        mask_no_nulo = df["PROCEDIMIENTO1"].notna() & (
            df["PROCEDIMIENTO1"].astype(str).str.strip() != ""
        )
        df = df[mask_no_nulo]

        if df.empty:
            return pd.DataFrame(columns=["anio", "n_total", "n_mapeados", "cobertura"])

        secciones = self.mapper.mapear(df["PROCEDIMIENTO1"])
        df_eval = df.assign(_sec=secciones.values)
        df_eval["_mapeado"] = (df_eval["_sec"] != SEC_DESCONOCIDA).astype("int64")

        agg = df_eval.groupby("anio").agg(
            n_total=("_mapeado", "size"),
            n_mapeados=("_mapeado", "sum"),
        )
        agg["cobertura"] = agg["n_mapeados"] / agg["n_total"]
        agg = agg.reset_index().sort_values("anio").reset_index(drop=True)

        bajo_umbral = agg[agg["cobertura"] < self.umbral_critico]
        if not bajo_umbral.empty:
            primer = bajo_umbral.iloc[0]
            raise CoberturaInsuficienteError(
                f"Cobertura CIE-9-MC insuficiente en año {int(primer['anio'])}: "
                f"{primer['cobertura']:.1%} < {self.umbral_critico:.1%} "
                f"(Req 15.2)"
            )
        return agg

    def cobertura_por_hospital(self) -> pd.DataFrame:
        """Fracción de egresos sin procedimiento codificado por hospital.

        Marca `cie9_cobertura_baja=True` si fracción > umbral (Req 6.6).
        """
        cols_proc = [f"PROCEDIMIENTO{i}" for i in range(1, 6)]
        df = self.df_grd[self.df_grd["MODALIDAD"] == "HOSPITALIZACION"].copy()
        cols_proc_existentes = [c for c in cols_proc if c in df.columns]

        df["_tiene_proc"] = df[cols_proc_existentes].apply(
            lambda col: col.notna() & (col.astype(str).str.strip() != "")
        ).any(axis=1).astype("int64")
        df["_sin_proc"] = 1 - df["_tiene_proc"]

        agg = df.groupby("COD_HOSPITAL").agg(
            n_total=("_sin_proc", "size"),
            n_sin_proc=("_sin_proc", "sum"),
        )
        agg["frac_sin_proc"] = agg["n_sin_proc"] / agg["n_total"]
        agg["cie9_cobertura_baja"] = (
            agg["frac_sin_proc"] > self.umbral_cobertura_baja_hospital
        )
        return agg.sort_index().reset_index()


# ----------------------------------------------------------------------------
# Validador_GRD (Req 15.3)
# ----------------------------------------------------------------------------
@dataclass
class Validador_GRD:
    """Valida que los códigos del Top-20 estén en la maestra GRD MINSAL."""

    top20: list[str]
    df_maestra_grd: pd.DataFrame
    columna_codigo: str = "IR_29301_COD_GRD"

    def emitir_advertencia(self) -> list[str]:
        """Retorna lista de códigos Top-20 ausentes en `df_maestra_grd`."""
        if self.columna_codigo not in self.df_maestra_grd.columns:
            candidatos = [
                c for c in self.df_maestra_grd.columns
                if "GRD" in str(c).upper() or "COD" in str(c).upper()
            ]
            if not candidatos:
                return list(self.top20)
            col = candidatos[0]
        else:
            col = self.columna_codigo

        codigos_maestra = set(self.df_maestra_grd[col].astype(str).str.strip().tolist())
        ausentes = [
            grd for grd in self.top20
            if str(grd).strip() not in codigos_maestra
        ]
        return ausentes


# ----------------------------------------------------------------------------
# Validador_Correlaciones (Req 4.6, 4.7)
# ----------------------------------------------------------------------------
@dataclass
class Validador_Correlaciones:
    """Detecta correlaciones altas en la Hospital_Matrix_Integrada."""

    matriz_integrada: pd.DataFrame
    umbral: float = 0.85

    def correlaciones_spearman(self) -> pd.DataFrame:
        """Matriz de pares con `|spearman_r|`, ordenados por magnitud."""
        num_cols = self.matriz_integrada.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return pd.DataFrame(
                columns=["feature_a", "feature_b", "spearman_r", "p_value", "advertencia"]
            )

        rows = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                vec_a = self.matriz_integrada[a].dropna()
                vec_b = self.matriz_integrada[b].dropna()
                idx_comun = vec_a.index.intersection(vec_b.index)
                if len(idx_comun) < 3:
                    continue
                r, p = stats.spearmanr(vec_a.loc[idx_comun], vec_b.loc[idx_comun])
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "spearman_r": r,
                    "p_value": p,
                    "advertencia": abs(r) > self.umbral,
                })
        return pd.DataFrame(rows).sort_values(
            "spearman_r", key=abs, ascending=False
        ).reset_index(drop=True)

    def advertencia_entropia_volumen(self) -> str | None:
        """Mensaje si Spearman(entropia_grd, egresos_por_anio) > umbral (Req 4.7)."""
        cols = self.matriz_integrada.columns
        if "entropia_grd" not in cols or "egresos_por_anio" not in cols:
            return None
        vec_e = self.matriz_integrada["entropia_grd"].dropna()
        vec_v = self.matriz_integrada["egresos_por_anio"].dropna()
        idx = vec_e.index.intersection(vec_v.index)
        if len(idx) < 3:
            return None
        r, _ = stats.spearmanr(vec_e.loc[idx], vec_v.loc[idx])
        if abs(r) > self.umbral:
            return (
                f"[WARN] Spearman(entropia_grd, egresos_por_anio) = {r:.3f} "
                f"> {self.umbral:.2f} (Req 4.7). La entropía podría no añadir "
                f"información incremental sobre el volumen."
            )
        return None
