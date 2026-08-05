"""Constructor de features extendidas (Pipeline_Integrado v2).

Calcula bloques adicionales sobre `grd_filtrado_v2.parquet` para enriquecer la
`Hospital_Matrix_Integrada`:

  Demograficas:
    - pct_pediatrico (EDAD < 15)
    - pct_geriatrico (EDAD >= 65)
    - pct_femenino_fertil (SEXO=MUJER y 15 <= EDAD < 50)

  Mix de ingreso:
    - pct_urgencia (TIPO_INGRESO_N == 'URGENCIA')
    - pct_programada (TIPO_INGRESO_N == 'PROGRAMADA')
    - pct_obstetrica_ingreso (TIPO_INGRESO_N == 'OBSTETRICA')

  Tipo de alta:
    - pct_alta_domicilio
    - pct_alta_fallecido
    - pct_alta_traslado_red

  Origen del paciente:
    - pct_origen_emergencia
    - pct_origen_referencia (otros hospitales)

  Pabellon:
    - pct_uso_pabellon (al menos un USOSPABELLON > 0)
    - pabellones_promedio

  Atencion obstetrica/neonatal explicita:
    - tasa_partos (CONDICIONDEALTANEONATO1 no nulo / total egresos hosp)
    - tasa_prematurez (PESORN1 < 2500 / partos)

  Ratios de eficiencia:
    - cv_estancia (coeficiente de variacion de estancia)
    - pct_estancia_larga (DIAS_ESTADA > 30)

Todas las features se calculan SOLO sobre HOSPITALIZACION (excluye CMA) para
hospitales elegibles. Las CMA siguen alimentando tasa_cma en otro modulo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columnas que consumen los siete bloques de features extendidas.
COLS_REQUERIDAS = (
    "COD_HOSPITAL", "EDAD", "SEXO", "TIPO_INGRESO_N", "TIPOALTA_N",
    "TIPO_PROCEDENCIA_N", "USOSPABELLON_N", "CONDICIONDEALTANEONATO1",
    "PESORN1", "DIAS_ESTADA",
)


@dataclass
class Constructor_Features_Extendidas:
    """Calcula features extendidas a partir del parquet enriquecido v2.

    Parameters
    ----------
    df_grd : pd.DataFrame
        DataFrame con columnas: COD_HOSPITAL, MODALIDAD, EDAD, SEXO,
        TIPO_INGRESO_N, TIPOALTA_N, TIPO_PROCEDENCIA_N, USOSPABELLON_N,
        CONDICIONDEALTANEONATO1, PESORN1, DIAS_ESTADA, ESPECIALIDAD_MEDICA.
    hospitales_elegibles : pd.Index
        Codigos de hospital validos (ya filtrados por elegibilidad).
    """

    df_grd: pd.DataFrame
    hospitales_elegibles: pd.Index
    _cache: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def _df_hosp_elegibles(self) -> pd.DataFrame:
        """Egresos de hospitalización elegibles, cacheados y sin columnas extra.

        Los siete bloques de features invocan este método; recalcular el filtro
        sobre todas las columnas del parquet en cada llamada multiplica el uso
        de memoria sin necesidad.
        """
        if self._cache is None:
            mask = (self.df_grd["MODALIDAD"].to_numpy() == "HOSPITALIZACION") & (
                self.df_grd["COD_HOSPITAL"].isin(self.hospitales_elegibles).to_numpy()
            )
            cols = [c for c in COLS_REQUERIDAS if c in self.df_grd.columns]
            self._cache = self.df_grd.loc[mask, cols]
        return self._cache

    # ----------------------------------------------------- features demograficas
    def features_demograficas(self) -> pd.DataFrame:
        df = self._df_hosp_elegibles()

        es_pediatrico = df["EDAD"] < 15
        es_geriatrico = df["EDAD"] >= 65
        es_mujer = df["SEXO"].astype(str).str.upper().str.strip() == "MUJER"
        es_fertil = (df["EDAD"] >= 15) & (df["EDAD"] < 50)

        agg = df.groupby("COD_HOSPITAL").agg(
            n_total=("EDAD", "size"),
        )
        agg["n_pediatrico"] = es_pediatrico.groupby(df["COD_HOSPITAL"]).sum()
        agg["n_geriatrico"] = es_geriatrico.groupby(df["COD_HOSPITAL"]).sum()
        agg["n_femenino_fertil"] = (es_mujer & es_fertil).groupby(df["COD_HOSPITAL"]).sum()
        agg["edad_mediana"] = df.groupby("COD_HOSPITAL")["EDAD"].median()

        agg["pct_pediatrico"] = agg["n_pediatrico"] / agg["n_total"]
        agg["pct_geriatrico"] = agg["n_geriatrico"] / agg["n_total"]
        agg["pct_femenino_fertil"] = agg["n_femenino_fertil"] / agg["n_total"]

        result = agg[["pct_pediatrico", "pct_geriatrico",
                      "pct_femenino_fertil", "edad_mediana"]].copy()
        return result.sort_index().reset_index()

    # -------------------------------------------------- features mix ingreso
    def features_mix_ingreso(self) -> pd.DataFrame:
        df = self._df_hosp_elegibles()
        tipo = df["TIPO_INGRESO_N"]

        agg = df.groupby("COD_HOSPITAL").agg(n_total=("EDAD", "size"))
        agg["n_urgencia"] = (tipo == "URGENCIA").groupby(df["COD_HOSPITAL"]).sum()
        agg["n_programada"] = (tipo == "PROGRAMADA").groupby(df["COD_HOSPITAL"]).sum()
        agg["n_obstetrica_ingreso"] = (tipo == "OBSTETRICA").groupby(df["COD_HOSPITAL"]).sum()

        agg["pct_urgencia"] = agg["n_urgencia"] / agg["n_total"]
        agg["pct_programada"] = agg["n_programada"] / agg["n_total"]
        agg["pct_obstetrica_ingreso"] = agg["n_obstetrica_ingreso"] / agg["n_total"]

        result = agg[["pct_urgencia", "pct_programada", "pct_obstetrica_ingreso"]].copy()
        return result.sort_index().reset_index()

    # ------------------------------------------------------- features tipo alta
    def features_tipo_alta(self) -> pd.DataFrame:
        df = self._df_hosp_elegibles()
        alta = df["TIPOALTA_N"]
        es_traslado = alta.astype(str).str.contains("DERIVACI", na=False)

        agg = df.groupby("COD_HOSPITAL").agg(n_total=("EDAD", "size"))
        agg["n_alta_domicilio"] = (alta == "DOMICILIO").groupby(df["COD_HOSPITAL"]).sum()
        agg["n_alta_fallecido"] = (alta == "FALLECIDO").groupby(df["COD_HOSPITAL"]).sum()
        agg["n_alta_traslado"] = es_traslado.groupby(df["COD_HOSPITAL"]).sum()

        agg["pct_alta_domicilio"] = agg["n_alta_domicilio"] / agg["n_total"]
        agg["pct_alta_fallecido"] = agg["n_alta_fallecido"] / agg["n_total"]
        agg["pct_alta_traslado"] = agg["n_alta_traslado"] / agg["n_total"]

        result = agg[["pct_alta_domicilio", "pct_alta_fallecido", "pct_alta_traslado"]].copy()
        return result.sort_index().reset_index()

    # ------------------------------------------------------- features procedencia
    def features_origen(self) -> pd.DataFrame:
        df = self._df_hosp_elegibles()
        proc = df["TIPO_PROCEDENCIA_N"].astype(str)
        es_emergencia = proc.str.contains("EMERGENCIA|URGENCIA", na=False)
        es_referencia = proc.str.contains("OTROS_HOSPITALES|RED_NACIONAL", na=False)

        agg = df.groupby("COD_HOSPITAL").agg(n_total=("EDAD", "size"))
        agg["n_origen_emergencia"] = es_emergencia.groupby(df["COD_HOSPITAL"]).sum()
        agg["n_origen_referencia"] = es_referencia.groupby(df["COD_HOSPITAL"]).sum()

        agg["pct_origen_emergencia"] = agg["n_origen_emergencia"] / agg["n_total"]
        agg["pct_origen_referencia"] = agg["n_origen_referencia"] / agg["n_total"]

        result = agg[["pct_origen_emergencia", "pct_origen_referencia"]].copy()
        return result.sort_index().reset_index()

    # ----------------------------------------------------------- features pabellon
    def features_pabellon(self) -> pd.DataFrame:
        # Se agrega sobre las series directamente: duplicar el marco de egresos
        # solo para alojar una columna auxiliar es innecesariamente costoso.
        df = self._df_hosp_elegibles()
        hospitales = df["COD_HOSPITAL"]
        # Limpieza outliers extremos: >10 pabellones suena raro, asumimos error de captura
        pab = df["USOSPABELLON_N"].clip(lower=0, upper=10)

        agg = pd.DataFrame({"n_total": hospitales.groupby(hospitales).size()})
        agg.index.name = "COD_HOSPITAL"
        agg["n_uso_pabellon"] = (pab > 0).groupby(hospitales).sum()
        agg["pabellones_promedio"] = pab.groupby(hospitales).mean()
        agg["pct_uso_pabellon"] = agg["n_uso_pabellon"] / agg["n_total"]

        result = agg[["pct_uso_pabellon", "pabellones_promedio"]].copy()
        return result.sort_index().reset_index()

    # --------------------------------------------- features obstetricas explicitas
    def features_obstetrica(self) -> pd.DataFrame:
        df = self._df_hosp_elegibles()
        tiene_neonato = df["CONDICIONDEALTANEONATO1"].notna() & (
            df["CONDICIONDEALTANEONATO1"].astype(str).str.strip() != ""
        )
        peso_rn = pd.to_numeric(df["PESORN1"], errors="coerce")
        es_prematuro = peso_rn < 2500  # < 2500 g indicador clasico de prematurez

        agg = df.groupby("COD_HOSPITAL").agg(n_total=("EDAD", "size"))
        agg["n_partos"] = tiene_neonato.groupby(df["COD_HOSPITAL"]).sum()
        agg["n_prematuros"] = (tiene_neonato & es_prematuro).groupby(df["COD_HOSPITAL"]).sum()

        agg["tasa_partos"] = agg["n_partos"] / agg["n_total"]
        # Prematurez sobre el total de partos (no sobre el total de egresos)
        agg["tasa_prematurez"] = np.where(
            agg["n_partos"] > 0, agg["n_prematuros"] / agg["n_partos"], 0.0
        )

        result = agg[["tasa_partos", "tasa_prematurez"]].copy()
        return result.sort_index().reset_index()

    # --------------------------------------------------- features estancia
    def features_estancia_extra(self) -> pd.DataFrame:
        df = self._df_hosp_elegibles()
        agg = df.groupby("COD_HOSPITAL").agg(
            n_total=("EDAD", "size"),
            estancia_std=("DIAS_ESTADA", "std"),
            estancia_mean=("DIAS_ESTADA", "mean"),
        )
        agg["n_estancia_larga"] = (df["DIAS_ESTADA"] > 30).groupby(df["COD_HOSPITAL"]).sum()
        agg["pct_estancia_larga"] = agg["n_estancia_larga"] / agg["n_total"]
        agg["cv_estancia"] = (
            agg["estancia_std"] / agg["estancia_mean"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        result = agg[["cv_estancia", "pct_estancia_larga"]].copy()
        return result.sort_index().reset_index()

    # -------------------------------------------------------- bundle completo
    def construir_todas(self) -> pd.DataFrame:
        """Devuelve un unico DataFrame con TODAS las features extendidas."""
        bloques = [
            self.features_demograficas(),
            self.features_mix_ingreso(),
            self.features_tipo_alta(),
            self.features_origen(),
            self.features_pabellon(),
            self.features_obstetrica(),
            self.features_estancia_extra(),
        ]
        out = bloques[0]
        for b in bloques[1:]:
            out = out.merge(b, on="COD_HOSPITAL", how="outer", validate="1:1")
        return out.sort_values("COD_HOSPITAL").reset_index(drop=True)
