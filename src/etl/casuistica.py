"""Constructor de features hospitalarias para el Pipeline_Integrado.

Implementa los seis bloques de features definidos en el diseño:

1. **Tradicionales** (Req 2.4, 9.1): egresos_por_anio, estancia_media, peso_medio_grd,
   severidad_media, mortalidad_media, estancia_mediana — calculadas solo sobre
   egresos de hospitalización.

2. **Diversidad** (Req 4.1, 4.4): entropia_grd (Shannon normalizada), comorbilidades_promedio.

3. **CMA** (Req 2.5, 2.6, 2.7): tasa_cma, peso_medio_cma.

4. **Vectores de capítulos CIE-10** (Req 5.3, 5.4): variantes principal y ponderado.

5. **Vectores de secciones CIE-9-MC** (Req 6.3): sobre PROCEDIMIENTO1..5.

6. **Vector Top-20 GRDs** (Req 7.3): proporciones del Top-20 nacional + OTROS_GRDS.

7. **Elegibilidad de hospitales** (Req 3): presencia en ≥3 de los 6 años del período,
   contando cualquier año con al menos un egreso de hospitalización registrado.

Todas las funciones son **deterministas** y producen outputs con el orden de
columnas/filas estable (Req 13.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.etl.cie_mappers import (
    CAP_DESCONOCIDO,
    SEC_DESCONOCIDA,
    Mapeador_CIE10,
    Mapeador_CIE9,
)
from src.etl.diversity import comorbilidades_por_egreso, shannon_normalizada


# ---------------------------------------------------------------------------
# Constantes (Req 3, 7)
# ---------------------------------------------------------------------------
MIN_ANIOS_PRESENTES = 3
TOP_K_GRDS = 20


def columnas_codificadas(df: pd.DataFrame, prefijo: str) -> list[str]:
    """Devuelve las columnas ``<prefijo><número>`` disponibles, ordenadas.

    El esquema GRD admite 35 diagnósticos y 30 procedimientos; detectar las
    columnas en el DataFrame mantiene la compatibilidad con fixtures y fuentes
    históricas que contienen solo un subconjunto de esos campos.
    """
    columnas: list[tuple[int, str]] = []
    for columna in df.columns:
        if columna.startswith(prefijo) and columna[len(prefijo):].isdigit():
            columnas.append((int(columna[len(prefijo):]), columna))
    return [columna for _, columna in sorted(columnas)]


def codigos_presentes(serie: pd.Series) -> pd.Series:
    """Máscara booleana de códigos efectivamente registrados.

    Considera nulos tanto los valores ausentes como las cadenas en blanco. Se
    aplica de a una columna para acotar el uso de memoria: evaluar las 65
    columnas clínicas a la vez sobre 5,8 millones de registros agota la RAM.

    Para columnas `category` la comprobación se resuelve sobre el vocabulario de
    categorías, evitando expandir millones de códigos a objetos de texto.
    """
    if isinstance(serie.dtype, pd.CategoricalDtype):
        en_blanco = [c for c in serie.cat.categories if str(c).strip() == ""]
        presentes = serie.notna()
        if en_blanco:
            presentes &= ~serie.isin(en_blanco)
        return presentes
    return serie.notna() & (serie.astype("string").str.strip() != "")


def mapear_codigos(serie: pd.Series, mapper, desconocido: str) -> np.ndarray:
    """Traduce códigos clínicos a capítulos o secciones de forma económica.

    Cuando la columna es `category`, el mapeo se resuelve sobre el vocabulario
    de categorías --unos pocos miles de códigos distintos-- y el resultado se
    expande indexando por posición. Normalizar y buscar millones de valores
    repetidos uno por uno multiplica el costo de memoria sin aportar nada.

    Parameters
    ----------
    serie : pd.Series
        Códigos clínicos a traducir.
    mapper : Mapeador_CIE10 | Mapeador_CIE9
        Mapeador ya construido.
    desconocido : str
        Etiqueta asignada a los códigos ausentes.
    """
    if not isinstance(serie.dtype, pd.CategoricalDtype):
        return mapper.mapear(serie).to_numpy()

    etiquetas = mapper.mapear(
        pd.Series(serie.cat.categories, dtype=object)
    ).to_numpy()
    codigos = serie.cat.codes.to_numpy()
    salida = np.empty(len(codigos), dtype=object)
    validos = codigos >= 0
    salida[validos] = etiquetas[codigos[validos]]
    salida[~validos] = desconocido
    return salida

# Capítulos CIE-10 estándar (22) + columna residual
COLUMNAS_CAPITULOS = [f"cap_{i:02d}" for i in range(1, 23)] + [CAP_DESCONOCIDO]

# Variantes del vector de capítulos
VARIANTE_PRINCIPAL = "principal"
VARIANTE_PONDERADO = "ponderado"


# ---------------------------------------------------------------------------
# Constructor_Features (Req 2-7, 9.1)
# ---------------------------------------------------------------------------
@dataclass
class Constructor_Features:
    """Construye los seis bloques de features hospitalarias.

    Parameters
    ----------
    df_grd : pd.DataFrame
        DataFrame con columnas mínimas: COD_HOSPITAL, MODALIDAD, anio,
        IR_29301_COD_GRD, IR_29301_PESO, DIAS_ESTADA, IR_29301_SEVERIDAD,
        IR_29301_MORTALIDAD, DIAGNOSTICO1..35, PROCEDIMIENTO1..30.
        El constructor también acepta subconjuntos de columnas clínicas.
    mapper_cie10 : Mapeador_CIE10
    mapper_cie9 : Mapeador_CIE9

    Notes
    -----
    Las features se calculan SIEMPRE sobre el conjunto de hospitales elegibles
    (presencia en ≥3 de los 6 años del período). Para obtenerlos:
    `hospitales_elegibles()`.
    """

    df_grd: pd.DataFrame
    mapper_cie10: Mapeador_CIE10
    mapper_cie9: Mapeador_CIE9
    _hospitales_elegibles: pd.Index | None = field(default=None, init=False, repr=False)
    _df_descartados: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _mascaras: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    # ----------------------------------------------------------------- helpers
    def _subset(self, mask: np.ndarray, cols: list[str] | None) -> pd.DataFrame:
        """Materializa solo las filas de la máscara y las columnas pedidas."""
        columnas = list(self.df_grd.columns) if cols is None else list(dict.fromkeys(cols))
        return self.df_grd.loc[mask, columnas]

    def _mask_modalidad(self, modalidad: str) -> np.ndarray:
        """Máscara booleana cacheada de una modalidad de egreso."""
        if modalidad not in self._mascaras:
            self._mascaras[modalidad] = (
                self.df_grd["MODALIDAD"].to_numpy() == modalidad
            )
        return self._mascaras[modalidad]

    def _mask_hosp_elegibles(self) -> np.ndarray:
        """Máscara de egresos de hospitalización de hospitales elegibles."""
        if "hosp_elegibles" not in self._mascaras:
            elegibles = self.hospitales_elegibles()
            self._mascaras["hosp_elegibles"] = self._mask_modalidad("HOSPITALIZACION") & (
                self.df_grd["COD_HOSPITAL"].isin(elegibles).to_numpy()
            )
        return self._mascaras["hosp_elegibles"]

    def _df_hosp(self, cols: list[str] | None = None) -> pd.DataFrame:
        """Subconjunto del df con MODALIDAD == 'HOSPITALIZACION'."""
        return self._subset(self._mask_modalidad("HOSPITALIZACION"), cols)

    def _df_cma(self, cols: list[str] | None = None) -> pd.DataFrame:
        """Subconjunto del df con MODALIDAD == 'CMA'."""
        return self._subset(self._mask_modalidad("CMA"), cols)

    def _df_hosp_elegibles(self, cols: list[str] | None = None) -> pd.DataFrame:
        """Egresos de hospitalización de hospitales elegibles.

        Se materializan únicamente las columnas solicitadas: copiar las más de
        90 columnas del parquet completo por cada indicador agota la memoria
        disponible con 5,8 millones de registros.
        """
        return self._subset(self._mask_hosp_elegibles(), cols)

    def _serie_hosp_elegibles(self, col: str) -> pd.Series:
        """Una sola columna restringida a los egresos elegibles."""
        return self.df_grd[col].loc[self._mask_hosp_elegibles()]

    @staticmethod
    def _stable_index(serie: pd.Series) -> pd.Index:
        """Garantiza orden determinista de hospitales (Req 13.5)."""
        return pd.Index(sorted(serie.unique()))

    # ----------------------------------------------------------- elegibilidad
    def hospitales_elegibles(self) -> pd.Index:
        """Hospitales con presencia en ≥3 de los 6 años del período.

        Solo cuenta egresos `MODALIDAD == 'HOSPITALIZACION'` (Req 3.2); un año
        cuenta como presente si el hospital registra al menos un egreso de
        hospitalización en ese año.

        Returns
        -------
        pd.Index
            Códigos de hospital ordenados alfabéticamente.

        Side Effects
        ------------
        Cachea el resultado y arma `self._df_descartados` (Req 3.3).
        """
        if self._hospitales_elegibles is not None:
            return self._hospitales_elegibles

        df_hosp = self._df_hosp(["COD_HOSPITAL", "anio"])
        # Conteo egresos por (hospital, año)
        conteo = df_hosp.groupby(["COD_HOSPITAL", "anio"]).size().reset_index(name="n")

        anios_presentes = (
            conteo.groupby("COD_HOSPITAL")["anio"].nunique().rename("anios_presentes")
        )

        meta = anios_presentes.to_frame()
        meta["elegible"] = meta["anios_presentes"] >= MIN_ANIOS_PRESENTES

        # Persistir descartados con motivo
        descartados = meta[~meta["elegible"]].copy()
        if len(descartados):
            descartados["motivo"] = f"anios_presentes < {MIN_ANIOS_PRESENTES}"
        self._df_descartados = descartados.reset_index()

        elegibles = meta[meta["elegible"]].index
        self._hospitales_elegibles = pd.Index(sorted(elegibles))
        return self._hospitales_elegibles

    def descartados(self) -> pd.DataFrame:
        """DataFrame con hospitales descartados y motivo (Req 3.3)."""
        if self._df_descartados is None:
            self.hospitales_elegibles()
        return self._df_descartados.copy() if self._df_descartados is not None else pd.DataFrame()

    # ----------------------------------------------------- features tradicionales
    def features_tradicionales(self) -> pd.DataFrame:
        """6 columnas: egresos_por_anio, estancia_media, estancia_mediana,
        peso_medio_grd, severidad_media, mortalidad_media. Solo HOSPITALIZACION.

        Req 2.4, 9.1.
        """
        df = self._df_hosp_elegibles([
            "COD_HOSPITAL", "anio", "DIAS_ESTADA", "IR_29301_PESO",
            "IR_29301_SEVERIDAD", "IR_29301_MORTALIDAD",
        ])
        # Conteos por hospital y año para egresos_por_anio
        n_por_anio = df.groupby(["COD_HOSPITAL", "anio"]).size().reset_index(name="n")
        agg_anios = n_por_anio.groupby("COD_HOSPITAL").agg(
            n_egresos=("n", "sum"),
            anios_presentes=("anio", "count"),
        )
        agg_anios["egresos_por_anio"] = (
            agg_anios["n_egresos"] / agg_anios["anios_presentes"]
        )

        # Estancia y métricas GRD por hospital
        agg_grd = df.groupby("COD_HOSPITAL").agg(
            estancia_media=("DIAS_ESTADA", "mean"),
            estancia_mediana=("DIAS_ESTADA", "median"),
            peso_medio_grd=("IR_29301_PESO", "mean"),
            severidad_media=("IR_29301_SEVERIDAD", "mean"),
            mortalidad_media=("IR_29301_MORTALIDAD", "mean"),
        )

        result = agg_grd.join(agg_anios[["egresos_por_anio"]], how="inner")
        result = result[
            [
                "egresos_por_anio",
                "estancia_media",
                "estancia_mediana",
                "peso_medio_grd",
                "severidad_media",
                "mortalidad_media",
            ]
        ]
        return result.sort_index().reset_index()

    # ---------------------------------------------------------- features CMA
    def features_cma(self) -> pd.DataFrame:
        """tasa_cma ∈ [0, 1] y peso_medio_cma (NaN si tasa_cma=0).

        Req 2.5, 2.6, 2.7.
        """
        elegibles = self.hospitales_elegibles()
        df_hosp = self._df_hosp(["COD_HOSPITAL"])
        df_cma = self._df_cma(["COD_HOSPITAL", "IR_29301_PESO"])

        n_hosp = df_hosp.groupby("COD_HOSPITAL").size().rename("n_hosp")
        n_cma = df_cma.groupby("COD_HOSPITAL").size().rename("n_cma")
        peso_cma = (
            df_cma.groupby("COD_HOSPITAL")["IR_29301_PESO"].mean().rename("peso_medio_cma")
        )

        df = pd.concat([n_hosp, n_cma, peso_cma], axis=1)
        df = df.reindex(elegibles)  # solo hospitales elegibles
        df.index.name = "COD_HOSPITAL"
        df["n_hosp"] = df["n_hosp"].fillna(0).astype("int64")
        df["n_cma"] = df["n_cma"].fillna(0).astype("int64")

        denominador = df["n_hosp"] + df["n_cma"]
        df["tasa_cma"] = np.where(
            denominador > 0, df["n_cma"] / denominador, 0.0
        )
        # Si n_cma=0, peso_medio_cma debe ser NaN explícitamente (Req 2.7)
        df.loc[df["n_cma"] == 0, "peso_medio_cma"] = np.nan

        result = df[["tasa_cma", "peso_medio_cma"]].copy()
        return result.sort_index().reset_index()

    # ---------------------------------------------------- features diversidad
    def features_diversidad(self) -> pd.DataFrame:
        """entropia_grd ∈ [0, 1] y comorbilidades_promedio ∈ [0, 34].

        El límite superior corresponde al esquema GRD completo (35 diagnósticos,
        uno principal y 34 secundarios).
        Req 4.1, 4.4, 4.5.
        """
        df = self._df_hosp_elegibles(["COD_HOSPITAL", "IR_29301_COD_GRD"])

        # Entropía Shannon por hospital sobre IR_29301_COD_GRD
        entropias = (
            df.groupby("COD_HOSPITAL")["IR_29301_COD_GRD"]
            .apply(lambda s: shannon_normalizada(s.value_counts()))
            .rename("entropia_grd")
        )

        # Comorbilidades promedio por hospital sobre todos los secundarios
        # disponibles. El conteo se acumula columna por columna en lugar de
        # materializar las 34 columnas secundarias a la vez: el promedio por
        # hospital equivale a la suma de presencias dividida por sus egresos.
        hospitales = df["COD_HOSPITAL"]
        n_egresos = hospitales.groupby(hospitales).size()
        presencias = pd.Series(0.0, index=n_egresos.index)
        for col in columnas_codificadas(self.df_grd, "DIAGNOSTICO")[1:]:
            presentes = codigos_presentes(self._serie_hosp_elegibles(col))
            presencias = presencias.add(
                presentes.groupby(hospitales.to_numpy()).sum(), fill_value=0.0
            )
        comorbs_promedio = (presencias / n_egresos).rename("comorbilidades_promedio")

        result = pd.concat([entropias, comorbs_promedio], axis=1)
        return result.sort_index().reset_index()

    # ------------------------------------------------- vectores de capítulos
    def vector_capitulos_principal(self) -> pd.DataFrame:
        """Vector de proporciones por capítulo CIE-10 sobre DIAGNOSTICO1.

        Req 5.3, 5.7.

        Returns
        -------
        pd.DataFrame
            Columnas: [COD_HOSPITAL, cap_01, ..., cap_22, CAP_DESCONOCIDO].
            Cada fila suma 1.0 ± 1e-6.
        """
        df = self._df_hosp_elegibles(["COD_HOSPITAL", "DIAGNOSTICO1"])
        capitulos = mapear_codigos(df["DIAGNOSTICO1"], self.mapper_cie10, CAP_DESCONOCIDO)
        df_cap = df.assign(_cap=capitulos)

        # Tabla de contingencia hospital × capítulo
        crosstab = pd.crosstab(df_cap["COD_HOSPITAL"], df_cap["_cap"])
        # Asegurar todas las columnas COLUMNAS_CAPITULOS presentes
        crosstab = crosstab.reindex(columns=COLUMNAS_CAPITULOS, fill_value=0)
        # Normalizar a proporciones por fila
        proporciones = crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0.0)

        # Validación dura (Req 5.7)
        sumas = proporciones.sum(axis=1)
        if not np.allclose(sumas, 1.0, atol=1e-6):
            mal = sumas[~np.isclose(sumas, 1.0, atol=1e-6)]
            raise AssertionError(
                f"vector_capitulos_principal: filas que no suman 1.0:\n{mal}"
            )

        result = proporciones.sort_index().reset_index()
        return result

    def vector_capitulos_ponderado(self) -> pd.DataFrame:
        """Vector ponderado: peso 1.0 a DIAGNOSTICO1 y 0.5 a cada secundario.

        Req 5.4, 5.7.
        """
        hospitales = self._serie_hosp_elegibles("COD_HOSPITAL")

        # Los pesos se acumulan por (hospital, capítulo) de a una columna de
        # diagnóstico. Concatenar los 35 bloques en un único marco largo
        # generaría decenas de millones de filas y agotaría la memoria.
        acumulado: pd.Series | None = None
        for col in columnas_codificadas(self.df_grd, "DIAGNOSTICO"):
            peso = 1.0 if col == "DIAGNOSTICO1" else 0.5
            codigos = self._serie_hosp_elegibles(col)
            presentes = codigos_presentes(codigos).to_numpy()
            if not presentes.any():
                continue
            sub = pd.DataFrame({
                "COD_HOSPITAL": hospitales.to_numpy()[presentes],
                "_cap": mapear_codigos(
                    codigos[presentes], self.mapper_cie10, CAP_DESCONOCIDO
                ),
            })
            conteo = sub.groupby(["COD_HOSPITAL", "_cap"]).size().mul(peso)
            acumulado = conteo if acumulado is None else acumulado.add(conteo, fill_value=0.0)

        if acumulado is None:
            # Edge case: ningún egreso con diagnóstico → vector vacío válido
            cols = [pd.Index(["COD_HOSPITAL"]).append(pd.Index(COLUMNAS_CAPITULOS))]
            return pd.DataFrame(columns=cols[0])

        acumulado.index.names = ["COD_HOSPITAL", "_cap"]
        agg = acumulado.unstack(fill_value=0.0)
        agg = agg.reindex(columns=COLUMNAS_CAPITULOS, fill_value=0.0)
        proporciones = agg.div(agg.sum(axis=1), axis=0).fillna(0.0)

        # Validación dura (Req 5.7)
        sumas = proporciones.sum(axis=1)
        if not np.allclose(sumas, 1.0, atol=1e-6):
            mal = sumas[~np.isclose(sumas, 1.0, atol=1e-6)]
            raise AssertionError(
                f"vector_capitulos_ponderado: filas que no suman 1.0:\n{mal}"
            )

        result = proporciones.sort_index().reset_index()
        return result

    def vector_capitulos(self, variante: Literal["principal", "ponderado"]) -> pd.DataFrame:
        """Devuelve la variante solicitada con columna `variante` añadida.

        Útil para persistir long format en `casuistica_capitulos.parquet`.
        """
        if variante == VARIANTE_PRINCIPAL:
            df = self.vector_capitulos_principal()
        elif variante == VARIANTE_PONDERADO:
            df = self.vector_capitulos_ponderado()
        else:
            raise ValueError(f"variante inválida: {variante!r}")
        df.insert(1, "variante", variante)
        return df

    # ------------------------------------------------ vector secciones CIE-9
    def vector_secciones(self) -> pd.DataFrame:
        """Vector de secciones CIE-9-MC sobre todos los procedimientos disponibles.

        Req 6.3, 6.5, 6.7.

        Returns
        -------
        pd.DataFrame
            Columnas: [COD_HOSPITAL, sec_NN..., SEC_DESCONOCIDA]. Suma=1.0±1e-6.
            Solo egresos con al menos un PROCEDIMIENTOk no nulo (Req 6.5).
        """
        hospitales = self._serie_hosp_elegibles("COD_HOSPITAL")

        # Se acumulan los conteos por (hospital, sección) columna por columna.
        # Los egresos sin ningún procedimiento no aportan códigos, de modo que
        # el resultado es idéntico a filtrarlos de antemano, pero sin construir
        # una matriz booleana de 30 columnas sobre millones de registros.
        acumulado: pd.Series | None = None
        for col in columnas_codificadas(self.df_grd, "PROCEDIMIENTO"):
            codigos = self._serie_hosp_elegibles(col)
            presentes = codigos_presentes(codigos).to_numpy()
            if not presentes.any():
                continue
            sub = pd.DataFrame({
                "COD_HOSPITAL": hospitales.to_numpy()[presentes],
                "_sec": mapear_codigos(
                    codigos[presentes], self.mapper_cie9, SEC_DESCONOCIDA
                ),
            })
            conteo = sub.groupby(["COD_HOSPITAL", "_sec"]).size()
            acumulado = conteo if acumulado is None else acumulado.add(conteo, fill_value=0)

        if acumulado is None:
            cols = ["COD_HOSPITAL"] + [SEC_DESCONOCIDA]
            return pd.DataFrame(columns=cols)

        acumulado.index.names = ["COD_HOSPITAL", "_sec"]
        crosstab = acumulado.unstack(fill_value=0.0)

        # Asegurar todas las secciones esperadas + SEC_DESCONOCIDA
        secciones_esperadas = self.mapper_cie9.secciones + [SEC_DESCONOCIDA]
        crosstab = crosstab.reindex(columns=sorted(set(secciones_esperadas)), fill_value=0)

        proporciones = crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0.0)

        # Validación dura (Req 6.7)
        sumas = proporciones.sum(axis=1)
        if not np.allclose(sumas, 1.0, atol=1e-6):
            mal = sumas[~np.isclose(sumas, 1.0, atol=1e-6)]
            raise AssertionError(
                f"vector_secciones: filas que no suman 1.0:\n{mal}"
            )

        return proporciones.sort_index().reset_index()

    # ----------------------------------------------------- vector Top-20 GRDs
    def top20_grds_nacionales(self) -> list[str]:
        """Identifica los 20 GRDs más frecuentes en hospitalización 2019-2024.

        Req 7.1.

        Returns
        -------
        list[str]
            Lista ordenada (de mayor a menor frecuencia) de los 20 GRDs.
        """
        df = self._df_hosp_elegibles(["IR_29301_COD_GRD"])
        top = df["IR_29301_COD_GRD"].value_counts().head(TOP_K_GRDS)
        return top.index.tolist()

    def vector_top20(self, top20: list[str] | None = None) -> pd.DataFrame:
        """Vector de proporciones del Top-20 + columna OTROS_GRDS.

        Req 7.3, 7.4, 7.5.
        """
        if top20 is None:
            top20 = self.top20_grds_nacionales()
        if len(top20) != TOP_K_GRDS:
            # Si hay menos de 20 GRDs distintos en hospitalización (improbable),
            # rellenar con sentinels para mantener la dimensionalidad.
            top20 = list(top20) + [f"__GRD_VACIO_{i}" for i in range(TOP_K_GRDS - len(top20))]

        df = self._df_hosp_elegibles(["COD_HOSPITAL", "IR_29301_COD_GRD"])
        es_top20 = df["IR_29301_COD_GRD"].isin(top20)
        df_clas = df.assign(
            _grd_label=np.where(
                es_top20, df["IR_29301_COD_GRD"].astype(str), "OTROS_GRDS"
            )
        )

        crosstab = pd.crosstab(df_clas["COD_HOSPITAL"], df_clas["_grd_label"])

        # Mapear nombres de columnas: top20[0] → top20_grd_01, ..., OTROS_GRDS
        col_rename = {grd: f"top20_grd_{i+1:02d}" for i, grd in enumerate(top20)}
        crosstab = crosstab.rename(columns=col_rename)
        col_orden = [f"top20_grd_{i+1:02d}" for i in range(TOP_K_GRDS)] + ["OTROS_GRDS"]
        crosstab = crosstab.reindex(columns=col_orden, fill_value=0)

        proporciones = crosstab.div(crosstab.sum(axis=1), axis=0).fillna(0.0)

        # Validación (Req 7.5)
        sumas = proporciones.sum(axis=1)
        if not np.allclose(sumas, 1.0, atol=1e-6):
            mal = sumas[~np.isclose(sumas, 1.0, atol=1e-6)]
            raise AssertionError(
                f"vector_top20: filas que no suman 1.0:\n{mal}"
            )

        return proporciones.sort_index().reset_index()

    def top20_grds_resumen(self, top20: list[str] | None = None) -> pd.DataFrame:
        """Tabla resumen para reports/tables/top20_grds_nacionales.csv (Req 7.2).

        Returns
        -------
        pd.DataFrame
            Columnas: [COD_GRD, frecuencia_nacional, pct_total_nacional, ranking].
        """
        if top20 is None:
            top20 = self.top20_grds_nacionales()
        df = self._df_hosp_elegibles(["IR_29301_COD_GRD"])
        total_egresos = len(df)
        counts = df["IR_29301_COD_GRD"].value_counts()
        rows = []
        for ranking, grd in enumerate(top20, start=1):
            freq = int(counts.get(grd, 0))
            rows.append({
                "ranking": ranking,
                "COD_GRD": grd,
                "frecuencia_nacional": freq,
                "pct_total_nacional": (freq / total_egresos * 100) if total_egresos else 0.0,
            })
        return pd.DataFrame(rows)
