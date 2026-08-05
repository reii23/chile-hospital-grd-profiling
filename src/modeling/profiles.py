"""Generador de perfiles clínicos por cluster (Req 12)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ETIQUETAS_CAPITULO = {
    "cap_01": "Infeccioso", "cap_02": "Oncológico", "cap_03": "Hematológico",
    "cap_04": "Endocrino-metabólico", "cap_05": "Salud mental",
    "cap_06": "Neurológico", "cap_07": "Oftalmológico", "cap_08": "Otorrino",
    "cap_09": "Cardiovascular", "cap_10": "Respiratorio", "cap_11": "Digestivo",
    "cap_12": "Dermatológico", "cap_13": "Musculoesquelético",
    "cap_14": "Genitourinario", "cap_15": "Obstétrico", "cap_16": "Perinatal",
    "cap_17": "Congénito", "cap_18": "Sintomático", "cap_19": "Traumatología",
    "cap_20": "Causas externas", "cap_21": "Contacto sanitario",
    "cap_22": "Especiales",
}

ETIQUETAS_SECCION = {
    "sec_00": "no clasificado", "sec_01": "neuroquirúrgico",
    "sec_02": "endocrino-quirúrgico", "sec_03": "oftalmológico-quirúrgico",
    "sec_03A": "diagnóstico misceláneo", "sec_04": "otorrino-quirúrgico",
    "sec_05": "cabeza/cuello", "sec_06": "respiratorio-quirúrgico",
    "sec_07": "cardiovascular-quirúrgico", "sec_08": "hemato-linfático",
    "sec_09": "digestivo-quirúrgico", "sec_10": "urológico-quirúrgico",
    "sec_11": "genital-masculino", "sec_12": "ginecológico",
    "sec_13": "obstétrico-quirúrgico", "sec_14": "ortopédico",
    "sec_15": "dermatológico-quirúrgico", "sec_16": "diagnóstico",
}


@dataclass
class Generador_Perfiles:
    matriz_integrada: pd.DataFrame
    asignaciones: pd.DataFrame
    casuistica_capitulos: pd.DataFrame
    casuistica_procedimientos: pd.DataFrame
    casuistica_top20: pd.DataFrame
    df_grd_top20_nombres: pd.DataFrame | None = None

    def centroides_originales(self) -> pd.DataFrame:
        df = self.matriz_integrada.merge(self.asignaciones, on="COD_HOSPITAL")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if "cluster" in num_cols:
            num_cols.remove("cluster")
        return df.groupby("cluster")[num_cols].mean().reset_index()

    def _top_n(self, df_cas, n, prefijo):
        merged = df_cas.merge(self.asignaciones, on="COD_HOSPITAL")
        cols = [c for c in df_cas.columns if c.startswith(prefijo)]
        result = {}
        for cluster, sub in merged.groupby("cluster"):
            medias = sub[cols].mean().sort_values(ascending=False)
            result[int(cluster)] = list(medias.head(n).items())
        return result

    def dominantes_por_cluster(self) -> pd.DataFrame:
        top_caps = self._top_n(self.casuistica_capitulos, 3, "cap_")
        top_secs = self._top_n(self.casuistica_procedimientos, 3, "sec_")
        top_grds = self._top_n(self.casuistica_top20, 5, "top20_grd_")
        n_hosp = self.asignaciones["cluster"].value_counts()

        rows = []
        for cluster in sorted(self.asignaciones["cluster"].unique()):
            caps = top_caps.get(cluster, [])
            secs = top_secs.get(cluster, [])
            grds = top_grds.get(cluster, [])
            etiqueta = self._proponer_etiqueta(caps, secs)
            rows.append({
                "cluster": int(cluster),
                "n_hospitales": int(n_hosp.get(cluster, 0)),
                "capitulos_dominantes": ";".join(f"{c}({p:.1%})" for c, p in caps),
                "secciones_dominantes": ";".join(f"{s}({p:.1%})" for s, p in secs),
                "grds_dominantes": ";".join(f"{g}({p:.1%})" for g, p in grds),
                "etiqueta_propuesta": etiqueta,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _proponer_etiqueta(caps, secs):
        if not caps:
            return "Sin perfil claro"
        cap_principal = caps[0][0]
        cap_label = ETIQUETAS_CAPITULO.get(cap_principal, cap_principal)
        if secs:
            sec_principal = secs[0][0]
            sec_label = ETIQUETAS_SECCION.get(sec_principal, sec_principal)
            return f"{cap_label} ({sec_label})"
        return cap_label

    def etiquetas_propuestas(self) -> dict[int, str]:
        df = self.dominantes_por_cluster()
        return dict(zip(df["cluster"], df["etiqueta_propuesta"]))
