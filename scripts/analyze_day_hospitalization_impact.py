"""Auditoría R-11: impacto de la hospitalización diurna en el perfil hospitalario.

El ETL histórico reúne hospitalización convencional, urgencia y hospitalización
 diurna en ``MODALIDAD == 'HOSPITALIZACION'``. Este análisis recupera
``TIPO_ACTIVIDAD`` desde el Parquet canónico y cuantifica, para los 65 hospitales
incluidos, cuánto cambian los indicadores de gestión si se conserva solo la
actividad con pernoctación (convencional + urgencia).

Salidas:
- reports/tables/hospitalizacion_diurna_por_establecimiento.csv
- reports/tables/hospitalizacion_diurna_resumen.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
GRD_PATH = ROOT / "data" / "processed" / "grd_filtrado.parquet"
MATRIX_PATH = ROOT / "data" / "processed" / "hospital_matrix.parquet"
TABLES_DIR = ROOT / "reports" / "tables"

ACTIVIDAD_DIURNA = "HOSPITALIZACIÓN DIURNA"
ACTIVIDADES_NOCTURNAS = {
    "HOSPITALIZACIÓN",
    "HOSPITALIZACIÓN EN URGENCIA",
}
COLUMNAS = [
    "COD_HOSPITAL",
    "TIPO_ACTIVIDAD",
    "MODALIDAD",
    "DIAS_ESTADA",
    "IR_29301_PESO",
    "IR_29301_SEVERIDAD",
    "IR_29301_MORTALIDAD",
]
METRICAS = {
    "estancia_media": "DIAS_ESTADA",
    "peso_medio_grd": "IR_29301_PESO",
    "severidad_media": "IR_29301_SEVERIDAD",
    "mortalidad_media": "IR_29301_MORTALIDAD",
}


def porcentaje(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    return np.where(denominador > 0, numerador / denominador, np.nan)


def agregar_lote(lote: pd.DataFrame, elegibles: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume un lote sin retener egresos individuales en memoria."""
    lote = lote[lote["COD_HOSPITAL"].astype(str).isin(elegibles)].copy()
    lote = lote[lote["MODALIDAD"].eq("HOSPITALIZACION")].copy()
    lote["COD_HOSPITAL"] = lote["COD_HOSPITAL"].astype(str)
    lote["es_diurna"] = lote["TIPO_ACTIVIDAD"].eq(ACTIVIDAD_DIURNA)
    lote["es_nocturna"] = lote["TIPO_ACTIVIDAD"].isin(ACTIVIDADES_NOCTURNAS)

    conteos = lote.groupby("COD_HOSPITAL").agg(
        n_hospitalizacion_actual=("TIPO_ACTIVIDAD", "size"),
        n_diurna=("es_diurna", "sum"),
        n_nocturna=("es_nocturna", "sum"),
    )

    resumenes: list[pd.DataFrame] = []
    for universo, mascara in {
        "actual": pd.Series(True, index=lote.index),
        "nocturna": lote["es_nocturna"],
    }.items():
        sub = lote.loc[mascara, ["COD_HOSPITAL", *METRICAS.values()]]
        agg = sub.groupby("COD_HOSPITAL").agg(["sum", "count"])
        agg.columns = [f"{variable}_{estadistico}_{universo}" for variable, estadistico in agg.columns]
        resumenes.append(agg)

    return conteos, pd.concat(resumenes, axis=1)


def main() -> None:
    if not GRD_PATH.exists():
        raise FileNotFoundError(f"No existe el insumo requerido: {GRD_PATH}")

    elegibles = set(pd.read_parquet(MATRIX_PATH, columns=["COD_HOSPITAL"])["COD_HOSPITAL"].astype(str))
    if len(elegibles) != 65:
        raise ValueError(f"Se esperaban 65 hospitales elegibles; se encontraron {len(elegibles)}")

    acumulado_conteos: pd.DataFrame | None = None
    acumulado_metricas: pd.DataFrame | None = None
    archivo = pq.ParquetFile(GRD_PATH)
    for lote in archivo.iter_batches(batch_size=250_000, columns=COLUMNAS):
        conteos, metricas = agregar_lote(lote.to_pandas(), elegibles)
        acumulado_conteos = conteos if acumulado_conteos is None else acumulado_conteos.add(conteos, fill_value=0)
        acumulado_metricas = metricas if acumulado_metricas is None else acumulado_metricas.add(metricas, fill_value=0)

    if acumulado_conteos is None or acumulado_metricas is None:
        raise RuntimeError("No se procesaron egresos elegibles")

    resultado = acumulado_conteos.join(acumulado_metricas, how="left").reindex(sorted(elegibles))
    resultado.index.name = "COD_HOSPITAL"
    resultado["proporcion_diurna_hospitalizacion"] = porcentaje(
        resultado["n_diurna"], resultado["n_hospitalizacion_actual"]
    )

    for nombre, columna in METRICAS.items():
        suma_actual = resultado[f"{columna}_sum_actual"]
        n_actual = resultado[f"{columna}_count_actual"]
        suma_nocturna = resultado[f"{columna}_sum_nocturna"]
        n_nocturna = resultado[f"{columna}_count_nocturna"]
        resultado[f"{nombre}_actual"] = porcentaje(suma_actual, n_actual)
        resultado[f"{nombre}_sin_diurna"] = porcentaje(suma_nocturna, n_nocturna)
        resultado[f"cambio_rel_{nombre}_pct"] = 100 * porcentaje(
            resultado[f"{nombre}_sin_diurna"] - resultado[f"{nombre}_actual"],
            resultado[f"{nombre}_actual"],
        )

    columnas_salida = [
        "n_hospitalizacion_actual", "n_nocturna", "n_diurna",
        "proporcion_diurna_hospitalizacion",
        *[col for nombre in METRICAS for col in (
            f"{nombre}_actual", f"{nombre}_sin_diurna", f"cambio_rel_{nombre}_pct"
        )],
    ]
    detalle = resultado[columnas_salida].reset_index().sort_values(
        "proporcion_diurna_hospitalizacion", ascending=False
    )
    detalle.to_csv(TABLES_DIR / "hospitalizacion_diurna_por_establecimiento.csv", index=False)

    total_actual = detalle["n_hospitalizacion_actual"].sum()
    total_diurna = detalle["n_diurna"].sum()
    cambios = [f"cambio_rel_{nombre}_pct" for nombre in METRICAS]
    resumen_filas = [
        {"estadistico": "hospitales_elegibles", "valor": len(detalle)},
        {"estadistico": "egresos_hospitalizacion_actual", "valor": int(total_actual)},
        {"estadistico": "egresos_hospitalizacion_diurna", "valor": int(total_diurna)},
        {"estadistico": "proporcion_diurna_nacional_pct", "valor": 100 * total_diurna / total_actual},
        {"estadistico": "mediana_proporcion_diurna_pct", "valor": 100 * detalle["proporcion_diurna_hospitalizacion"].median()},
        {"estadistico": "p95_proporcion_diurna_pct", "valor": 100 * detalle["proporcion_diurna_hospitalizacion"].quantile(0.95)},
        {"estadistico": "max_proporcion_diurna_pct", "valor": 100 * detalle["proporcion_diurna_hospitalizacion"].max()},
        {"estadistico": "hospitales_sobre_5pct_diurna", "valor": int((detalle["proporcion_diurna_hospitalizacion"] > 0.05).sum())},
        {"estadistico": "hospitales_sobre_10pct_diurna", "valor": int((detalle["proporcion_diurna_hospitalizacion"] > 0.10).sum())},
    ]
    for cambio in cambios:
        resumen_filas.extend([
            {"estadistico": f"mediana_{cambio}", "valor": detalle[cambio].median()},
            {"estadistico": f"max_abs_{cambio}", "valor": detalle[cambio].abs().max()},
        ])
    resumen = pd.DataFrame(resumen_filas)
    resumen.to_csv(TABLES_DIR / "hospitalizacion_diurna_resumen.csv", index=False)

    print("AUDITORÍA R-11 — HOSPITALIZACIÓN DIURNA")
    print(f"Hospitales elegibles: {len(detalle)}")
    print(f"Diurna nacional: {total_diurna:,}/{total_actual:,} ({100 * total_diurna / total_actual:.3f}%)")
    print(
        "Proporción por hospital (mediana / P95 / máxima): "
        f"{100 * detalle['proporcion_diurna_hospitalizacion'].median():.3f}% / "
        f"{100 * detalle['proporcion_diurna_hospitalizacion'].quantile(0.95):.3f}% / "
        f"{100 * detalle['proporcion_diurna_hospitalizacion'].max():.3f}%"
    )
    print(f"Hospitales >5%: {(detalle['proporcion_diurna_hospitalizacion'] > 0.05).sum()}; "
          f">10%: {(detalle['proporcion_diurna_hospitalizacion'] > 0.10).sum()}")
    print("\nCinco mayores proporciones:")
    print(detalle[["COD_HOSPITAL", "n_diurna", "proporcion_diurna_hospitalizacion", *cambios]].head().to_string(index=False, float_format="%.3f"))
    print("\nCambios relativos al retirar diurna (mediana / máximo absoluto):")
    for cambio in cambios:
        print(f"  {cambio}: {detalle[cambio].median():.3f}% / {detalle[cambio].abs().max():.3f}%")
    print(f"\nPersistidos: {TABLES_DIR / 'hospitalizacion_diurna_por_establecimiento.csv'}")
    print(f"Persistidos: {TABLES_DIR / 'hospitalizacion_diurna_resumen.csv'}")


if __name__ == "__main__":
    main()
