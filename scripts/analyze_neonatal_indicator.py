"""Auditoria del indicador técnico ``tasa_partos``.

La columna histórica ``tasa_partos`` no contabilizaba partos: representa la
proporción de egresos de hospitalización con CONDICIONDEALTANEONATO1 informada.
Este análisis conserva el nombre técnico de la matriz para compatibilidad y
cuantifica su asociación con variables demográfico-obstétricas relacionadas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MATRIZ = ROOT / "data" / "processed" / "hospital_matrix.parquet"
TABLES = ROOT / "reports" / "tables"

COLUMNA_TECNICA = "tasa_partos"
VARIABLES_RELACIONADAS = {
    "pct_obstetrica_ingreso": "Proporción de ingresos obstétricos",
    "pct_femenino_fertil": "Proporción de egresos de mujeres fértiles",
    "tasa_prematurez": "Proporción de egresos neonatales con peso < 2.500 g",
    "pct_pediatrico": "Proporción de egresos pediátricos",
    "edad_mediana": "Edad mediana al ingreso",
}


def main() -> None:
    matriz = pd.read_parquet(MATRIZ)
    columnas = [COLUMNA_TECNICA, *VARIABLES_RELACIONADAS]
    faltantes = set(columnas) - set(matriz.columns)
    if faltantes:
        raise ValueError(f"Columnas ausentes en la matriz: {sorted(faltantes)}")

    filas: list[dict[str, float | str | int]] = []
    for variable, descripcion in VARIABLES_RELACIONADAS.items():
        pares = matriz[[COLUMNA_TECNICA, variable]].dropna()
        filas.append(
            {
                "indicador_tecnico": COLUMNA_TECNICA,
                "etiqueta_publica": "proporcion_egresos_neonatales",
                "variable_relacionada": variable,
                "descripcion_variable": descripcion,
                "n_hospitales": len(pares),
                "pearson_r": pares[COLUMNA_TECNICA].corr(pares[variable], method="pearson"),
                "spearman_rho": pares[COLUMNA_TECNICA].corr(pares[variable], method="spearman"),
                "redundancia_rho_ge_0_90": abs(pares[COLUMNA_TECNICA].corr(pares[variable], method="spearman")) >= 0.90,
            }
        )

    resultado = pd.DataFrame(filas).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    salida = TABLES / "correlaciones_indicador_neonatal.csv"
    resultado.to_csv(salida, index=False)

    print("AUDITORÍA R-05 — INDICADOR NEONATAL")
    print("Columna técnica: tasa_partos")
    print("Etiqueta de informe: proporción de egresos neonatales")
    print(resultado.to_string(index=False, float_format="%.3f"))
    print(f"\nPersistido: {salida.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
