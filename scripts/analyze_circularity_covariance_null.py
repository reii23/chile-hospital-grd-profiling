"""Nula multivariada con covarianza preservada para la circularidad Ward--Kruskal.

Extiende ``analyze_clustering_circularity.py``. La prueba original permuta cada
columna de forma independiente, preservando márgenes y destruyendo toda
covarianza. Este script usa una nula más exigente: simula matrices normales
multivariadas con la media y la matriz de covarianza observadas después de las
transformaciones y el escalado robusto.

La nula preserva en esperanza la dependencia lineal entre variables, pero no
conserva observaciones, etiquetas hospitalarias ni una partición de grupos. En
cada réplica se ejecutan Ward K=4 y Kruskal--Wallis con corrección
Benjamini--Hochberg sobre las 28 variables interpretables.

Salidas (reports/tables/):
  - nula_covarianza_circularidad_resumen.csv
  - nula_covarianza_circularidad_detalle.csv
  - nula_covarianza_circularidad_metricas.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import kruskal
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import PROCESSED_DIR, TABLES_DIR

ALPHA = 0.05
RANDOM_STATE = 20260806
N_REPETICIONES = int(os.environ.get("N_REP", "500"))


def transformar_y_escalar(datos: pd.DataFrame, columnas: list[str]) -> np.ndarray:
    """Aplica las mismas transformaciones de la matriz antes de Ward."""
    transformados = datos[columnas].copy()
    for col in LOG1P_COLS:
        if col in transformados.columns:
            transformados[col] = np.log1p(transformados[col].clip(lower=0))
    return RobustScaler().fit_transform(transformados.to_numpy(dtype=float))


def contar_significativas(valores: np.ndarray, labels: np.ndarray) -> tuple[int, np.ndarray]:
    """Devuelve el número de variables con BH <= ALPHA y sus p-valores BH."""
    p_valores = []
    for idx in range(valores.shape[1]):
        muestras = [valores[labels == grupo, idx] for grupo in np.unique(labels)]
        try:
            _, p_valor = kruskal(*muestras)
        except ValueError:
            p_valor = np.nan
        p_valores.append(p_valor)

    p_array = np.asarray(p_valores, dtype=float)
    validos = ~np.isnan(p_array)
    p_bh = np.full_like(p_array, np.nan)
    if validos.any():
        _, p_bh[validos], _, _ = multipletests(
            p_array[validos], alpha=ALPHA, method="fdr_bh"
        )
    return int(np.nansum(p_bh <= ALPHA)), p_bh


def raiz_covarianza_positiva(covarianza: np.ndarray) -> np.ndarray:
    """Obtiene una raíz simétrica de una covarianza, tolerando error numérico."""
    valores, vectores = np.linalg.eigh(covarianza)
    valores = np.clip(valores, a_min=0.0, a_max=None)
    return (vectores * np.sqrt(valores)) @ vectores.T


def main() -> None:
    print("=" * 80)
    print("NULA CON COVARIANZA PRESERVADA: Ward K=4 -> Kruskal-Wallis + BH")
    print("=" * 80)

    matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
    matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)

    excluir_interpretables = {"COD_HOSPITAL", "peso_medio_cma_imputado"}
    columnas_interpretables = [
        columna
        for columna in matriz.columns
        if columna not in excluir_interpretables
        and not columna.startswith("dim_")
        and pd.api.types.is_numeric_dtype(matriz[columna])
    ]
    columnas_covarianza = [
        columna
        for columna in matriz.columns
        if columna not in {"COD_HOSPITAL", "peso_medio_cma_imputado"}
        and pd.api.types.is_numeric_dtype(matriz[columna])
    ]
    columnas_clustering = [
        columna for columna in matriz.columns if columna not in COLS_DROP_CLUSTERING
    ]

    if matriz[columnas_covarianza].isna().any().any():
        raise ValueError("La nula multivariada requiere variables numéricas completas.")

    print(f"\n[1/4] Hospitales: {len(matriz)}")
    print(f"  Variables de covarianza: {len(columnas_covarianza)}")
    print(f"  Variables Ward: {len(columnas_clustering)}")
    print(f"  Variables Kruskal-Wallis: {len(columnas_interpretables)}")

    X_cov = transformar_y_escalar(matriz, columnas_covarianza)
    indices_interpretables = [columnas_covarianza.index(c) for c in columnas_interpretables]
    indices_clustering = [columnas_covarianza.index(c) for c in columnas_clustering]

    etiquetas_reales = fcluster(linkage(X_cov[:, indices_clustering], method="ward"), t=4, criterion="maxclust")
    n_sig_real, _ = contar_significativas(X_cov[:, indices_interpretables], etiquetas_reales)
    print(f"\n[2/4] Baseline real: {n_sig_real}/{len(columnas_interpretables)} variables significativas")

    media = X_cov.mean(axis=0)
    covarianza = np.cov(X_cov, rowvar=False, ddof=1)
    raiz_covarianza = raiz_covarianza_positiva(covarianza)
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"\n[3/4] Ejecutando {N_REPETICIONES} réplicas de la nula multivariada...")
    resumen: list[dict[str, float | int]] = []
    detalle: list[dict[str, float | int | str]] = []
    for replica in range(N_REPETICIONES):
        ruido = rng.standard_normal(size=X_cov.shape)
        X_nula = ruido @ raiz_covarianza.T + media
        etiquetas = fcluster(linkage(X_nula[:, indices_clustering], method="ward"), t=4, criterion="maxclust")
        n_sig, p_bh = contar_significativas(X_nula[:, indices_interpretables], etiquetas)

        cov_muestral = np.cov(X_nula, rowvar=False, ddof=1)
        error_covarianza = np.linalg.norm(cov_muestral - covarianza, ord="fro") / np.linalg.norm(covarianza, ord="fro")
        resumen.append(
            {
                "replica": replica,
                "n_clusters_efectivo": int(len(np.unique(etiquetas))),
                "n_variables_significativas": n_sig,
                "error_relativo_covarianza_frobenius": error_covarianza,
            }
        )
        detalle.extend(
            {
                "replica": replica,
                "variable": variable,
                "p_valor_ajustado_BH": p_valor,
            }
            for variable, p_valor in zip(columnas_interpretables, p_bh)
        )
        if (replica + 1) % 100 == 0:
            print(f"  ... {replica + 1}/{N_REPETICIONES} réplicas completadas")

    df_resumen = pd.DataFrame(resumen)
    df_detalle = pd.DataFrame(detalle)
    ruta_resumen = TABLES_DIR / "nula_covarianza_circularidad_resumen.csv"
    ruta_detalle = TABLES_DIR / "nula_covarianza_circularidad_detalle.csv"
    df_resumen.to_csv(ruta_resumen, index=False)
    df_detalle.to_csv(ruta_detalle, index=False)

    n_extremas = int((df_resumen["n_variables_significativas"] >= n_sig_real).sum())
    p_empirico = (n_extremas + 1) / (N_REPETICIONES + 1)
    proporcion_extrema = n_extremas / N_REPETICIONES
    cuantiles = df_resumen["n_variables_significativas"].quantile([0.5, 0.9, 0.95, 0.99])
    metricas = pd.DataFrame(
        [{
            "n_repeticiones": N_REPETICIONES,
            "n_variables_significativas_real": n_sig_real,
            "mediana_nula": cuantiles.loc[0.5],
            "percentil_95_nula": cuantiles.loc[0.95],
            "maximo_nula": int(df_resumen["n_variables_significativas"].max()),
            "repeticiones_nula_igual_o_mayor_real": n_extremas,
            "p_valor_empirico_con_correccion": p_empirico,
            "error_relativo_covarianza_frobenius_mediano": df_resumen[
                "error_relativo_covarianza_frobenius"
            ].median(),
        }]
    )
    ruta_metricas = TABLES_DIR / "nula_covarianza_circularidad_metricas.csv"
    metricas.to_csv(ruta_metricas, index=False)
    print("\n[4/4] Resumen")
    print(f"  Real: {n_sig_real}/{len(columnas_interpretables)}")
    print("  Cuantiles nula:")
    print(cuantiles.to_string())
    print(f"  Máximo nula: {df_resumen['n_variables_significativas'].max()}")
    print(f"  P(nula >= real): {proporcion_extrema:.4f}")
    print(f"  p empírico con corrección: {p_empirico:.4f}")
    print(
        "  Error relativo de covarianza (mediana): "
        f"{df_resumen['error_relativo_covarianza_frobenius'].median():.4f}"
    )
    print(
        f"\nPersistido: {ruta_resumen.name}, {ruta_detalle.name}, "
        f"{ruta_metricas.name}"
    )


if __name__ == "__main__":
    main()
