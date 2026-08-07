"""Valida la estabilidad de la subtipología exploratoria de Nivel 2.

Aplica al núcleo generalista (C1, n=54) el mismo protocolo de submuestreo
utilizado para Nivel 1: 100 réplicas sin reemplazo, con 80 % de hospitales,
repreprocesamiento y clustering Ward. Evalúa K_sub=2 y K_sub=3, soluciones
compactas respaldadas por las métricas internas, y K_sub=10, partición
exploratoria originalmente reportada.

Para cada K se reportan Silhouette, ARI/NMI frente a la partición de
referencia, coasignación, estabilidad individual y Jaccard por subgrupo.
Las salidas llevan el prefijo ``estabilidad_nivel2_`` para no sobrescribir
los artefactos canónicos de Nivel 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS
from src.utils.io import PROCESSED_DIR, TABLES_DIR

RANDOM_STATE = 42
N_ITERACIONES = int(os.environ.get("N_REP", "100"))
FRAC_SUBMUESTRA = 0.80
K_EVALUADOS = (2, 3, 10)


def preprocesar(matriz_sub: pd.DataFrame, columnas: list[str]) -> tuple[np.ndarray, list[str]]:
    """Aplica el preprocesamiento canónico dentro de cada submuestra."""
    feats = matriz_sub[columnas].copy().fillna(0.0)
    ids = matriz_sub["COD_HOSPITAL"].astype(str).tolist()
    for col in LOG1P_COLS:
        if col in feats.columns:
            feats[col] = np.log1p(feats[col].clip(lower=0))
    return RobustScaler().fit_transform(feats.values), ids


def etiquetas_ward(X: np.ndarray, k: int) -> np.ndarray:
    """Obtiene una partición Ward con exactamente k grupos."""
    return fcluster(linkage(X, method="ward"), t=k, criterion="maxclust")


def jaccard_mejor_emparejamiento(
    ids_sub: list[str], labels_ref: pd.Series, labels_sub: np.ndarray, cluster: int
) -> float:
    """Calcula el mejor Jaccard del grupo de referencia dentro de una réplica."""
    miembros_ref = {h for h in ids_sub if labels_ref[h] == cluster}
    if not miembros_ref:
        return np.nan

    mejor = 0.0
    for cluster_sub in np.unique(labels_sub):
        miembros_sub = {
            ids_sub[i] for i, etiqueta in enumerate(labels_sub) if etiqueta == cluster_sub
        }
        union = miembros_ref | miembros_sub
        mejor = max(mejor, len(miembros_ref & miembros_sub) / len(union))
    return mejor


def evaluar_k(
    matriz: pd.DataFrame,
    ids: list[str],
    columnas: list[str],
    labels_referencia: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evalúa estabilidad por submuestreo para un K_sub dado."""
    n_total = len(ids)
    n_sub = int(round(n_total * FRAC_SUBMUESTRA))
    labels_ref = pd.Series(labels_referencia, index=ids)
    clusters_ref = sorted(labels_ref.unique())
    indexador = {hospital: i for i, hospital in enumerate(ids)}

    co_juntos = np.zeros((n_total, n_total), dtype=np.int64)
    co_mismo = np.zeros((n_total, n_total), dtype=np.int64)
    filas = []
    jaccard = {cluster: [] for cluster in clusters_ref}

    for replica in range(N_ITERACIONES):
        idx_sub = rng.choice(n_total, size=n_sub, replace=False)
        ids_sub_set = {ids[i] for i in idx_sub}
        matriz_sub = matriz[matriz["COD_HOSPITAL"].isin(ids_sub_set)].copy()
        X_sub, ids_sub = preprocesar(matriz_sub, columnas)
        labels_sub = etiquetas_ward(X_sub, k)
        ref_sub = labels_ref.reindex(ids_sub).to_numpy()

        filas.append(
            {
                "K_sub": k,
                "replica": replica + 1,
                "n_sub": n_sub,
                "silhouette": silhouette_score(X_sub, labels_sub),
                "ari_vs_referencia": adjusted_rand_score(ref_sub, labels_sub),
                "nmi_vs_referencia": normalized_mutual_info_score(ref_sub, labels_sub),
            }
        )

        for i, hospital_i in enumerate(ids_sub):
            indice_i = indexador[hospital_i]
            for j in range(i + 1, len(ids_sub)):
                indice_j = indexador[ids_sub[j]]
                co_juntos[indice_i, indice_j] += 1
                co_juntos[indice_j, indice_i] += 1
                if labels_sub[i] == labels_sub[j]:
                    co_mismo[indice_i, indice_j] += 1
                    co_mismo[indice_j, indice_i] += 1

        for cluster in clusters_ref:
            valor = jaccard_mejor_emparejamiento(ids_sub, labels_ref, labels_sub, cluster)
            if not np.isnan(valor):
                jaccard[cluster].append(valor)

    df_replicas = pd.DataFrame(filas)
    with np.errstate(invalid="ignore", divide="ignore"):
        proporcion = np.where(co_juntos > 0, co_mismo / co_juntos, np.nan)
    np.fill_diagonal(proporcion, 1.0)
    df_coasignacion = pd.DataFrame(proporcion, index=ids, columns=ids)

    filas_individual = []
    for hospital in ids:
        indice = indexador[hospital]
        cluster = labels_ref[hospital]
        vecinos = [
            indexador[otro]
            for otro in ids
            if otro != hospital and labels_ref[otro] == cluster
        ]
        valores = proporcion[indice, vecinos]
        valores = valores[~np.isnan(valores)]
        filas_individual.append(
            {
                "K_sub": k,
                "COD_HOSPITAL": hospital,
                "subgrupo_referencia": f"S{cluster - 1}",
                "n_vecinos_subgrupo": len(vecinos),
                "estabilidad_individual": float(np.mean(valores)) if len(valores) else np.nan,
            }
        )
    df_individual = pd.DataFrame(filas_individual)

    filas_grupo = []
    for cluster in clusters_ref:
        grupo = df_individual[df_individual["subgrupo_referencia"] == f"S{cluster - 1}"]
        valores_jaccard = jaccard[cluster]
        filas_grupo.append(
            {
                "K_sub": k,
                "subgrupo": f"S{cluster - 1}",
                "n_hospitales": len(grupo),
                "estabilidad_individual_media": grupo["estabilidad_individual"].mean(),
                "estabilidad_individual_mediana": grupo["estabilidad_individual"].median(),
                "jaccard_promedio": float(np.mean(valores_jaccard)),
                "jaccard_mediana": float(np.median(valores_jaccard)),
            }
        )
    df_grupos = pd.DataFrame(filas_grupo)

    resumen = {"K_sub": k}
    for nombre, columna in (
        ("silhouette", "silhouette"),
        ("ari", "ari_vs_referencia"),
        ("nmi", "nmi_vs_referencia"),
    ):
        cuantiles = df_replicas[columna].quantile([0.025, 0.5, 0.975])
        resumen[f"{nombre}_p025"] = cuantiles.loc[0.025]
        resumen[f"{nombre}_mediana"] = cuantiles.loc[0.5]
        resumen[f"{nombre}_p975"] = cuantiles.loc[0.975]

    return df_replicas, df_coasignacion, df_individual, df_grupos, resumen


def main() -> None:
    print("=" * 80)
    print("ESTABILIDAD POR SUBMUESTREO — NIVEL 2")
    print("=" * 80)

    matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
    matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)
    asignacion = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype=str)
    asignacion["COD_HOSPITAL"] = asignacion["COD_HOSPITAL"].astype(str)

    nucleo_ids = asignacion.loc[
        asignacion["nivel1_etiqueta"] == "MAINSTREAM", "COD_HOSPITAL"
    ].tolist()
    matriz_nucleo = matriz[matriz["COD_HOSPITAL"].isin(nucleo_ids)].copy()
    matriz_nucleo = matriz_nucleo.set_index("COD_HOSPITAL").loc[nucleo_ids].reset_index()
    columnas = [col for col in matriz.columns if col not in COLS_DROP_CLUSTERING]
    n_total = len(nucleo_ids)
    n_sub = int(round(n_total * FRAC_SUBMUESTRA))

    print(f"Núcleo generalista: n={n_total}; submuestra: n={n_sub}; réplicas={N_ITERACIONES}")
    print(f"Soluciones evaluadas: K_sub={K_EVALUADOS}")

    # La referencia debe reproducir exactamente la solución histórica: el
    # escalado se ajustó sobre los 65 hospitales y luego se extrajo C1. En
    # cambio, cada réplica reescala solo su submuestra dentro de evaluar_k,
    # como perturbación deliberada del protocolo de estabilidad.
    X_global, ids_global = preprocesar(matriz, columnas)
    indices_nucleo = [ids_global.index(hospital) for hospital in nucleo_ids]
    X_total = X_global[indices_nucleo]
    ids_total = nucleo_ids
    rng = np.random.default_rng(RANDOM_STATE)
    resumenes = []
    replicas_todas = []
    individuales_todas = []
    grupos_todos = []

    for k in K_EVALUADOS:
        labels_referencia = etiquetas_ward(X_total, k)
        tamanos = pd.Series(labels_referencia).value_counts().sort_values(ascending=False).tolist()
        sil_referencia = silhouette_score(X_total, labels_referencia)
        print(f"\nK_sub={k}: silhouette completo={sil_referencia:.4f}; tamaños={tamanos}")

        replicas, coasignacion, individuales, grupos, resumen = evaluar_k(
            matriz_nucleo, ids_total, columnas, labels_referencia, k, rng
        )
        resumen["silhouette_referencia"] = sil_referencia
        resumen["tamanos_referencia"] = str(tamanos)
        resumenes.append(resumen)
        replicas_todas.append(replicas)
        individuales_todas.append(individuales)
        grupos_todos.append(grupos)
        coasignacion.to_csv(TABLES_DIR / f"estabilidad_nivel2_coasignacion_k{k}.csv")

        print(
            "  ARI mediana={:.4f}; NMI mediana={:.4f}; Silhouette mediana={:.4f}".format(
                resumen["ari_mediana"], resumen["nmi_mediana"], resumen["silhouette_mediana"]
            )
        )

    pd.concat(replicas_todas, ignore_index=True).to_csv(
        TABLES_DIR / "estabilidad_nivel2_submuestreo.csv", index=False
    )
    pd.concat(individuales_todas, ignore_index=True).to_csv(
        TABLES_DIR / "estabilidad_nivel2_individual.csv", index=False
    )
    pd.concat(grupos_todos, ignore_index=True).to_csv(
        TABLES_DIR / "estabilidad_nivel2_por_subgrupo.csv", index=False
    )
    df_resumen = pd.DataFrame(resumenes)
    df_resumen.to_csv(TABLES_DIR / "estabilidad_nivel2_resumen.csv", index=False)

    print("\nResumen percentilar:")
    print(df_resumen.to_string(index=False, float_format="%.4f"))
    print("\nArchivos generados en reports/tables/ con prefijo estabilidad_nivel2_.")


if __name__ == "__main__":
    main()
