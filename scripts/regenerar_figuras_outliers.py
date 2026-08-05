"""Regenera la única figura de atípicos incluida en el informe: la proyección PCA."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import PROCESSED_DIR, TABLES_DIR
from src.modeling.referencia import COLS_DROP_CLUSTERING, LOG1P_COLS

RANDOM_STATE = 42
OUTPUT_PATH = ROOT / "formato-tesis" / "Tesis_Reinaldo_Pacheco_Editable" / "img" / "atipicos_pca.png"
CLUSTER_COLORS = {0: "#d1495b", 1: "#4c6fd0", 2: "#e58b3c", 3: "#55a868"}
CLUSTER_LABELS = {0: "C0 (n=3)", 1: "C1 (n=54)", 2: "C2 (n=5)", 3: "C3 (n=3)"}
SHORT_NAMES = {"106102": "Eduardo Pereira", "112104": "Neurocirugía", "111195": "HUAP", "112103": "INER", "113130": "Exequiel González", "112102": "Luis Calvo Mackenna", "112100": "Del Salvador", "111100": "San Borja-Arriarán", "107100": "Gustavo Fricke"}


def main() -> None:
    matrix = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
    matrix["COD_HOSPITAL"] = matrix["COD_HOSPITAL"].astype(str)
    ids = matrix["COD_HOSPITAL"].tolist()
    features = matrix.drop(columns=[column for column in COLS_DROP_CLUSTERING if column in matrix.columns]).copy()
    for column in LOG1P_COLS:
        if column in features:
            features[column] = np.log1p(features[column].clip(lower=0))
    X = RobustScaler().fit_transform(features)

    assignment = pd.read_csv(TABLES_DIR / "asignacion_jerarquica_final.csv", dtype={"COD_HOSPITAL": str})
    labels = assignment.set_index("COD_HOSPITAL")["nivel1_K4"].astype(int).reindex(ids).to_numpy()
    if pd.isna(labels).any():
        raise ValueError("La asignación Ward final no cubre todos los hospitales de la matriz.")
    outliers = pd.read_csv(TABLES_DIR / "atipicos_corregidos.csv", dtype={"COD_HOSPITAL": str})
    outlier_ids = set(outliers.loc[outliers["outlier_union"], "COD_HOSPITAL"])
    if len(outlier_ids) != 9:
        raise ValueError(f"Se esperaban 9 hospitales marcados; se encontraron {len(outlier_ids)}.")

    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    figure, axis = plt.subplots(figsize=(11, 7.4))
    for cluster, label in CLUSTER_LABELS.items():
        mask = labels == cluster
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=66, color=CLUSTER_COLORS[cluster], edgecolors="#303030", linewidth=0.45, alpha=0.86, label=label)
    outlier_indices = [index for index, hospital_id in enumerate(ids) if hospital_id in outlier_ids]
    axis.scatter(coordinates[outlier_indices, 0], coordinates[outlier_indices, 1], s=175, facecolors="none", edgecolors="#e53935", linewidth=1.8, label="Atípico por $\\geq 1$ criterio (n=9)", zorder=4)
    for index in outlier_indices:
        axis.annotate(SHORT_NAMES.get(ids[index], ids[index]), (coordinates[index, 0], coordinates[index, 1]), xytext=(4, 4), textcoords="offset points", fontsize=7.4, zorder=5)
    axis.set(title="PCA de hospitales y atípicos identificados — Ward K=4", xlabel="PC1", ylabel="PC2")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8, framealpha=0.95)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Guardada: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
