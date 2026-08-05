"""Regenera las figuras Ward finales del Nivel 1."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "data" / "processed" / "hospital_matrix_scaled.parquet"
OUTPUT_DIR = ROOT / "formato-tesis" / "Tesis_Reinaldo_Pacheco_Editable" / "img"


def main() -> None:
    matrix = pd.read_parquet(MATRIX_PATH)
    hospital_ids = matrix.pop("COD_HOSPITAL").astype(str).tolist()
    X_scaled = matrix.to_numpy()
    linkage_matrix = linkage(X_scaled, method="ward")
    labels_by_k = {k: fcluster(linkage_matrix, t=k, criterion="maxclust") - 1 for k in (2, 3, 4)}
    silhouettes = {k: silhouette_score(X_scaled, labels) for k, labels in labels_by_k.items()}

    figure, axis = plt.subplots(figsize=(16, 6))
    dendrogram(linkage_matrix, labels=hospital_ids, leaf_rotation=90, leaf_font_size=6,
               color_threshold=linkage_matrix[-4, 2], above_threshold_color="gray", ax=axis)
    axis.set(title="Dendrograma de agrupamiento jerárquico global (Ward, n=65)",
             xlabel="Código de hospital", ylabel="Distancia de fusión (Ward)")
    for index, color, label in ((-2, "tab:orange", "Corte K=2"), (-3, "tab:red", "Corte K=3"), (-4, "tab:green", "Corte K=4")):
        axis.axhline(linkage_matrix[index, 2], color=color, linestyle="--", alpha=0.7, label=label)
    axis.legend(loc="upper left")
    figure.tight_layout()
    dendrogram_path = OUTPUT_DIR / "dendrograma_ward_nivel1.png"
    figure.savefig(dendrogram_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X_scaled)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for axis, k in zip(axes, (2, 3, 4)):
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels_by_k[k], cmap="tab10",
                     s=70, edgecolors="black", linewidth=0.6)
        axis.set_title(f"K={k} | Silhouette={silhouettes[k]:.3f}")
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    figure.suptitle("Proyección PCA de la matriz institucional (Ward, n=65)", y=1.02, fontsize=12)
    figure.tight_layout()
    pca_path = OUTPUT_DIR / "pca_particiones_nivel1.png"
    figure.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Dendrograma: {dendrogram_path}")
    print(f"PCA: {pca_path}")
    for k in (2, 3, 4):
        print(f"K={k}: silhouette={silhouettes[k]:.6f}; tamaños={np.bincount(labels_by_k[k]).tolist()}")


if __name__ == "__main__":
    main()
