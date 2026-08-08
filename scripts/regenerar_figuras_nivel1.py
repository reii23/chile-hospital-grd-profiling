"""Regenera las figuras Ward finales del Nivel 1."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "data" / "processed" / "hospital_matrix_scaled.parquet"
OUTPUT_DIR = ROOT / "TT_Reinaldo_Pacheco" / "img"
ASSIGNMENT_PATH = ROOT / "reports" / "tables" / "asignacion_jerarquica_final.csv"
CLUSTER_COLORS = {
    "C0": "#1f77b4",
    "MAINSTREAM": "#2ca02c",
    "C2": "#9467bd",
    "C3": "#8c564b",
}
CLUSTER_DISPLAY_NAMES = {
    "C0": "C0: pediátricos",
    "MAINSTREAM": "C1: núcleo generalista",
    "C2": "C2: carga clínica",
    "C3": "C3: perfil monográfico",
}


def main() -> None:
    matrix = pd.read_parquet(MATRIX_PATH)
    hospital_ids = matrix.pop("COD_HOSPITAL").astype(str).tolist()
    X_scaled = matrix.to_numpy()
    linkage_matrix = linkage(X_scaled, method="ward")
    assignments = pd.read_csv(ASSIGNMENT_PATH, dtype={"COD_HOSPITAL": str})
    assignments = assignments.set_index("COD_HOSPITAL")["nivel1_etiqueta"]
    cluster_by_leaf = assignments.reindex(hospital_ids)
    if cluster_by_leaf.isna().any():
        missing = cluster_by_leaf[cluster_by_leaf.isna()].index.tolist()
        raise ValueError(f"Hospitales sin etiqueta Ward K=4: {missing}")
    expected_sizes = {"C0": 3, "MAINSTREAM": 54, "C2": 5, "C3": 3}
    observed_sizes = cluster_by_leaf.value_counts().to_dict()
    if observed_sizes != expected_sizes:
        raise ValueError(
            f"La partición del dendrograma no coincide con K=4: {observed_sizes}"
        )

    labels_by_k = {
        k: fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
        for k in (2, 3, 4)
    }
    silhouettes = {k: silhouette_score(X_scaled, labels) for k, labels in labels_by_k.items()}

    node_cluster_sets = {
        leaf: {cluster_name}
        for leaf, cluster_name in enumerate(cluster_by_leaf.tolist())
    }
    n_leaves = len(hospital_ids)
    for row_index, (left, right, _, _) in enumerate(linkage_matrix):
        node_cluster_sets[n_leaves + row_index] = (
            node_cluster_sets[int(left)] | node_cluster_sets[int(right)]
        )

    def color_branch(node_id: int) -> str:
        clusters = node_cluster_sets[node_id]
        if len(clusters) == 1:
            return CLUSTER_COLORS[next(iter(clusters))]
        return "#6c757d"

    def cut_height(k: int) -> float:
        """Altura estrictamente entre las fusiones que delimitan una solución K."""
        return float((linkage_matrix[-k, 2] + linkage_matrix[-(k - 1), 2]) / 2)

    cut_heights = {k: cut_height(k) for k in (2, 3, 4)}
    figure, axis = plt.subplots(figsize=(17, 7))
    dendrogram(
        linkage_matrix,
        labels=hospital_ids,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=0.0,
        link_color_func=color_branch,
        ax=axis,
    )
    axis.set(
        title="Dendrograma Ward del Nivel 1 (n=65)",
        xlabel="Código de hospital",
        ylabel="Distancia de fusión (Ward)",
    )
    cut_styles = {
        2: ("#e69f00", "Corte K=2"),
        3: ("#d55e00", "Corte K=3"),
        4: ("#009e73", "Corte K=4"),
    }
    for k, (color, label) in cut_styles.items():
        axis.axhline(
            cut_heights[k], color=color, linestyle="--", linewidth=1.4,
            alpha=0.85, label=label,
        )
    branch_handles = [
        Patch(color=CLUSTER_COLORS[cluster], label=CLUSTER_DISPLAY_NAMES[cluster])
        for cluster in ("C0", "MAINSTREAM", "C2", "C3")
    ]
    fusion_handle = Line2D([0], [0], color="#6c757d", label="Fusión entre clústeres")
    cut_handles = [
        Line2D([0], [0], color=color, linestyle="--", label=label)
        for color, label in cut_styles.values()
    ]
    axis.legend(
        handles=branch_handles + [fusion_handle] + cut_handles,
        loc="upper left", fontsize=8, ncol=2,
    )
    figure.tight_layout()
    dendrogram_path = OUTPUT_DIR / "dendrograma_ward_nivel1.png"
    figure.savefig(dendrogram_path, dpi=200, bbox_inches="tight")
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
    print(f"K=4 canónico: tamaños={observed_sizes}")
    for k in (2, 3, 4):
        print(f"Altura visual de corte K={k}: {cut_heights[k]:.6f}")
    for k in (2, 3, 4):
        print(f"K={k}: silhouette={silhouettes[k]:.6f}; tamaños={np.bincount(labels_by_k[k]).tolist()}")


if __name__ == "__main__":
    main()
