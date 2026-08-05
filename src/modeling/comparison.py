"""Comparador entre clusterings (Req 11)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


@dataclass
class Comparador_Clusterings:
    clusters_anterior: pd.DataFrame
    clusters_nuevo: pd.DataFrame
    nombres_hospitales: pd.DataFrame | None = None

    def alinear(self) -> pd.DataFrame:
        merged = self.clusters_anterior.merge(
            self.clusters_nuevo, on="COD_HOSPITAL",
            suffixes=("_anterior", "_nuevo"), how="inner",
        )
        return merged.sort_values("COD_HOSPITAL").reset_index(drop=True)

    def adjusted_rand(self) -> float:
        m = self.alinear()
        if m.empty:
            return 0.0
        return float(adjusted_rand_score(m["cluster_anterior"], m["cluster_nuevo"]))

    def jaccard_matrix(self) -> pd.DataFrame:
        m = self.alinear()
        if m.empty:
            return pd.DataFrame(columns=["cluster_nuevo", "cluster_anterior", "jaccard"])
        clusters_n = sorted(m["cluster_nuevo"].unique())
        clusters_a = sorted(m["cluster_anterior"].unique())
        rows = []
        for cn in clusters_n:
            for ca in clusters_a:
                hosp_n = set(m[m["cluster_nuevo"] == cn]["COD_HOSPITAL"])
                hosp_a = set(m[m["cluster_anterior"] == ca]["COD_HOSPITAL"])
                inter = len(hosp_n & hosp_a)
                union = len(hosp_n | hosp_a)
                rows.append({
                    "cluster_nuevo": int(cn),
                    "cluster_anterior": int(ca),
                    "jaccard": inter / union if union else 0.0,
                    "n_interseccion": inter,
                    "n_union": union,
                })
        return pd.DataFrame(rows)

    def _hungarian_match(self) -> dict[int, int]:
        jacc = self.jaccard_matrix()
        if jacc.empty:
            return {}
        wide = jacc.pivot(
            index="cluster_nuevo", columns="cluster_anterior", values="jaccard"
        ).fillna(0.0)
        costos = -wide.values
        if costos.shape[0] > costos.shape[1]:
            pad = np.zeros((costos.shape[0], costos.shape[0] - costos.shape[1]))
            costos_pad = np.hstack([costos, pad])
            row_ind, col_ind = linear_sum_assignment(costos_pad)
            mapping = {
                int(wide.index[r]): int(wide.columns[c]) if c < len(wide.columns) else -1
                for r, c in zip(row_ind, col_ind)
            }
        else:
            row_ind, col_ind = linear_sum_assignment(costos)
            mapping = {int(wide.index[r]): int(wide.columns[c]) for r, c in zip(row_ind, col_ind)}
        return mapping

    def hospitales_que_cambiaron(self) -> pd.DataFrame:
        m = self.alinear()
        if m.empty:
            return m
        mapping = self._hungarian_match()
        m["cluster_nuevo_alineado"] = m["cluster_nuevo"].map(mapping)
        cambios = m[m["cluster_nuevo_alineado"] != m["cluster_anterior"]].copy()
        if self.nombres_hospitales is not None and not cambios.empty:
            cambios = cambios.merge(self.nombres_hospitales, on="COD_HOSPITAL", how="left")
        return cambios.reset_index(drop=True)

    def plot_pca2d(self, X_scaled: pd.DataFrame, path, title="Comparación clusterings") -> Path:
        m = self.alinear()
        if m.empty:
            return Path(path)
        X = X_scaled.loc[m["COD_HOSPITAL"]].values
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 7))
        markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "P", 5: "X"}
        for ca in sorted(m["cluster_anterior"].unique()):
            mask = m["cluster_anterior"].values == ca
            sc = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=m.loc[mask, "cluster_nuevo"], cmap="tab10",
                marker=markers.get(int(ca), "o"),
                s=80, edgecolors="black", linewidth=0.8,
                label=f"cluster_anterior={int(ca)}",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
