"""Regenera las dos figuras finales del subclustering exploratorio de Nivel 2."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
SEARCH_PATH = TABLES_DIR / "busqueda_subclustering.csv"
ASSIGNMENT_PATH = TABLES_DIR / "asignacion_jerarquica_final.csv"
OUTPUT_DIR = ROOT / "TT_Reinaldo_Pacheco" / "img"
REPORTED_K_SUB = 9


def main() -> None:
    search = pd.read_csv(SEARCH_PATH).sort_values("K_sub")
    assignment = pd.read_csv(ASSIGNMENT_PATH)
    n_mainstream = int((assignment["nivel1_etiqueta"] == "MAINSTREAM").sum())

    figure, axis_silhouette = plt.subplots(figsize=(7.2, 4.8))
    axis_db = axis_silhouette.twinx()
    axis_silhouette.plot(search["K_sub"], search["silhouette_sub"], "o-", color="tab:blue", label="Silhouette", linewidth=2)
    axis_db.plot(search["K_sub"], search["davies_sub"], "s--", color="tab:orange", label="Davies-Bouldin", linewidth=2)
    selected = search.loc[search["K_sub"] == REPORTED_K_SUB].iloc[0]
    axis_silhouette.scatter(REPORTED_K_SUB, selected["silhouette_sub"], s=90, color="tab:blue", zorder=4)
    axis_db.scatter(REPORTED_K_SUB, selected["davies_sub"], s=90, color="tab:orange", zorder=4)
    axis_silhouette.axhline(0.20, color="gray", linestyle="--", alpha=0.55, label="Referencia 0,20")
    axis_silhouette.set(xlabel=r"$K_{sub}$", ylabel="Silhouette", title=f"Métricas internas del Nivel 2 (n={n_mainstream})")
    axis_db.set_ylabel("Davies-Bouldin", color="tab:orange")
    axis_silhouette.set_xticks(search["K_sub"])
    axis_silhouette.grid(True, alpha=0.25)
    handles_1, labels_1 = axis_silhouette.get_legend_handles_labels()
    handles_2, labels_2 = axis_db.get_legend_handles_labels()
    axis_silhouette.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "metricas_nivel2.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(search["K_sub"], search["sub_n_clusters_n4plus"], "o-", linewidth=2, color="tab:green", label=r"Clústeres $n\geq4$")
    axis.plot(search["K_sub"], search["sub_n_singletons"], "s--", linewidth=2, color="tab:red", label="Grupos unitarios")
    axis.plot(search["K_sub"], search["sub_n_pares"], "^--", linewidth=2, color="tab:purple", label="Pares")
    axis.axvline(REPORTED_K_SUB, color="black", linestyle=":", alpha=0.7, label=rf"Solución reportada ($K_{{sub}}={REPORTED_K_SUB}$)")
    axis.set(xlabel=r"$K_{sub}$", ylabel="Cantidad de grupos", title=f"Balance de tamaños del Nivel 2 (n={n_mainstream})")
    axis.set_xticks(search["K_sub"])
    axis.set_ylim(bottom=-0.25)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "seleccion_nivel2.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
