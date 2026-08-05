"""Pruebas de significancia por variable y por NIVEL de la jerarquia.

Logica solicitada:
  - Variable NUMERICA  -> Kruskal-Wallis entre grupos.
  - Variable de FRECUENCIA / categorica -> Chi-cuadrado de independencia.
  - Regla de lectura: p_ajustado > 0.05  => la variable NO discrimina entre grupos
                      (homogenea; no aporta al perfil de ese nivel).
                      p_ajustado <= 0.05 => la variable SI distingue los grupos.

Corrección por comparaciones múltiples:
  - Benjamini-Hochberg (FDR) sobre los p-valores de Kruskal-Wallis.
  - Más potente que Bonferroni cuando las variables tienen correlaciones entre sí.

Tamaño de efecto:
  - Épsilon cuadrado (ε²) para Kruskal-Wallis (versión insesgada):
        ε² = (H - (k - 1)) / (n - k)
    Umbrales: ~0.01 pequeño, ~0.06 moderado, ~0.14 grande.
  - V de Cramér para chi-cuadrado (variables categóricas).

Se evalua en DOS niveles de la particion jerarquica (Aglomerativo Ward):

  NIVEL 1: entre los 4 clusters del corte K=4
           (C0 pediatricos, C2 institutos, C3 alta-CMA, MAINSTREAM).

  NIVEL 2: SOLO dentro del mainstream, entre sus subclusters (S0..S8)
           obtenidos al subdividir el nucleo.

Fuente de la partición: `reports/tables/asignacion_jerarquica_final.csv`.
Variables en escala original: `data/processed/hospital_matrix.parquet`.

Salidas:
- `reports/tables/pruebas_kruskal_nivel1.csv`
- `reports/tables/pruebas_kruskal_nivel2_mainstream.csv`
- `reports/tables/pruebas_chi2_categoricas.csv`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
ALPHA = 0.05

# ---------------------------------------------------------------------------
# 1. Cargar variables (escala original) + asignacion jerarquica
# ---------------------------------------------------------------------------
matriz = pd.read_parquet(PROCESSED / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)

asig = pd.read_csv(TABLES / "asignacion_jerarquica_final.csv", dtype=str)
asig["COD_HOSPITAL"] = asig["COD_HOSPITAL"].astype(str)

df = matriz.merge(asig, on="COD_HOSPITAL", how="inner")

# Variables numericas interpretables (excluye id, aux y dimensiones PCA)
EXCLUIR = {"COD_HOSPITAL", "peso_medio_cma_imputado"}
num_cols = [
    c for c in matriz.columns
    if c not in EXCLUIR and not c.startswith("dim_")
    and pd.api.types.is_numeric_dtype(matriz[c])
]


def epsilon_cuadrado(H: float, k: int, n: int) -> float:
    """Épsilon cuadrado (versión insesgada del tamaño de efecto para Kruskal-Wallis).

    ε² = (H - (k - 1)) / (n - k)
    Umbrales de referencia: ~0.01 pequeño, ~0.06 moderado, ~0.14 grande.
    """
    return max(0.0, (H - (k - 1)) / (n - k)) if (n - k) > 0 else np.nan


def kruskal_por_nivel(data: pd.DataFrame, col_grupo: str) -> pd.DataFrame:
    """Kruskal-Wallis de cada variable numerica entre los grupos de col_grupo.

    Aplica corrección de Benjamini-Hochberg sobre los p-valores y calcula
    épsilon cuadrado como tamaño de efecto. Agrega la mediana por grupo
    para leer el sentido de la diferencia.
    """
    grupos_unicos = sorted(data[col_grupo].unique())
    n_total = len(data)
    k_grupos = len(grupos_unicos)
    filas = []
    for col in num_cols:
        muestras = [
            data.loc[data[col_grupo] == gnom, col].dropna().values
            for gnom in grupos_unicos
        ]
        muestras = [m for m in muestras if len(m) > 0]
        try:
            H, p = kruskal(*muestras)
        except ValueError:
            H, p = np.nan, np.nan
        eps2 = epsilon_cuadrado(H, k_grupos, n_total) if pd.notna(H) else np.nan
        fila = {
            "variable": col,
            "test": "Kruskal-Wallis",
            "estadistico_H": H,
            "p_valor": p,
            "epsilon_cuadrado": eps2,
        }
        # medianas por grupo
        for gnom in grupos_unicos:
            fila[f"med_{gnom}"] = data.loc[data[col_grupo] == gnom, col].median()
        filas.append(fila)

    result = pd.DataFrame(filas)

    # ── Corrección Benjamini-Hochberg sobre p-valores válidos ──────────────
    mask_valid = result["p_valor"].notna()
    p_vals = result.loc[mask_valid, "p_valor"].values
    if len(p_vals) > 0:
        reject, p_adj, _, _ = multipletests(p_vals, alpha=ALPHA, method="fdr_bh")
        result.loc[mask_valid, "p_valor_ajustado_BH"] = p_adj
        result.loc[mask_valid, "discrimina_BH"] = np.where(reject, "si", "no")
    else:
        result["p_valor_ajustado_BH"] = np.nan
        result["discrimina_BH"] = "no"

    # Ordenar por tamaño de efecto (ε²) descendente para jerarquizar variables
    result = result.sort_values("epsilon_cuadrado", ascending=False).reset_index(drop=True)
    return result


def resumen(df_res: pd.DataFrame, titulo: str) -> None:
    n_sig = (df_res.get("discrimina_BH", pd.Series(dtype=str)) == "si").sum()
    print("=" * 90)
    print(titulo)
    print("=" * 90)
    cols_show = [
        "variable", "estadistico_H", "p_valor", "p_valor_ajustado_BH",
        "discrimina_BH", "epsilon_cuadrado",
    ]
    cols_show = [c for c in cols_show if c in df_res.columns]
    print(df_res[cols_show].to_string(index=False, float_format="%.4g"))
    print(f"\n  -> SI discriminan (BH ajustado p<=0.05): {n_sig}/{len(df_res)}")
    print(f"  -> NO discriminan : {len(df_res) - n_sig}/{len(df_res)}\n")


# ---------------------------------------------------------------------------
# 2. NIVEL 1 — entre los 4 clusters del corte K=4
# ---------------------------------------------------------------------------
df_n1 = df.copy()
df_n1["grupo_n1"] = df_n1["nivel1_etiqueta"]
print("NIVEL 1 — tamanos:", df_n1["grupo_n1"].value_counts().sort_index().to_dict(), "\n")
kw1 = kruskal_por_nivel(df_n1, "grupo_n1")
kw1.to_csv(TABLES / "pruebas_kruskal_nivel1.csv", index=False)
resumen(kw1, "NIVEL 1 · KRUSKAL-WALLIS + BH + ε² entre C0 / C2 / C3 / MAINSTREAM (K=4)")

# ---------------------------------------------------------------------------
# 3. NIVEL 2 — solo dentro del mainstream, entre subclusters S*
# ---------------------------------------------------------------------------
df_n2 = df[df["nivel1_etiqueta"] == "MAINSTREAM"].copy()
df_n2["grupo_n2"] = df_n2["nivel2_etiqueta"]
print("NIVEL 2 (mainstream) — tamanos:",
      df_n2["grupo_n2"].value_counts().sort_index().to_dict(), "\n")
kw2 = kruskal_por_nivel(df_n2, "grupo_n2")
kw2.to_csv(TABLES / "pruebas_kruskal_nivel2_mainstream.csv", index=False)
resumen(kw2, "NIVEL 2 · KRUSKAL-WALLIS + BH + ε² entre subclusters del MAINSTREAM (S0..S8)")

# ---------------------------------------------------------------------------
# 4. Chi-cuadrado (Servicio de Salud) en cada nivel + V de Cramér
# ---------------------------------------------------------------------------
g = pd.read_parquet(
    PROCESSED / "grd_filtrado.parquet", columns=["COD_HOSPITAL", "SERVICIO_SALUD"]
)
g["COD_HOSPITAL"] = g["COD_HOSPITAL"].astype(str)
serv = g.drop_duplicates("COD_HOSPITAL").set_index("COD_HOSPITAL")["SERVICIO_SALUD"]
df["SERVICIO_SALUD"] = df["COD_HOSPITAL"].map(serv)

filas_chi = []
for nivel, sub, col_grupo in [
    ("nivel1", df, "nivel1_etiqueta"),
    ("nivel2_mainstream", df[df["nivel1_etiqueta"] == "MAINSTREAM"], "nivel2_etiqueta"),
]:
    tabla = pd.crosstab(sub[col_grupo], sub["SERVICIO_SALUD"])
    chi2, p_chi, dof, _ = chi2_contingency(tabla)
    n = tabla.values.sum()
    r, c = tabla.shape

    # V de Cramér SIN corregir (sesgado al alza en tablas dispersas: muchas
    # categorías de Servicio de Salud frente a un n pequeño; ver Bergsma 2013).
    v_cramer_sin_corregir = float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else np.nan

    # V de Cramér CORREGIDO por sesgo (Bergsma & Wicher, 2013):
    # resta el sesgo esperado de phi^2, r y c bajo independencia antes de
    # calcular V. Evita que tablas dispersas (n pequeño, muchas categorías)
    # produzcan valores de V artificialmente altos.
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    v_cramer_corregido = float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan

    filas_chi.append({
        "nivel": nivel,
        "variable": "SERVICIO_SALUD",
        "test": "Chi-cuadrado",
        "estadistico_chi2": chi2,
        "gl": dof,
        "p_valor": p_chi,
        "V_cramer_sin_corregir": v_cramer_sin_corregir,
        "V_cramer_bergsma": v_cramer_corregido,
        "discrimina (p<=0.05)": ("si" if p_chi <= ALPHA else "no"),
        "nota": "tabla dispersa (n pequeno, muchas categorias de Servicio de "
                "Salud): V sin corregir esta inflado; se reporta V corregido "
                "por sesgo (Bergsma 2013) como referencia principal",
    })

df_chi = pd.DataFrame(filas_chi)
df_chi.to_csv(TABLES / "pruebas_chi2_categoricas.csv", index=False)
print("=" * 90)
print("CHI-CUADRADO · SERVICIO_SALUD por nivel + V de Cramér")
print("=" * 90)
print(df_chi[["nivel", "estadistico_chi2", "gl", "p_valor",
              "V_cramer_sin_corregir", "V_cramer_bergsma", "discrimina (p<=0.05)"]]
      .to_string(index=False, float_format="%.4g"))
print("\n>>> Tablas guardadas en reports/tables/")
