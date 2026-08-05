"""Regresion logistica LASSO contra la clasificacion MINSAL (Observacion 16
de correcciones.txt).

Problema: la metodologia (Seccion sec:lasso del Capitulo 3) describe un
procedimiento completo de regresion logistica con penalizacion L1, validacion
cruzada estratificada de 5 pliegues y frecuencia de seleccion por variable,
pero nunca se ejecuto ni se reporto en resultados. El codigo ya existia
(Reductor_Dimensional.fit_lasso en src/modeling/reducer.py) pero jamas se
invocaba desde ningun script.

Proposito del analisis (no es predictivo): identificar que variables de
actividad clinica GRD permiten reconstruir la clasificacion administrativa
de Nivel de Complejidad MINSAL (Alta/Mediana), y cuales variables funcionales
NO se relacionan con ella. Esto complementa cuantitativamente el contraste
ARI/NMI de la Seccion sec:res-minsal (que solo dice CUANTO concuerdan ambas
clasificaciones) explicando POR QUE: que dimensiones de la actividad clinica
"delatan" la complejidad administrativa y cuales son ciegas para ella.

Se usan las 27 variables directas (interpretables) de la matriz institucional,
excluyendo las 8 componentes PCA de casuistica (dim_01..dim_08), que no tienen
lectura directa por si mismas y dificultarian la interpretacion de los
coeficientes.

Salidas (reports/tables/):
  - lasso_coeficientes.csv           (coeficientes del modelo final, todas las variables)
  - lasso_frecuencia_seleccion.csv   (frecuencia de seleccion por variable, 5 pliegues)
  - lasso_desempeno_predictivo.csv   (accuracy/AUC por pliegue, fuera de muestra)
  - lasso_variables_seleccionadas.csv (variables estables, frecuencia >= 0.6)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.reducer import Reductor_Dimensional
from src.utils.io import PROCESSED_DIR, TABLES_DIR

RANDOM_STATE = 42
BASE_EST = ROOT / "info-hospitales" / "Base de Establecimientos 2023.xlsx"
COLS_EXCLUIR = {"COD_HOSPITAL", "peso_medio_cma", "peso_medio_cma_imputado"}
COLS_PCA = [f"dim_{i:02d}" for i in range(1, 9)]

print("=" * 80)
print("REGRESION LOGISTICA LASSO: variables clinicas vs. clasificacion MINSAL")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Cargar matriz institucional (27 variables directas, sin componentes PCA)
# ---------------------------------------------------------------------------
print("\n[1/5] Cargando matriz institucional (27 variables directas)...")
matriz = pd.read_parquet(PROCESSED_DIR / "hospital_matrix.parquet")
matriz["COD_HOSPITAL"] = matriz["COD_HOSPITAL"].astype(str)

cols_directas = [
    c for c in matriz.columns
    if c not in COLS_EXCLUIR and c not in COLS_PCA and c != "COD_HOSPITAL"
]
print(f"  Variables directas usadas como predictoras: {len(cols_directas)}")
print(f"  {cols_directas}")

matriz_directa = matriz[["COD_HOSPITAL"] + cols_directas].copy()

# ---------------------------------------------------------------------------
# 2. Cargar etiqueta MINSAL (misma fuente que v2_comparacion_minsal.py)
# ---------------------------------------------------------------------------
print("\n[2/5] Cargando clasificacion MINSAL (Nivel de Complejidad)...")
base = pd.read_excel(BASE_EST, skiprows=1)
base.columns = [str(c).strip() for c in base.columns]
base["COD_HOSPITAL"] = (
    base["Código Vigente"].astype(str).str.replace(".0", "", regex=False).str.strip()
)
minsal_raw = (
    base.dropna(subset=["Nivel de Complejidad"])
    .drop_duplicates("COD_HOSPITAL")
    .set_index("COD_HOSPITAL")["Nivel de Complejidad"]
)

target = matriz_directa["COD_HOSPITAL"].map(minsal_raw)
n_sin_etiqueta = target.isna().sum()
if n_sin_etiqueta:
    print(f"  [aviso] {n_sin_etiqueta} hospitales sin etiqueta MINSAL, se excluyen")
print(f"  Distribucion de la etiqueta MINSAL: {target.value_counts().to_dict()}")

# Codificar como binaria (Alta=1, Mediana=0) para LogisticRegression estandar
target_bin = target.map({"Alta Complejidad": 1, "Mediana Complejidad": 0})
mask_valido = target_bin.notna()
print(f"  Hospitales usados en el modelo (con etiqueta binaria valida): "
      f"{mask_valido.sum()} de {len(matriz_directa)}")

matriz_lasso = matriz_directa.loc[mask_valido].reset_index(drop=True)
y = target_bin.loc[mask_valido].astype(int).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Ejecutar fit_lasso() del Reductor_Dimensional (codigo ya existente)
# ---------------------------------------------------------------------------
print("\n[3/5] Ejecutando Reductor_Dimensional.fit_lasso() (5-fold CV, C=0.5)...")
y_indexado = pd.Series(y.values, index=matriz_lasso["COD_HOSPITAL"].values)
reductor = Reductor_Dimensional(matriz_casuistica=matriz_lasso, target_minsal=y_indexado)
resultado_lasso = reductor.fit_lasso()

if resultado_lasso is None:
    raise RuntimeError("fit_lasso() devolvio None: revisar tamanio de muestra o clases.")

print(f"  Variables con coeficiente no nulo en el modelo final: "
      f"{(np.abs(resultado_lasso['coef'].values) > 1e-8).sum()} de {len(cols_directas)}")
print(f"  K variables seleccionadas (frecuencia >= 0.6, tope {reductor.n_max}): "
      f"{resultado_lasso['K']}")
print(f"  Seleccion marcada como inestable (alguna var. bajo 0.6 en el top-K): "
      f"{resultado_lasso['inestable']}")

# ---------------------------------------------------------------------------
# 4. Persistir coeficientes, frecuencia de seleccion y variables estables
# ---------------------------------------------------------------------------
print("\n[4/5] Persistiendo coeficientes y frecuencia de seleccion...")
coef_df = resultado_lasso["coef"].T.reset_index()
coef_df.columns = ["variable", "coeficiente"]
coef_df = coef_df.sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)
coef_df.to_csv(TABLES_DIR / "lasso_coeficientes.csv", index=False)
print("  Top 10 variables por magnitud de coeficiente (modelo final, todos los datos):")
print(coef_df.head(10).to_string(index=False, float_format="%.4f"))

frecuencia_df = resultado_lasso["frecuencia_seleccion"].reset_index()
frecuencia_df.columns = ["variable", "frecuencia_seleccion"]
frecuencia_df.to_csv(TABLES_DIR / "lasso_frecuencia_seleccion.csv", index=False)
print("\n  Frecuencia de seleccion (proporcion de los 5 pliegues con coef. != 0):")
print(frecuencia_df.to_string(index=False, float_format="%.2f"))

vars_seleccionadas_df = pd.DataFrame({
    "variable": resultado_lasso["variables_seleccionadas"],
})
vars_seleccionadas_df = vars_seleccionadas_df.merge(frecuencia_df, on="variable", how="left")
vars_seleccionadas_df.to_csv(TABLES_DIR / "lasso_variables_seleccionadas.csv", index=False)
print(f"\n  Variables seleccionadas (K={resultado_lasso['K']}): "
      f"{resultado_lasso['variables_seleccionadas']}")

# ---------------------------------------------------------------------------
# 5. Desempeno predictivo fuera de muestra (accuracy y AUC por pliegue)
# ---------------------------------------------------------------------------
print("\n[5/5] Evaluando desempeno predictivo fuera de muestra (5-fold CV)...")
X = matriz_lasso[cols_directas].values

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
filas_desempeno = []
for fold_id, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    # Ajustar el escalado solo en el conjunto de entrenamiento evita que la
    # distribución del pliegue de prueba intervenga en la evaluación.
    scaler_fold = StandardScaler()
    X_train = scaler_fold.fit_transform(X[train_idx])
    X_test = scaler_fold.transform(X[test_idx])

    clf = LogisticRegression(
        penalty="l1", solver="saga", C=0.5, max_iter=2000, random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y.iloc[train_idx])
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y.iloc[test_idx], y_pred)
    try:
        auc = roc_auc_score(y.iloc[test_idx], y_proba)
    except ValueError:
        auc = np.nan

    filas_desempeno.append({
        "fold": fold_id, "n_test": len(test_idx),
        "n_positivos_test": int(y.iloc[test_idx].sum()),
        "accuracy": acc, "auc": auc,
    })

df_desempeno = pd.DataFrame(filas_desempeno)
df_desempeno.to_csv(TABLES_DIR / "lasso_desempeno_predictivo.csv", index=False)
print(df_desempeno.to_string(index=False, float_format="%.3f"))
print(f"\n  Accuracy promedio (5-fold CV): {df_desempeno['accuracy'].mean():.3f} "
      f"(desv. {df_desempeno['accuracy'].std():.3f})")
print(f"  AUC promedio (5-fold CV): {df_desempeno['auc'].mean():.3f} "
      f"(desv. {df_desempeno['auc'].std():.3f})")

# Baseline trivial: predecir siempre la clase mayoritaria
prop_mayoritaria = max(y.mean(), 1 - y.mean())
print(f"  Baseline (predecir siempre la clase mayoritaria): accuracy = {prop_mayoritaria:.3f}")

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"  Variable objetivo: Nivel de Complejidad MINSAL (Alta=1 vs. Mediana=0)")
print(f"  n = {mask_valido.sum()} hospitales, {len(cols_directas)} variables predictoras")
print(f"  Accuracy CV: {df_desempeno['accuracy'].mean():.3f} vs. baseline {prop_mayoritaria:.3f}")
print(f"  AUC CV: {df_desempeno['auc'].mean():.3f}")
print(f"  Variables mas estables (frecuencia de seleccion = 1.0): "
      f"{frecuencia_df.loc[frecuencia_df['frecuencia_seleccion'] == 1.0, 'variable'].tolist()}")
print("\nAnalisis completo.")
