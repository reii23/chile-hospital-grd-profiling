# Caracterización funcional de hospitales públicos chilenos mediante aprendizaje no supervisado de datos GRD

**Trabajo de título — Ingeniería de Ejecución en Computación e Informática, Universidad de Santiago de Chile**
**Autor:** Reinaldo Alexis Pacheco Parra · **Profesor guía:** Manuel Villalobos-Cid

## Propósito

Repositorio reproducible del perfilamiento funcional de hospitales públicos chilenos a partir de egresos codificados por Grupos Relacionados por el Diagnóstico (GRD), correspondientes al período 2019--2024. El análisis caracteriza perfiles institucionales desde su actividad, complejidad, diversidad de casuística, modalidad de atención y componentes reducidos de mezcla clínica; no pretende reemplazar la clasificación administrativa de complejidad.

La matriz final considera **65 hospitales y 35 variables de clustering**: 27 indicadores clínico-operativos directos y 8 componentes principales de casuística. `peso_medio_cma` se mantiene para análisis descriptivo, pero se excluye del clustering y de LASSO porque no es comparable estructuralmente en todos los establecimientos.

## Resultados principales

- Método principal: agrupamiento jerárquico aglomerativo Ward, con escalamiento robusto.
- Solución reportada de Nivel 1: **K=4**, con tamaños **[3, 54, 5, 3]**.
- Métricas de la solución K=4: silhouette **0,4008**, Calinski--Harabasz **22,446** y Davies--Bouldin **0,990**.
- K=2 obtiene un silhouette superior (0,6045), por lo que K=4 se interpreta como un compromiso de granularidad e interpretación, no como un óptimo estadístico absoluto.
- Nivel 2: subdivisión exploratoria del clúster generalista con `K_sub=10`.

Las cifras canónicas del barrido de K están en `reports/tables/clustering_ward_metricas.csv`, y la partición reportada en `reports/tables/asignacion_jerarquica_final.csv`. El resto de los resultados tabulares vive en `reports/tables/`.

## Estructura

```text
.
├── data/
│   ├── raw/                    # Fuentes GRD anuales originales
│   └── processed/              # Parquet intermedios y matrices locales
├── src/
│   ├── etl/                    # Lectura GRD, mapeadores CIE y construcción de variables
│   ├── modeling/               # Matriz, reducción y métodos de clustering
│   └── utils/                  # I/O y validación
├── scripts/
│   ├── build_extended_grd.py               # data/raw/*.txt -> grd_filtrado.parquet
│   ├── build_dataset.py                    # Matriz institucional y métricas Ward K=2..10
│   ├── build_hierarchical_clustering.py    # Partición reportada (Nivel 1 y Nivel 2)
│   ├── generate_descriptive_statistics.py  # Estadística descriptiva de la matriz
│   ├── analyze_group_differences.py        # Kruskal--Wallis y chi-cuadrado
│   ├── compare_minsal.py                   # Comparación descriptiva con MINSAL
│   ├── compare_internal_homogeneity.py     # Homogeneidad interna Ward vs MINSAL
│   ├── analyze_minsal_lasso.py             # Selección de variables con LASSO
│   ├── analyze_algorithmic_sensitivity.py  # K-means, GMM y MST-kNN
│   ├── analyze_mstknn_sensitivity.py       # Barrido MST-kNN mutuo, k=3..8
│   ├── analyze_compositional_sensitivity.py
│   ├── analyze_pca_component_sensitivity.py
│   ├── analyze_resampling_stability.py
│   ├── analyze_outlier_sensitivity.py
│   ├── analyze_temporal_sensitivity.py
│   ├── analyze_eligibility_sensitivity.py
│   ├── analyze_eligibility_threshold.py
│   ├── analyze_clustering_circularity.py   # Prueba de permutación de circularidad
│   └── regenerar_figuras_{nivel1,nivel2,outliers}.py  # Figuras incluidas en la tesis
├── notebooks/                  # Registro exploratorio (ETL, EDA, clustering, XAI)
├── reports/
│   ├── tables/                 # Resultados tabulares canónicos
│   └── figures/                # Figuras exploratorias (las de la tesis van en formato-tesis/)
├── formato-tesis/              # Fuente LaTeX y figuras de la tesis
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Datos y reproducibilidad

Todos los insumos que el pipeline necesita están versionados en este
repositorio, con dos excepciones que exceden el límite de tamaño de GitHub.

Insumos incluidos:

- `insumos/maestras/CIE-10.xlsx`, `insumos/maestras/CIE-9 .xlsx` y
  `insumos/maestras/Tablas maestras bases GRD.xlsx`: clasificaciones clínicas y
  maestra de establecimientos de las bases GRD.
- `info-hospitales/Base de Establecimientos 2023.xlsx`: clasificación MINSAL de
  Nivel de Complejidad, usada en las comparaciones externas.
- `data/processed/*.parquet`: matriz institucional (65 × 38), matriz escalada,
  vectores de casuística y variables intermedias.

No incluidos por tamaño:

| Artefacto | Tamaño | Cómo obtenerlo |
|---|---|---|
| `data/raw/GRD_PUBLICO_*.txt` | 3,8 GB (hasta 1 GB por archivo) | Descargar del Tablero GRD del MINSAL y colocar en `data/raw/` |
| `data/processed/grd_filtrado.parquet` | 198 MB | Generar con `scripts/build_extended_grd.py` a partir de `data/raw/` |

**15 de los 21 scripts se ejecutan directamente sobre un clon del
repositorio**, sin descargar nada. Los 6 que requieren el paso anterior son
`build_extended_grd.py`, `build_dataset.py`, `analyze_group_differences.py`,
`analyze_temporal_sensitivity.py`, `analyze_eligibility_sensitivity.py` y
`analyze_eligibility_threshold.py`, porque recorren los egresos individuales.

Las 80 tablas de `reports/tables/` se reproducen de forma determinista: las
semillas están fijadas en el código (`RANDOM_STATE = 42`) y se verificó que
ejecuciones repetidas generan salidas idénticas byte a byte.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Ejecución

El orden importa: los análisis leen la partición reportada
(`reports/tables/asignacion_jerarquica_final.csv`), que produce
`build_hierarchical_clustering.py`.

```bash
# 1. Construcción de datos y de la partición reportada
.venv/bin/python scripts/build_extended_grd.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_hierarchical_clustering.py

# 2. Descripción y contrastes entre grupos
.venv/bin/python scripts/generate_descriptive_statistics.py
.venv/bin/python scripts/analyze_group_differences.py
.venv/bin/python scripts/analyze_clustering_circularity.py

# 3. Comparación con la clasificación MINSAL
.venv/bin/python scripts/compare_minsal.py
.venv/bin/python scripts/compare_internal_homogeneity.py
.venv/bin/python scripts/analyze_minsal_lasso.py

# 4. Análisis de sensibilidad
.venv/bin/python scripts/analyze_algorithmic_sensitivity.py
.venv/bin/python scripts/analyze_mstknn_sensitivity.py
.venv/bin/python scripts/analyze_compositional_sensitivity.py
.venv/bin/python scripts/analyze_pca_component_sensitivity.py
.venv/bin/python scripts/analyze_resampling_stability.py
.venv/bin/python scripts/analyze_outlier_sensitivity.py
.venv/bin/python scripts/analyze_temporal_sensitivity.py
.venv/bin/python scripts/analyze_eligibility_sensitivity.py
.venv/bin/python scripts/analyze_eligibility_threshold.py

# 5. Figuras incluidas en la tesis
.venv/bin/python scripts/regenerar_figuras_nivel1.py
.venv/bin/python scripts/regenerar_figuras_nivel2.py
.venv/bin/python scripts/regenerar_figuras_outliers.py
```

Los análisis de sensibilidad describen estabilidad o concordancia interna de las particiones; no constituyen validación externa ni permiten establecer causalidad.

## Tecnologías

Python · pandas · NumPy · SciPy · scikit-learn · matplotlib · pyarrow · pytest · Hypothesis

## Licencia

El código de este repositorio (`src/`, `scripts/`, `notebooks/`, `tests/`) se
publica bajo licencia MIT. Ver [`LICENSE`](LICENSE).

La licencia no cubre los datos ni las fuentes de referencia, que se rigen por las
condiciones de sus titulares originales: los egresos GRD 2019--2024 del MINSAL,
las clasificaciones CIE-10 y CIE-9-MC, y la Base de Establecimientos del DEIS.
Ninguno de esos insumos se distribuye aquí.

El documento de la tesis se publica por separado bajo
[Creative Commons Atribución-Chile 3.0](https://creativecommons.org/licenses/by/3.0/cl/).
