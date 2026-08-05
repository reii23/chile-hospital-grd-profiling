"""Reductor dimensional (PCA + LASSO condicional) (Req 8)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


N_MIN_COMPONENTES = 5
N_MAX_COMPONENTES = 8
VAR_THRESHOLD = 0.80
RANDOM_STATE = 42


@dataclass
class Reductor_Dimensional:
    matriz_casuistica: pd.DataFrame
    target_minsal: pd.Series | None = None
    n_min: int = N_MIN_COMPONENTES
    n_max: int = N_MAX_COMPONENTES
    var_threshold: float = VAR_THRESHOLD
    random_state: int = RANDOM_STATE

    _id_col: str = field(default="COD_HOSPITAL", init=False)

    def _matriz_X(self):
        df = self.matriz_casuistica.copy()
        if self._id_col in df.columns:
            idx = pd.Index(df[self._id_col].values)
            num_df = df.drop(columns=[self._id_col]).select_dtypes(include="number")
        else:
            idx = df.index
            num_df = df.select_dtypes(include="number")
        return idx, num_df.values, num_df.columns

    def fit_pca(self) -> dict:
        idx, X, cols = self._matriz_X()
        if X.shape[0] < 2 or X.shape[1] < 2:
            raise ValueError(f"Matriz demasiado pequeña: {X.shape}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_max_efectivo = min(self.n_max, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_max_efectivo, random_state=self.random_state)
        scores = pca.fit_transform(X_scaled)

        var_acum = np.cumsum(pca.explained_variance_ratio_)
        idx_k = int(np.argmax(var_acum >= self.var_threshold))
        if var_acum[idx_k] < self.var_threshold:
            K = n_max_efectivo
        else:
            K = max(self.n_min, idx_k + 1)
            K = min(K, n_max_efectivo)

        scores_K = scores[:, :K]
        loadings_K = pca.components_[:K, :]
        evr_K = pca.explained_variance_ratio_[:K]

        componentes = pd.DataFrame(
            scores_K, index=idx,
            columns=[f"dim_{i+1:02d}" for i in range(K)],
        )
        componentes.index.name = self._id_col
        componentes = componentes.sort_index().reset_index()

        loadings = pd.DataFrame(
            loadings_K,
            index=[f"dim_{i+1:02d}" for i in range(K)],
            columns=cols,
        )

        return {
            "componentes": componentes,
            "loadings": loadings,
            "explained_variance_ratio": evr_K,
            "K": K,
            "scaler": scaler,
            "pca": pca,
        }

    def fit_lasso(self) -> dict | None:
        if self.target_minsal is None:
            return None

        idx, X, cols = self._matriz_X()
        y = self.target_minsal.reindex(idx).dropna()
        if len(y) < 10 or y.nunique() < 2:
            return None

        X_alineado = pd.DataFrame(X, index=idx, columns=cols).loc[y.index].values

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        seleccion_count = np.zeros(X_alineado.shape[1], dtype=int)

        for train_idx, _ in kfold.split(X_alineado, y):
            # El escalador se ajusta solo en entrenamiento para evitar que la
            # distribución de los hospitales del pliegue retenido influya en
            # la selección de variables.
            scaler_fold = StandardScaler()
            X_train = scaler_fold.fit_transform(X_alineado[train_idx])
            clf = LogisticRegression(
                penalty="l1", solver="saga", C=0.5,
                max_iter=2000, random_state=self.random_state,
            )
            clf.fit(X_train, y.iloc[train_idx])
            no_cero = (np.abs(clf.coef_) > 1e-8).any(axis=0)
            seleccion_count += no_cero.astype(int)

        frecuencia = pd.Series(
            seleccion_count / 5, index=cols,
            name="frecuencia_seleccion",
        ).sort_values(ascending=False)

        candidatas = frecuencia[frecuencia >= 0.6].index.tolist()
        K = max(self.n_min, min(self.n_max, len(candidatas)))
        seleccionadas = frecuencia.head(K).index.tolist()
        inestable = (frecuencia.head(K).min() < 0.6) if K > 0 else True

        # El modelo final se ajusta sobre todos los hospitales y conserva su
        # propio escalador para interpretar y reproducir sus coeficientes.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_alineado)
        clf_final = LogisticRegression(
            penalty="l1", solver="saga", C=0.5,
            max_iter=2000, random_state=self.random_state,
        )
        clf_final.fit(X_scaled, y)
        coef_df = pd.DataFrame(
            clf_final.coef_, columns=cols,
            index=[f"clase_{i}" for i in range(clf_final.coef_.shape[0])],
        )

        return {
            "variables_seleccionadas": seleccionadas,
            "coef": coef_df,
            "frecuencia_seleccion": frecuencia,
            "inestable": bool(inestable),
            "K": K,
            "scaler": scaler,
        }

    def fit(self) -> dict:
        return {"pca": self.fit_pca(), "lasso": self.fit_lasso()}
