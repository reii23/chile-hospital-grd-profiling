"""Preprocesador y clusterizador K-means++ con bootstrap (Req 10)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler


SKEW_THRESHOLD = 1.5
LOG1P_DEFAULT: tuple[str, ...] = ("egresos_por_anio", "peso_medio_grd", "peso_medio_cma")
SILHOUETTE_UMBRAL = 0.40
RANDOM_STATE = 42
N_INIT_KMEANS = 50
BOOTSTRAP_N = 100
BOOTSTRAP_SUBSAMPLE = 0.80


@dataclass
class Preprocesador:
    log1p_cols: list[str] = field(default_factory=lambda: list(LOG1P_DEFAULT))
    skew_threshold: float = SKEW_THRESHOLD

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "COD_HOSPITAL" in df.columns:
            df = df.set_index("COD_HOSPITAL")

        num = df.select_dtypes(include="number").copy()
        cols_log = set(c for c in self.log1p_cols if c in num.columns)
        for col in num.columns:
            if col in cols_log:
                continue
            try:
                skew = float(stats.skew(num[col].dropna(), nan_policy="omit"))
            except Exception:
                skew = 0.0
            if abs(skew) > self.skew_threshold:
                cols_log.add(col)

        for col in cols_log:
            valores = num[col]
            if (valores < -1).any():
                shift = abs(valores.min()) + 1
                num[col] = np.log1p(valores + shift)
            else:
                num[col] = np.log1p(valores)

        scaler = RobustScaler()
        scaled = scaler.fit_transform(num)
        result = pd.DataFrame(scaled, index=num.index, columns=num.columns)
        return result


@dataclass
class Clusterizador:
    X_scaled: pd.DataFrame
    k_grid: tuple[int, ...] = (2, 3, 4, 5, 6)
    n_init: int = N_INIT_KMEANS
    random_state: int = RANDOM_STATE

    def evaluar_k(self) -> pd.DataFrame:
        X = self.X_scaled.values
        rows = []
        for k in self.k_grid:
            if k >= len(X):
                continue
            km = KMeans(
                n_clusters=k, init="k-means++",
                n_init=self.n_init, random_state=self.random_state,
            )
            labels = km.fit_predict(X)
            try:
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
            except ValueError:
                sil, ch, db = -1.0, 0.0, np.inf
            rows.append({
                "K": k,
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(db),
            })
        return pd.DataFrame(rows)

    def seleccionar_k(self, df_metricas: pd.DataFrame) -> int:
        if df_metricas.empty:
            raise ValueError("df_metricas vacío")
        validos = df_metricas[df_metricas["silhouette"] >= SILHOUETTE_UMBRAL]
        if not validos.empty:
            best = validos["silhouette"].max()
            cand = validos[validos["silhouette"] == best]
            return int(cand["K"].min())
        best_ch = df_metricas["calinski_harabasz"].max()
        cand = df_metricas[df_metricas["calinski_harabasz"] == best_ch]
        return int(cand["K"].min())

    def ajustar(self, k: int) -> pd.DataFrame:
        km = KMeans(
            n_clusters=k, init="k-means++",
            n_init=self.n_init, random_state=self.random_state,
        )
        labels = km.fit_predict(self.X_scaled.values)
        result = pd.DataFrame({
            "COD_HOSPITAL": self.X_scaled.index.values,
            "cluster": labels.astype("int8"),
        })
        return result.sort_values("COD_HOSPITAL").reset_index(drop=True)

    def bootstrap(self, k: int, n_iter: int = BOOTSTRAP_N,
                  subsample: float = BOOTSTRAP_SUBSAMPLE) -> dict:
        X = self.X_scaled.values
        n = X.shape[0]
        n_sub = max(2, int(n * subsample))
        rng = np.random.default_rng(self.random_state)
        silhouettes = []
        for _ in range(n_iter):
            idx = rng.choice(n, size=n_sub, replace=False)
            X_sub = X[idx]
            km = KMeans(
                n_clusters=k, init="k-means++",
                n_init=self.n_init, random_state=self.random_state,
            )
            labels = km.fit_predict(X_sub)
            try:
                sil = silhouette_score(X_sub, labels)
            except ValueError:
                sil = -1.0
            silhouettes.append(sil)
        silhouettes = np.array(silhouettes)
        return {
            "silhouette_media": float(silhouettes.mean()),
            "silhouette_std": float(silhouettes.std(ddof=1)),
            "silhouette_ic_inf": float(np.percentile(silhouettes, 2.5)),
            "silhouette_ic_sup": float(np.percentile(silhouettes, 97.5)),
            "silhouettes": silhouettes,
        }
