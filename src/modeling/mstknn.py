"""MST-kNN clustering algorithm.

Combina el Minimum Spanning Tree (MST) y el k-Nearest Neighbors graph (kNN).
Las aristas que sobreviven son las que pertenecen a AMBOS grafos. Los componentes
conectados del grafo resultante son los clusters.

Referencia metodológica:
- Inostroza-Ponta M., Mar-Molinero C. (2009). "MST-kNN: a tool for the
  construction of a network of similarity relationships."
- Carlier A., Lillo J., Inostroza-Ponta M., Villalobos-Cid M. (2020).
  "Evaluating the categorization of Chilean public hospitals by case-mix
  complexity: a genetic algorithm approach." [referencia 11 del informe]

El algoritmo NO requiere especificar K. Solo requiere:
- `k`: número de vecinos del kNN graph (típicamente 3-7 para n=50-100)
- `metric`: distancia (default: euclidiana)

Implementación:
1. Construir matriz de distancias completa entre los n puntos.
2. Construir el MST usando Kruskal (scipy).
3. Construir el grafo kNN simétrico (cada punto se conecta con sus k vecinos
   más cercanos; se simetriza considerando aristas mutuas).
4. Intersección: aristas presentes en MST AND kNN.
5. Identificar componentes conectados → clusters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


@dataclass
class MSTkNN:
    """Clustering MST-kNN.

    Parameters
    ----------
    k : int
        Número de vecinos en el grafo kNN. Para n=50-100 usar k=3..7.
    metric : str, default='euclidean'
        Métrica de distancia.
    knn_simetrico : bool, default=True
        Si True, una arista (i, j) está en el kNN solo si j es vecino de i Y
        i es vecino de j (kNN mutuo). Más estricto pero más estable.
        Si False, arista existe si j es vecino de i O viceversa.
    """

    k: int = 4
    metric: str = "euclidean"
    knn_simetrico: bool = True

    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Aplica MST-kNN y retorna (labels, info_diagnóstica).

        Returns
        -------
        labels : np.ndarray
            Cluster de cada punto (0..K-1).
        info : dict
            Diagnóstico del algoritmo: K resultante, n_aristas en cada grafo,
            n_aristas en intersección, distribución de tamaños de cluster.
        """
        n = X.shape[0]
        if self.k >= n:
            raise ValueError(f"k={self.k} debe ser < n={n}")

        # 1. Matriz de distancias completa
        dist_matrix = squareform(pdist(X, metric=self.metric))

        # 2. MST usando matriz de distancias
        mst_sparse = minimum_spanning_tree(csr_matrix(dist_matrix))
        # Aristas del MST: (i, j) donde MST[i, j] > 0
        mst_dense = mst_sparse.toarray()
        # Hacer simétrico (MST es undirected)
        mst_aristas = set()
        for i in range(n):
            for j in range(n):
                if mst_dense[i, j] > 0:
                    mst_aristas.add((min(i, j), max(i, j)))

        # 3. kNN graph
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        # indices[i, 0] siempre es i mismo, descartar
        knn_aristas = set()
        if self.knn_simetrico:
            # Solo aristas mutuas: i ∈ kNN(j) AND j ∈ kNN(i)
            vecinos = {i: set(indices[i, 1:].tolist()) for i in range(n)}
            for i in range(n):
                for j in vecinos[i]:
                    if i in vecinos[j]:
                        knn_aristas.add((min(i, j), max(i, j)))
        else:
            for i in range(n):
                for j in indices[i, 1:]:
                    knn_aristas.add((min(i, int(j)), max(i, int(j))))

        # 4. Intersección MST ∩ kNN
        aristas_finales = mst_aristas & knn_aristas

        # 5. Construir grafo final como matriz sparse y encontrar componentes
        rows = [i for i, j in aristas_finales]
        cols = [j for i, j in aristas_finales]
        data = [1] * len(aristas_finales)
        if len(aristas_finales) > 0:
            grafo_final = csr_matrix(
                (data + data, (rows + cols, cols + rows)), shape=(n, n)
            )
        else:
            grafo_final = csr_matrix((n, n))

        n_componentes, labels = connected_components(
            grafo_final, directed=False, return_labels=True,
        )

        info = {
            "k": self.k,
            "K_descubierto": int(n_componentes),
            "n_aristas_MST": len(mst_aristas),
            "n_aristas_kNN": len(knn_aristas),
            "n_aristas_interseccion": len(aristas_finales),
            "ratio_aristas": len(aristas_finales) / len(mst_aristas) if mst_aristas else 0.0,
            "tamanos_clusters": pd.Series(labels).value_counts().sort_index().tolist(),
        }
        return labels, info
