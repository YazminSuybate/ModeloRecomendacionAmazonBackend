"""VD-D3: diversidad intra-lista (I12 ILD@K) por usuario (Ziegler et al., 2005).

ILD@K = (2 / (K(K − 1))) · Σ_{i<j} (1 − sim(i, j)), con sim = similitud coseno entre los
vectores de contenido de los ítems. Se usan los mismos vectores TF-IDF para todos los modelos,
de modo que el Pre-Test (popularidad) y el Post-Test (híbrido) se miden igual.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def intra_list_diversity(
    recommended: np.ndarray, item_vectors: sp.csr_matrix, k: int
) -> np.ndarray:
    """ILD@K de cada fila de `recommended` (n_usuarios, ≥k)."""
    if k < 2:
        raise ValueError("ILD@K requiere K >= 2")
    vectors = normalize(sp.csr_matrix(item_vectors), norm="l2", axis=1)
    n_pairs = k * (k - 1) / 2
    upper = np.triu_indices(k, 1)
    ild = np.empty(len(recommended))
    for row, items in enumerate(recommended):
        block = vectors[items[:k]]
        similarity = (block @ block.T).toarray()
        ild[row] = (1.0 - similarity[upper]).sum() / n_pairs
    return ild
