"""VI-D3: tasa de hibridación del ranking (I6).

Compara el Top-K del sistema híbrido con el Top-K del módulo ALS aislado: la proporción de ítems
del ranking final que ALS por sí solo no habría recomendado es lo que aporta la hibridación con el
módulo de contenido. 0 = el híbrido equivale a ALS; 1 = ningún ítem coincide.
"""

from __future__ import annotations

import numpy as np


def hybridization_rate(hybrid_top_k: np.ndarray, als_top_k: np.ndarray, k: int) -> np.ndarray:
    """|Top-K híbrido \\ Top-K ALS| / K para cada usuario (filas alineadas)."""
    if hybrid_top_k.shape[0] != als_top_k.shape[0]:
        raise ValueError("Las listas del híbrido y de ALS deben tener los mismos usuarios")
    return np.array(
        [
            len(set(hybrid[:k].tolist()) - set(als[:k].tolist())) / k
            for hybrid, als in zip(hybrid_top_k, als_top_k, strict=True)
        ]
    )
