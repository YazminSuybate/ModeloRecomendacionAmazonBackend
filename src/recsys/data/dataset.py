"""Contenedor de interacciones que consumen todos los modelos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.schema import ITEM_IDX, USER_IDX


@dataclass(frozen=True)
class InteractionData:
    """Interacciones ya codificadas, con las dimensiones del catálogo completo."""

    frame: pd.DataFrame
    n_users: int
    n_items: int

    def with_frame(self, frame: pd.DataFrame) -> InteractionData:
        """Mismo catálogo y usuarios, otras interacciones (p. ej., una sub-partición)."""
        return InteractionData(frame.reset_index(drop=True), self.n_users, self.n_items)

    def to_csr(self, values: np.ndarray | None = None) -> sp.csr_matrix:
        """Matriz usuarios × productos; por defecto con 1 en cada interacción."""
        data = np.ones(len(self.frame), dtype=np.float32) if values is None else values
        return sp.csr_matrix(
            (data, (self.frame[USER_IDX].to_numpy(), self.frame[ITEM_IDX].to_numpy())),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )
