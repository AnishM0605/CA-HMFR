from pathlib import Path

import faiss
import numpy as np


class FAISSBuilder:
    """
    Builds and saves a FAISS index from image embeddings.
    """

    def __init__(self):

        self.index = None

    def build(self, embeddings: np.ndarray):

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return self.index

    def save(
        self,
        save_path
    ):

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(save_path)
        )

    @staticmethod
    def load(index_path):

        return faiss.read_index(
            str(index_path)
        )