import json
from pathlib import Path
from typing import Dict, List, Union

import faiss
import numpy as np
import torch


class HybridSearcher:
    """
    Performs FAISS-based vector retrieval using FashionCLIP embeddings.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        metadata_path: Union[str, Path]
    ):

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        top_k: int = 50
    ) -> List[Dict]:

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        query_embedding = query_embedding.astype(np.float32)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            item = dict(self.metadata[idx])

            item["vector_score"] = float(score)

            results.append(item)

        return results