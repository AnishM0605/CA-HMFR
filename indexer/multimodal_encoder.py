from pathlib import Path
from typing import List

import numpy as np
from tqdm.auto import tqdm

from models.vision_encoder import VisionEncoder


class MultimodalEncoder:
    """
    Generates FashionCLIP embeddings for a collection of images.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
    ):
        self.encoder = VisionEncoder(model_path)
        self.batch_size = batch_size

    def build(self, image_paths: List[str]) -> np.ndarray:

        all_embeddings = []

        for i in tqdm(range(0, len(image_paths), self.batch_size)):

            batch = image_paths[i:i + self.batch_size]

            embeddings = self.encoder.encode_images(batch)

            all_embeddings.append(embeddings.numpy())

        return np.vstack(all_embeddings)

    @staticmethod
    def save(embeddings: np.ndarray, save_path: str):

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(save_path, embeddings)