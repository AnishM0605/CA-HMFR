from typing import List

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class QueryEncoder:
    """
    Wrapper around FashionCLIP for text embedding extraction.

    This class mirrors the implementation of VisionEncoder so that
    both image and text embeddings are generated using the same
    FashionCLIP model.
    """

    def __init__(self, model_path: str, device: str = None):

        self.device = (
            device
            if device is not None
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, query: str) -> torch.Tensor:
        """
        Encode a single text query.

        Parameters
        ----------
        query : str

        Returns
        -------
        torch.Tensor
            Shape: (512,)
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if len(query) == 0:
            raise ValueError("Query cannot be empty.")

        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)

        # Compatibility with different Transformers versions
        if isinstance(outputs, torch.Tensor):
            embedding = outputs
        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state[:, 0]
        else:
            raise RuntimeError(
                "Unsupported output type returned by get_text_features()."
            )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1
        )

        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def encode_texts(self, queries: List[str]) -> torch.Tensor:
        """
        Encode multiple text queries.

        Parameters
        ----------
        queries : List[str]

        Returns
        -------
        torch.Tensor
            Shape: (N,512)
        """

        if len(queries) == 0:
            raise ValueError("Query list cannot be empty.")

        inputs = self.processor(
            text=queries,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            embeddings = outputs
        elif hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise RuntimeError(
                "Unsupported output type returned by get_text_features()."
            )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1
        )

        return embeddings.cpu()