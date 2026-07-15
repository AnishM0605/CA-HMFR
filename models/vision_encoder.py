from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class VisionEncoder:
    """
    Wrapper around FashionCLIP for image embedding extraction.
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

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load an image from path or PIL Image.
        """

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        return Image.open(image).convert("RGB")

    @torch.no_grad()
    def encode_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Encode a single image.

        Returns
        -------
        torch.Tensor
            Shape: (512,)
        """

        image = self._load_image(image)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.get_image_features(**inputs)

        embedding = outputs.pooler_output

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1
        )

        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        Encode multiple images.

        Parameters
        ----------
        image_paths : list

        Returns
        -------
        torch.Tensor
            Shape: (N,512)
        """

        images = [
            self._load_image(img)
            for img in image_paths
        ]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model.get_image_features(**inputs)

        embeddings = outputs.pooler_output

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1
        )

        return embeddings.cpu()