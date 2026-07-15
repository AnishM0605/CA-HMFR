from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class CaptionGenerator:
    """
    Generates image captions using BLIP.
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        max_new_tokens: int = 30,
    ):

        self.device = (
            device
            if device
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = BlipProcessor.from_pretrained(model_path)

        self.model = BlipForConditionalGeneration.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens

    def _load_image(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Image.Image:

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        return Image.open(image).convert("RGB")

    @torch.no_grad()
    def generate_caption(
        self,
        image: Union[str, Path, Image.Image]
    ) -> str:

        image = self._load_image(image)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens
        )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption.strip()

    @torch.no_grad()
    def generate_captions(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[str]:

        captions = []

        for image in image_paths:
            captions.append(
                self.generate_caption(image)
            )

        return captions