from pathlib import Path
from typing import Dict, List, Union
import json

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from config import SCENE_MAPPING_PATH


class SceneClassifier:
    """
    Scene classification using Places365.
    """

    def __init__(self, model_dir: str, device: str = None):

        self.device = (
            device
            if device
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_dir = Path(model_dir)

        weight_path = model_dir / "resnet50_places365.pth"
        category_path = model_dir / "categories_places365.txt"

        # Load category names
        self.classes = self._load_categories(category_path)

        # Load scene mapping
        with open(SCENE_MAPPING_PATH, "r") as f:
            self.scene_mapping = json.load(f)

        # Build model
        self.model = models.resnet50(
            weights=None,
            num_classes=365
        )

        checkpoint = torch.load(
            weight_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }

        else:

            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_categories(self, category_file: Path) -> List[str]:

        classes = []

        with open(category_file) as f:

            for line in f:

                label = line.strip().split(" ")[0]

                label = label[3:]

                label = label.replace("_", " ")

                classes.append(label)

        return classes

    def _normalize_scene(self, raw_scene: str) -> str:

        raw_scene = raw_scene.lower()

        for category, labels in self.scene_mapping.items():

            if raw_scene in [x.lower() for x in labels]:
                return category

        return "other"

    def _load_image(self, image: Union[str, Path, Image.Image]):

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        return Image.open(image).convert("RGB")

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        topk: int = 5
    ) -> Dict:

        image = self._load_image(image)

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)

        probs = F.softmax(logits, dim=1)

        scores, indices = torch.topk(probs, topk)

        scores = scores.squeeze().cpu().tolist()
        indices = indices.squeeze().cpu().tolist()

        top_predictions = []

        for idx, score in zip(indices, scores):

            top_predictions.append(
                (
                    self.classes[idx],
                    round(float(score), 4)
                )
            )

        raw_scene = top_predictions[0][0]

        normalized_scene = self._normalize_scene(raw_scene)

        return {

            "scene": normalized_scene,

            "raw_scene": raw_scene,

            "confidence": top_predictions[0][1],

            "top5": top_predictions

        }