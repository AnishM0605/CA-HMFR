import json
from pathlib import Path
from typing import Dict, List


class MetadataBuilder:
    """
    Builds metadata records for the retrieval system.

    This class DOES NOT run ML models.
    It only combines outputs from previously executed modules.
    """

    SEMANTIC_TAGS = {
        "fashion week": "fashion_week",
        "runway": "runway",
        "catwalk": "runway",
        "fashion show": "runway",
        "formal": "formal",
        "casual": "casual",
        "business": "business",
        "wedding": "wedding",
        "party": "party",
        "sport": "sports",
        "sports": "sports",
        "office": "office",
        "park": "park",
        "street": "street",
        "shopping": "shopping",
        "mall": "shopping"
    }

    @staticmethod
    def _generate_tags(
        caption: str,
        clothing: Dict,
        scene: Dict
    ) -> List[str]:

        tags = set()

        # --------------------------------------------------
        # Caption-based semantic tags
        # --------------------------------------------------

        caption = caption.lower()

        for phrase, tag in MetadataBuilder.SEMANTIC_TAGS.items():

            if phrase in caption:
                tags.add(tag)

        # --------------------------------------------------
        # People
        # --------------------------------------------------

        for person in clothing.get("people", []):

            tags.add(person.lower())

        # --------------------------------------------------
        # Scene
        # --------------------------------------------------

        if scene.get("normalized"):

            tags.add(scene["normalized"].lower())

        # --------------------------------------------------
        # Garments
        # --------------------------------------------------

        for garment in clothing.get("garments", []):

            if garment.get("type"):
                tags.add(garment["type"].lower())

            if garment.get("color"):
                tags.add(garment["color"].lower())

            if garment.get("material"):
                tags.add(garment["material"].lower())

            if garment.get("pattern"):
                tags.add(garment["pattern"].lower())

            if garment.get("style"):
                tags.add(garment["style"].lower())

        return sorted(tags)

    @staticmethod
    def build(
        image_id: int,
        image_path: str,
        caption: str,
        clothing: Dict,
        scene: Dict
    ) -> Dict:

        image_path = Path(image_path)

        metadata = {

            "image_id": image_id,

            "image_name": image_path.name,

            "image_path": str(image_path),

            "embedding_index": image_id,

            "caption": caption,

            "scene": {

                "normalized": scene["normalized"],

                "raw": scene["raw"],

                "source": scene["source"],

                "confidence": scene["confidence"]

            },

            "people": clothing["people"],

            "garments": clothing["garments"],

            "tags": MetadataBuilder._generate_tags(
                caption,
                clothing,
                scene
            )

        }

        return metadata

    @staticmethod
    def save(
        metadata,
        save_path
    ):

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(save_path, "w") as f:

            json.dump(
                metadata,
                f,
                indent=4
            )