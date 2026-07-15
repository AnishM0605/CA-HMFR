import re


class SceneResolver:
    """
    Resolves the final scene using both
    BLIP caption and Places365 prediction.
    """

    BLIP_SCENE_RULES = {

        "runway": [
            "runway",
            "fashion week",
            "fashion show",
            "catwalk"
        ],

        "office": [
            "office",
            "conference room",
            "meeting room",
            "workspace"
        ],

        "park": [
            "park",
            "garden",
            "forest",
            "playground"
        ],

        "street": [
            "street",
            "road",
            "crosswalk",
            "sidewalk",
            "downtown"
        ],

        "shopping": [
            "mall",
            "shopping",
            "boutique",
            "store",
            "shop"
        ],

        "restaurant": [
            "restaurant",
            "cafe",
            "coffee shop",
            "dining"
        ],

        "home": [
            "living room",
            "bedroom",
            "kitchen",
            "home"
        ],

        "sports": [
            "gym",
            "stadium",
            "locker room",
            "court"
        ]
    }

    def resolve(self, caption: str, scene_prediction: dict):

        caption = caption.lower()

        # -----------------------------
        # First preference:
        # BLIP caption
        # -----------------------------

        for scene, keywords in self.BLIP_SCENE_RULES.items():

            for keyword in keywords:

                if keyword in caption:

                    return {
                        "normalized": scene,
                        "raw": keyword,
                        "source": "blip",
                        "confidence": 1.0
                    }

        # -----------------------------
        # Otherwise
        # use Places365
        # -----------------------------

        return {
            "normalized": scene_prediction["scene"],
            "raw": scene_prediction["raw_scene"],
            "source": "places365",
            "confidence": scene_prediction["confidence"]
        }