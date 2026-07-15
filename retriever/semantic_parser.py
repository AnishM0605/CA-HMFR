"""
retriever/semantic_parser.py

Semantic parser for CA-HMFR.

Converts a natural language fashion query into structured semantic
attributes that will later be used by the Hybrid Reranker.
"""

from typing import Dict, List
import re


class SemanticParser:

    # ---------------------------------------------------------
    # Fashion Vocabulary
    # ---------------------------------------------------------

    COLORS = {
        "black", "white", "blue", "red", "green",
        "yellow", "pink", "purple", "orange",
        "brown", "grey", "gray", "beige",
        "navy", "maroon", "gold", "silver",
        "cream", "olive", "khaki"
    }

    GARMENTS = {
        "shirt",
        "tshirt",
        "t-shirt",
        "tee",
        "jacket",
        "coat",
        "hoodie",
        "sweater",
        "dress",
        "jeans",
        "pants",
        "trousers",
        "shorts",
        "skirt",
        "blazer",
        "suit",
        "kurta",
        "top",
        "saree",
        "sari",
        "shoe",
        "boot",
        "sneaker",
        "cap",
        "hat",
        "scarf"
    }

    PEOPLE = {
        "man",
        "woman",
        "boy",
        "girl",
        "male",
        "female",
        "person",
        "kid",
        "child"
    }

    STYLES = {
        "casual",
        "formal",
        "sport",
        "sports",
        "athletic",
        "denim",
        "vintage",
        "oversized",
        "slim",
        "loose",
        "ethnic",
        "traditional",
        "party",
        "winter",
        "summer"
    }

    SCENES = {
        "street",
        "runway",
        "beach",
        "office",
        "park",
        "indoor",
        "outdoor",
        "shopping",
        "mall",
        "home",
        "studio",
        "sports"
    }

    # ---------------------------------------------------------
    # Token Normalization
    # ---------------------------------------------------------

    NORMALIZATION = {

        # People
        "women": "woman",
        "men": "man",
        "girls": "girl",
        "boys": "boy",
        "people": "person",
        "kids": "kid",
        "children": "child",

        # Garments
        "shirts": "shirt",
        "jackets": "jacket",
        "hoodies": "hoodie",
        "coats": "coat",
        "dresses": "dress",
        "skirts": "skirt",
        "boots": "boot",
        "shoes": "shoe",
        "caps": "cap",
        "hats": "hat",
        "scarves": "scarf",

        # Alternate spellings
        "tshirts": "tshirt",
        "t-shirts": "t-shirt"
    }

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    def _tokenize(self, query: str) -> List[str]:
        """
        Tokenize and normalize a query.
        """

        query = query.lower().strip()

        tokens = re.findall(r"[a-zA-Z\-]+", query)

        normalized_tokens = [
            self.NORMALIZATION.get(token, token)
            for token in tokens
        ]

        return normalized_tokens

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def parse(self, query: str) -> Dict:

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        tokens = self._tokenize(query)

        colors = sorted(
            {
                token
                for token in tokens
                if token in self.COLORS
            }
        )

        garments = sorted(
            {
                token
                for token in tokens
                if token in self.GARMENTS
            }
        )

        people = sorted(
            {
                token
                for token in tokens
                if token in self.PEOPLE
            }
        )

        styles = sorted(
            {
                token
                for token in tokens
                if token in self.STYLES
            }
        )

        scenes = sorted(
            {
                token
                for token in tokens
                if token in self.SCENES
            }
        )

        semantic_tokens = sorted(
            set(
                colors +
                garments +
                people +
                styles +
                scenes
            )
        )

        return {

            "people": people,

            "garments": garments,

            "colors": colors,

            "styles": styles,

            "scene": scenes[0] if scenes else None,

            "tokens": semantic_tokens
        }