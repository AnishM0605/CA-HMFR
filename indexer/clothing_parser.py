from typing import Dict, List

import spacy


class ClothingParser:
    """
    Parses BLIP-generated captions into structured fashion metadata.

    Example Output:
    {
        "caption": "...",
        "people": ["woman"],
        "style": ["formal"],
        "garments": [
            {
                "type": "shirt",
                "color": "white",
                "material": None,
                "pattern": None,
                "style": "formal"
            }
        ]
    }
    """

    GARMENTS = {
        "shirt", "t-shirt", "top", "hoodie", "jacket", "coat",
        "blazer", "dress", "skirt", "pants", "trousers",
        "jeans", "shorts", "tie", "suit", "sweater",
        "cardigan", "raincoat", "boots", "heels",
        "sneakers", "scarf", "hat", "cap", "bag",
        "handbag", "tights", "leggings", "vest",
        "blouse", "shirt", "shirt", "shirt"
    }

    COLORS = {
        "red", "blue", "green", "yellow", "black", "white",
        "brown", "orange", "pink", "purple", "gray", "grey",
        "beige", "gold", "silver", "navy", "maroon",
        "olive", "cream", "cyan", "teal", "khaki", "tan"
    }

    COLOR_MODIFIERS = {
        "light",
        "dark",
        "bright",
        "deep",
        "pale",
        "off"
    }

    MATERIALS = {
        "denim",
        "leather",
        "cotton",
        "silk",
        "linen",
        "wool",
        "velvet",
        "lace",
        "satin",
        "suede",
        "polyester",
        "knit"
    }

    PATTERNS = {
        "striped",
        "checked",
        "checkered",
        "plaid",
        "floral",
        "printed",
        "polka",
        "solid"
    }

    STYLES = {
        "formal",
        "casual",
        "business",
        "sport",
        "sportswear",
        "streetwear",
        "party",
        "winter",
        "summer",
        "traditional",
        "outerwear",
        "oversized",
        "vintage",
        "elegant"
    }

    PEOPLE = {
        "man",
        "woman",
        "boy",
        "girl",
        "person",
        "model",
        "people"
    }

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def _extract_color(self, doc, token):

        start = max(0, token.i - 3)
        end = min(len(doc), token.i + 3)

        words = [t.text for t in doc[start:end]]

        for i, word in enumerate(words):

            if word in self.COLORS:

                if i > 0 and words[i - 1] in self.COLOR_MODIFIERS:
                    return f"{words[i-1]} {word}"

                return word

        return None

    def _extract_material(self, doc, token):

        start = max(0, token.i - 3)
        end = min(len(doc), token.i + 3)

        for neighbor in doc[start:end]:

            if neighbor.text in self.MATERIALS:
                return neighbor.text

        return None

    def _extract_pattern(self, doc, token):

        start = max(0, token.i - 3)
        end = min(len(doc), token.i + 3)

        for neighbor in doc[start:end]:

            if neighbor.text in self.PATTERNS:
                return neighbor.text

        return None

    def _extract_style(self, doc, token):

        start = max(0, token.i - 4)
        end = min(len(doc), token.i + 4)

        for neighbor in doc[start:end]:

            if neighbor.text in self.STYLES:
                return neighbor.text

        return None

    def parse(self, caption: str) -> Dict:

        doc = self.nlp(caption.lower())

        garments = []
        people = []
        styles = []

        for token in doc:

            word = token.text

            if word in self.PEOPLE:
                people.append(word)

            if word in self.STYLES:
                styles.append(word)

            if word in self.GARMENTS:

                garments.append(
                    {
                        "type": word,
                        "color": self._extract_color(doc, token),
                        "material": self._extract_material(doc, token),
                        "pattern": self._extract_pattern(doc, token),
                        "style": self._extract_style(doc, token),
                    }
                )

        return {
            "caption": caption,
            "people": sorted(set(people)),
            "style": sorted(set(styles)),
            "garments": garments,
        }