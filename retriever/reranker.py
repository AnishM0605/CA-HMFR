"""
retriever/reranker.py

Hybrid reranker for CA-HMFR.

This module combines dense vector similarity from the vision encoder
(FashionCLIP/OpenCLIP) with structured semantic reasoning extracted
from the user query.

Signals used:
1. Vector similarity
2. Garment match
3. Style match
4. People match
5. Scene match
6. Tag overlap

The final score is dynamically normalized based on which signals
are actually present in the parsed query.
"""

from typing import Dict, List


class HybridReranker:

    def __init__(
        self,
        vector_weight: float = 0.45,
        garment_weight: float = 0.20,
        style_weight: float = 0.15,
        people_weight: float = 0.10,
        scene_weight: float = 0.05,
        tag_weight: float = 0.05,
    ):
        self.vector_weight = vector_weight
        self.garment_weight = garment_weight
        self.style_weight = style_weight
        self.people_weight = people_weight
        self.scene_weight = scene_weight
        self.tag_weight = tag_weight

    # ==========================================================
    # Matching Functions
    # ==========================================================

    def _garment_match(self, query: Dict, candidate: Dict) -> float:
        query_garments = {g.lower() for g in query.get("garments", [])}
        if not query_garments:
            return 0.0

        candidate_garments = {
            garment["type"].lower()
            for garment in candidate.get("garments", [])
            if garment.get("type") is not None
        }

        matches = len(query_garments & candidate_garments)
        return matches / len(query_garments)

    def _style_match(self, query: Dict, candidate: Dict) -> float:
        query_styles = {style.lower() for style in query.get("styles", [])}
        if not query_styles:
            return 0.0

        caption = candidate.get("caption", "").lower()
        tags = {tag.lower() for tag in candidate.get("tags", [])}

        matches = 0
        for style in query_styles:
            if style in caption or style in tags:
                matches += 1

        return matches / len(query_styles)

    def _people_match(self, query: Dict, candidate: Dict) -> float:
        query_people = {p.lower() for p in query.get("people", [])}
        if not query_people:
            return 0.0

        candidate_people = {p.lower() for p in candidate.get("people", [])}
        matches = len(query_people & candidate_people)
        return matches / len(query_people)

    def _scene_match(self, query: Dict, candidate: Dict) -> float:
        query_scene = query.get("scene")
        if query_scene is None:
            return 0.0

        candidate_scene = (
            candidate.get("scene", {}).get("normalized", "").lower()
        )
        return float(query_scene.lower() == candidate_scene)

    def _tag_overlap(self, query: Dict, candidate: Dict) -> float:
        query_tokens = {token.lower() for token in query.get("tokens", [])}
        if not query_tokens:
            return 0.0

        metadata_tags = {tag.lower() for tag in candidate.get("tags", [])}
        caption_words = set(
            candidate.get("caption", "")
            .lower()
            .replace(",", "")
            .replace(".", "")
            .split()
        )

        candidate_tokens = metadata_tags | caption_words
        overlap = len(query_tokens & candidate_tokens)
        return overlap / len(query_tokens)

    # ==========================================================
    # Dynamic Score Computation
    # ==========================================================

    def _compute_score(self, query: Dict, candidate: Dict):
        scores = {}
        weights = {}

        # Vector similarity — always present
        scores["vector"] = candidate["vector_score"]
        weights["vector"] = self.vector_weight

        if query.get("garments"):
            scores["garment"] = self._garment_match(query, candidate)
            weights["garment"] = self.garment_weight

        if query.get("styles"):
            scores["style"] = self._style_match(query, candidate)
            weights["style"] = self.style_weight

        if query.get("people"):
            scores["people"] = self._people_match(query, candidate)
            weights["people"] = self.people_weight

        if query.get("scene"):
            scores["scene"] = self._scene_match(query, candidate)
            weights["scene"] = self.scene_weight

        if query.get("tokens"):
            scores["tag"] = self._tag_overlap(query, candidate)
            weights["tag"] = self.tag_weight

        total_weight = sum(weights.values())

        weighted_sum = 0.0
        for key in scores:
            weighted_sum += scores[key] * weights[key]

        final_score = weighted_sum / total_weight

        return final_score, scores

    # ==========================================================
    # Public API
    # ==========================================================

    def rerank(
        self,
        parsed_query: Dict,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Rerank FAISS candidates using hybrid semantic scoring.

        Parameters
        ----------
        parsed_query : Dict
            Output of SemanticParser.parse()
        candidates : List[Dict]
            Output of HybridSearcher.search()

        Returns
        -------
        List[Dict]
            Candidates sorted by final_score, descending.
        """
        reranked = []

        for candidate in candidates:
            item = candidate.copy()
            final_score, scores = self._compute_score(parsed_query, item)

            # -------------------------------------------------
            # Explainability — keep individual signal scores
            # visible on each result, useful for debugging and
            # for the write-up (show *why* a result ranked high)
            # -------------------------------------------------
            item["garment_score"] = scores.get("garment", 0.0)
            item["style_score"] = scores.get("style", 0.0)
            item["people_score"] = scores.get("people", 0.0)
            item["scene_score"] = scores.get("scene", 0.0)
            item["tag_score"] = scores.get("tag", 0.0)
            item["final_score"] = final_score

            reranked.append(item)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked