"""
evaluation/metrics.py

Evaluation metrics for the CA-HMFR retrieval system.

Since the project uses free-form natural language queries and does not
have ground-truth relevance labels, this module computes descriptive
statistics over the reranked retrieval results.

These metrics are useful for analysing the behaviour of the hybrid
reranker and comparing different versions of the retrieval pipeline.
"""

from typing import Dict, List
import numpy as np


class RetrievalMetrics:

    def __init__(self):
        self.reset()

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):

        self.vector_scores = []

        self.final_scores = []

        self.garment_scores = []

        self.style_scores = []

        self.people_scores = []

        self.scene_scores = []

        self.tag_scores = []

    # ---------------------------------------------------------
    # Update metrics using one query's results
    # ---------------------------------------------------------

    def update(
        self,
        results: List[Dict]
    ):

        for item in results:

            self.vector_scores.append(
                item.get("vector_score", 0.0)
            )

            self.final_scores.append(
                item.get("final_score", 0.0)
            )

            self.garment_scores.append(
                item.get("garment_score", 0.0)
            )

            self.style_scores.append(
                item.get("style_score", 0.0)
            )

            self.people_scores.append(
                item.get("people_score", 0.0)
            )

            self.scene_scores.append(
                item.get("scene_score", 0.0)
            )

            self.tag_scores.append(
                item.get("tag_score", 0.0)
            )

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    def _mean(self, values):

        if len(values) == 0:
            return 0.0

        return float(np.mean(values))

    # ---------------------------------------------------------
    # Compute summary
    # ---------------------------------------------------------

    def summary(self):

        return {

            "Average Vector Score":
                self._mean(self.vector_scores),

            "Average Final Score":
                self._mean(self.final_scores),

            "Average Garment Score":
                self._mean(self.garment_scores),

            "Average Style Score":
                self._mean(self.style_scores),

            "Average People Score":
                self._mean(self.people_scores),

            "Average Scene Score":
                self._mean(self.scene_scores),

            "Average Tag Score":
                self._mean(self.tag_scores),

            "Queries Evaluated":
                len(self.final_scores)
        }

    # ---------------------------------------------------------
    # Pretty print
    # ---------------------------------------------------------

    def print_summary(self):

        summary = self.summary()

        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        for key, value in summary.items():

            if isinstance(value, float):
                print(f"{key:<30}: {value:.4f}")
            else:
                print(f"{key:<30}: {value}")