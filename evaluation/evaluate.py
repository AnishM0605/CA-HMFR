"""
evaluation/evaluate.py

Runs the complete evaluation pipeline.

Workflow

Sample Queries
      ↓
Fashion Retrieval System
      ↓
Top-K Results
      ↓
Metrics
      ↓
Save Results
"""

import json
from pathlib import Path

from evaluation.sample_queries import SAMPLE_QUERIES
from evaluation.metrics import RetrievalMetrics


def evaluate(
    retrieval_system,
    top_k=5,
    save_results=True,
):

    metrics = RetrievalMetrics()

    evaluation_results = []

    print("=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    for query in SAMPLE_QUERIES:

        print(f"\nQuery : {query}")
        print("-" * 80)

        results = retrieval_system.search(
            query=query,
            top_k=top_k
        )

        metrics.update(results)

        query_results = []

        for rank, result in enumerate(results, start=1):

            print(
                f"{rank}. "
                f"{result['caption']} "
                f"(Score: {result['final_score']:.4f})"
            )

            query_results.append({

                "rank": rank,

                "image_name": result["image_name"],

                "caption": result["caption"],

                "vector_score": result["vector_score"],

                "final_score": result["final_score"],

                "garment_score": result.get(
                    "garment_score", 0.0
                ),

                "style_score": result.get(
                    "style_score", 0.0
                ),

                "people_score": result.get(
                    "people_score", 0.0
                ),

                "scene_score": result.get(
                    "scene_score", 0.0
                ),

                "tag_score": result.get(
                    "tag_score", 0.0
                )
            })

        evaluation_results.append({

            "query": query,

            "results": query_results

        })

    print("\n")

    metrics.print_summary()

    if save_results:

        output_dir = Path("evaluation/results")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = output_dir / "evaluation_results.json"

        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                evaluation_results,
                f,
                indent=4
            )

        print("\nResults saved to")
        print(output_file)

    return evaluation_results