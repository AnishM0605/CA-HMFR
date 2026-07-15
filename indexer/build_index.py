import math
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from config import (
    FASHIONCLIP_MODEL_PATH,
    BLIP_MODEL_PATH,
    PLACES365_MODEL_PATH,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    BATCH_SIZE
)

from models.vision_encoder import VisionEncoder
from indexer.caption_generator import CaptionGenerator
from indexer.clothing_parser import ClothingParser
from indexer.scene_classifier import SceneClassifier
from indexer.scene_resolver import SceneResolver
from indexer.metadata_builder import MetadataBuilder


class BuildIndexPipeline:
    """
    Complete indexing pipeline.

    Generates:
    • Vision encoder embeddings (FashionCLIP or OpenCLIP)
    • metadata.json

    Does NOT build FAISS — that happens in a separate step.
    """

    def __init__(self):
        print("=" * 70)
        print("Loading Models...")
        print("=" * 70)

        self.encoder = VisionEncoder(FASHIONCLIP_MODEL_PATH)
        self.captioner = CaptionGenerator(BLIP_MODEL_PATH)
        self.parser = ClothingParser()
        self.scene_classifier = SceneClassifier(PLACES365_MODEL_PATH)
        self.scene_resolver = SceneResolver()

        print("✓ Vision Encoder Loaded")
        print("✓ BLIP Loaded")
        print("✓ Clothing Parser Loaded")
        print("✓ Places365 Loaded")
        print("✓ Scene Resolver Loaded")
        print("=" * 70)

    def create_batches(self, image_paths: List[str]):
        """Yield batches of image paths."""
        for i in range(0, len(image_paths), BATCH_SIZE):
            yield image_paths[i:i + BATCH_SIZE]

    def build(self, image_paths: List[str]):
        embeddings = []
        metadata = []
        image_index = 0

        print()
        print("=" * 70)
        print(f"Processing {len(image_paths)} Images")
        print("=" * 70)

        # ======================================================
        # Process Images Batch-by-Batch
        # ======================================================
        for batch_paths in tqdm(
            self.create_batches(image_paths),
            total=math.ceil(len(image_paths) / BATCH_SIZE),
            desc="Building Index"
        ):
            # Generate embeddings in batch (efficient — one forward pass per batch)
            batch_embeddings = self.encoder.encode_images(batch_paths)

            # Process each image individually for the rest of the pipeline
            for embedding, image_path in zip(batch_embeddings, batch_paths):
                try:
                    caption = self.captioner.generate_caption(image_path)
                    clothing = self.parser.parse(caption)
                    places_scene = self.scene_classifier.predict(image_path)
                    resolved_scene = self.scene_resolver.resolve(caption, places_scene)

                    image_metadata = MetadataBuilder.build(
                        image_id=image_index,
                        image_path=image_path,
                        caption=caption,
                        clothing=clothing,
                        scene=resolved_scene
                    )

                    metadata.append(image_metadata)
                    embeddings.append(embedding.numpy())
                    image_index += 1

                except Exception as e:
                    print("\n-----------------------------------")
                    print(f"Skipping Image:")
                    print(image_path)
                    print("-----------------------------------")
                    print(e)
                    print("-----------------------------------")
                    continue

        # ======================================================
        # Finalize and Save
        # ======================================================
        print()
        print("=" * 70)
        print("Finalizing Outputs...")
        print("=" * 70)

        embeddings = np.asarray(embeddings, dtype=np.float32)

        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        np.save(EMBEDDINGS_PATH, embeddings)
        MetadataBuilder.save(metadata, METADATA_PATH)

        print()
        print("=" * 70)
        print("Index Building Complete")
        print("=" * 70)
        print(f"Images Indexed    : {len(metadata)}")
        print(f"Embeddings Shape  : {embeddings.shape}")
        print(f"Metadata Saved    : {METADATA_PATH}")
        print(f"Embeddings Saved  : {EMBEDDINGS_PATH}")
        print("=" * 70)

        return embeddings, metadata


def main(image_paths):
    pipeline = BuildIndexPipeline()
    embeddings, metadata = pipeline.build(image_paths)
    return embeddings, metadata