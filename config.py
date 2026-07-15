from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INDEX_DIR = PROJECT_ROOT / "index_store"

CONFIG_DIR = PROJECT_ROOT / "config"

# ==========================================================
# Model Paths
# ==========================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")

FASHIONCLIP_MODEL_PATH = DRIVE_ROOT / "FASHION_CLIP_MODEL"

BLIP_MODEL_PATH = DRIVE_ROOT / "BLIP_MODEL"

PLACES365_MODEL_PATH = DRIVE_ROOT / "PLACES365_MODEL"

# ==========================================================
# Configuration Files
# ==========================================================

SCENE_MAPPING_PATH = CONFIG_DIR / "scene_mapping.json"

# ==========================================================
# Output Files
# ==========================================================

EMBEDDINGS_PATH = INDEX_DIR / "fashionclip_embeddings.npy"

METADATA_PATH = INDEX_DIR / "metadata.json"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"

# ==========================================================
# Runtime
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32