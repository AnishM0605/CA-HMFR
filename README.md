# CA-HMFR: Compositional-Aware Hybrid Multimodal Fashion & Context Retrieval

> An intelligent multimodal fashion retrieval system that combines visual semantics, scene understanding, clothing attributes, and hybrid ranking to improve retrieval beyond a vanilla CLIP-based pipeline.

---

## Overview

Modern vision-language models such as CLIP enable zero-shot image retrieval using natural language. However, vanilla CLIP often struggles with:

- Fine-grained fashion attributes
- Clothing composition
- Scene understanding
- Multi-attribute queries
- Context-aware retrieval

CA-HMFR (Compositional-Aware Hybrid Multimodal Fashion & Context Retrieval) addresses these limitations by combining multiple complementary sources of information into a unified retrieval framework.

Instead of relying solely on image embeddings, the system enriches every indexed image with:

- FashionCLIP visual embeddings
- BLIP-generated captions
- Clothing attribute extraction
- Places365 scene understanding
- Structured metadata
- Hybrid semantic reranking

The result is a retrieval system capable of understanding queries involving clothing, colors, locations, and style simultaneously.

---

# Features

- Fashion-specific image embeddings using FashionCLIP
- Automatic image caption generation with BLIP
- Scene recognition using Places365
- Clothing attribute extraction
- Metadata enrichment
- Fast vector search using FAISS
- Natural language query encoding
- Semantic query parsing
- Hybrid reranking
- Zero-shot retrieval
- Modular architecture
- Production-ready pipeline

---

# Repository Structure

```
fashion-retrieval/

│
├── config.py
│
├── models/
│   ├── clip_model.py
│   ├── blip_model.py
│   ├── scene_model.py
│
├── indexer/
│   ├── build_index.py
│   ├── caption_generator.py
│   ├── clothing_parser.py
│   ├── metadata_builder.py
│   ├── multimodal_encoder.py
│   └── scene_resolver.py
│
├── retriever/
│   ├── query.py
│   ├── semantic_parser.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   └── retrieval_system.py
│
├── evaluation/
│   ├── sample_queries.py
│   ├── metrics.py
│   ├── evaluate.py
│   └── results/
│
├── index_store/
│   ├── metadata.json
│   ├── fashionclip_embeddings.npy
│   └── faiss.index
│
└── notebooks/
    ├── 01_build_index.ipynb
    └── 02_retrieval_demo.ipynb
```

---

# System Architecture

```
                    Offline Indexing Pipeline

                 Fashion Images Dataset
                         │
                         ▼
                FashionCLIP Encoder
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      BLIP Caption   Clothing Parser  Places365
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Metadata Builder
                         │
                         ▼
             Embedding + Metadata Store
                         │
                         ▼
                  FAISS Vector Index
```

---

```
                    Online Retrieval Pipeline

              Natural Language Query
                        │
                        ▼
             FashionCLIP Text Encoder
                        │
                        ▼
                Semantic Query Parser
                        │
                        ▼
                  FAISS Search
                        │
                        ▼
               Hybrid Re-ranking
                        │
                        ▼
                 Top-K Retrieved Images
                        │
                        ▼
              Visualization & Explanation
```

---

# Dataset

The project uses the **Fashionpedia Dataset**.

Dataset characteristics:

- Diverse fashion categories
- Multiple clothing styles
- Rich color variations
- Indoor and outdoor environments
- Fashion-centric imagery

For efficient experimentation, a subset of images was sampled and indexed.

Each image is enriched with:

- Visual embedding
- Caption
- Clothing attributes
- Scene category
- Metadata

---

# Models Used

## 1. FashionCLIP

Purpose:

- Image embeddings
- Text embeddings
- Zero-shot multimodal retrieval

Reason for selection:

FashionCLIP is specifically adapted for fashion semantics and provides stronger fashion-aware representations than generic CLIP.

---

## 2. BLIP

Purpose:

- Automatic caption generation

BLIP converts visual information into descriptive natural language, providing additional semantic context for retrieval.

---

## 3. Places365

Purpose:

Scene recognition.

Example predictions:

- Office
- Park
- Street
- Home
- Shopping mall

This enables context-aware retrieval.

---

## 4. Clothing Attribute Extraction

Extracts fashion-specific information such as:

- Clothing type
- Garment category
- Color
- Outfit information

These attributes are incorporated into the metadata.

---

## 5. FAISS

Facebook AI Similarity Search is used for efficient nearest-neighbor retrieval over image embeddings.

Advantages:

- Fast similarity search
- Scalable indexing
- Low latency retrieval

---

# Offline Indexing Pipeline

Notebook:

```
01_build_index.ipynb
```

Pipeline:

```
Images
   │
   ▼
FashionCLIP Embeddings
   │
   ▼
BLIP Caption Generation
   │
   ▼
Clothing Attribute Extraction
   │
   ▼
Places365 Scene Classification
   │
   ▼
Metadata Construction
   │
   ▼
Embedding Storage (.npy)
   │
   ▼
FAISS Index Creation
```

Generated files:

```
fashionclip_embeddings.npy

metadata.json

faiss.index
```

---

# Online Retrieval Pipeline

Notebook:

```
02_retrieval_demo.ipynb
```

Pipeline:

```
Natural Language Query
        │
        ▼
Query Encoding
        │
        ▼
Semantic Parsing
        │
        ▼
FAISS Search
        │
        ▼
Hybrid Re-ranking
        │
        ▼
Top-K Retrieval
```

---

# Hybrid Retrieval Strategy

Unlike a vanilla CLIP retrieval pipeline, CA-HMFR combines multiple signals.

Ranking considers:

- Visual similarity
- Caption similarity
- Clothing attributes
- Scene compatibility
- Semantic query understanding

This produces more contextually relevant retrieval results for complex fashion queries.

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/fashion-retrieval.git

cd fashion-retrieval
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

---

# Usage

## Step 1

Run

```
01_build_index.ipynb
```

This will generate:

- Image embeddings
- Metadata
- FAISS index

---

## Step 2

Run

```
02_retrieval_demo.ipynb
```

Example query:

```
Professional business attire inside a modern office
```

Example output:

```
Top-5 most relevant fashion images
```

---

# Example Queries

Attribute Query

```
A person wearing a bright yellow raincoat
```

Context Query

```
Professional business attire inside a modern office
```

Semantic Query

```
Someone wearing a blue shirt sitting on a park bench
```

Style Query

```
Casual weekend outfit for a city walk
```

Compositional Query

```
A red tie and a white shirt in a formal setting
```

---

# Evaluation

The system is evaluated using representative natural-language fashion queries covering:

- Clothing attributes
- Scene context
- Style understanding
- Multi-attribute composition
- Zero-shot retrieval

The evaluation pipeline measures retrieval quality by comparing the returned Top-K images against the expected semantic intent of each query.

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- FashionCLIP
- BLIP
- Places365
- FAISS
- NumPy
- OpenCV
- PIL
- Matplotlib
- Jupyter Notebook

---

# Why CA-HMFR?

Vanilla CLIP retrieves images primarily based on global semantic similarity.

CA-HMFR improves retrieval by integrating:

- Fashion-aware embeddings
- Scene understanding
- Caption semantics
- Clothing attributes
- Hybrid reranking

This richer representation enables better handling of compositional and context-aware fashion queries.

---

# Current Limitations

- Retrieval quality depends on caption generation accuracy.
- Clothing attribute extraction relies on heuristic parsing.
- Scene recognition may be imperfect for ambiguous environments.
- The current FAISS index is memory-based and not distributed.
- Very fine-grained garment relationships remain challenging.

---

# Future Work

Potential improvements include:

- Weather-aware retrieval
- Geographic location understanding
- Stronger fashion-specific reranking models
- Larger-scale vector databases (e.g., Milvus, Qdrant)
- Cross-attention based multimodal fusion
- Learning-to-rank approaches
- User feedback-driven retrieval refinement
- Fine-tuning on domain-specific fashion datasets

---

# References

- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- FashionCLIP
- BLIP: Bootstrapping Language-Image Pre-training
- Places365
- FAISS
- Fashionpedia Dataset



