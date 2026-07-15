"""
retriever/retrieval_system.py

High-level retrieval pipeline for CA-HMFR.

This module orchestrates the complete online retrieval workflow.

Pipeline

User Query
    ↓
Semantic Parser
    ↓
FashionCLIP Query Encoder
    ↓
FAISS Hybrid Search
    ↓
Hybrid Reranker
    ↓
Top-K Results
"""

from typing import Dict, List


class FashionRetrievalSystem:
    """
    High-level interface for the CA-HMFR retrieval system.

    This class orchestrates all retrieval components while keeping
    each individual module independent.
    """

    def __init__(
        self,
        query_encoder,
        semantic_parser,
        hybrid_searcher,
        reranker,
    ):

        self.query_encoder = query_encoder
        self.semantic_parser = semantic_parser
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_query(
        self,
        query: str,
    ):

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if len(query) == 0:
            raise ValueError("Query cannot be empty.")

        return query

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 50,
    ) -> List[Dict]:
        """
        Perform end-to-end fashion retrieval.

        Parameters
        ----------
        query : str
            User text query.

        top_k : int
            Number of final results.

        retrieval_k : int
            Number of FAISS candidates before reranking.

        Returns
        -------
        List[Dict]
            Top ranked retrieval results.
        """

        query = self._validate_query(query)

        # -----------------------------------------
        # Semantic Parsing
        # -----------------------------------------

        parsed_query = self.semantic_parser.parse(query)

        # -----------------------------------------
        # Query Encoding
        # -----------------------------------------

        query_embedding = self.query_encoder.encode_text(query)

        # -----------------------------------------
        # Initial Retrieval
        # -----------------------------------------

        candidates = self.hybrid_searcher.search(
            query_embedding=query_embedding,
            top_k=retrieval_k,
        )

        # -----------------------------------------
        # Hybrid Reranking
        # -----------------------------------------

        reranked = self.reranker.rerank(
            parsed_query=parsed_query,
            candidates=candidates,
        )

        return reranked[:top_k]

    # ---------------------------------------------------------
    # Optional Debug API
    # ---------------------------------------------------------

    def explain(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 50,
    ):
        """
        Returns both parsed query and ranked results.
        Useful for debugging and demonstrations.
        """

        query = self._validate_query(query)

        parsed_query = self.semantic_parser.parse(query)

        query_embedding = self.query_encoder.encode_text(query)

        candidates = self.hybrid_searcher.search(
            query_embedding=query_embedding,
            top_k=retrieval_k,
        )

        reranked = self.reranker.rerank(
            parsed_query=parsed_query,
            candidates=candidates,
        )

        return {
            "query": query,
            "parsed_query": parsed_query,
            "results": reranked[:top_k],
        }