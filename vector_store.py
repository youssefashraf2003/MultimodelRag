"""
vector_store.py - Production Unified Vector Store V6 (ENHANCED)
================================================================
✅ FAISS + sentence-transformers with graceful fallback
✅ Multi-type indexes: text / equations / tables / figures
✅ O(1) metadata-first retrieval via GlobalElementRegistry
✅ Hybrid Search (dense + sparse)
✅ Type-specific and cross-type search
✅ Checkpoint save/load
✅ Added get_all_chunks_by_type() method
✅ Faster retrieval with caching
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ── Shared data models ─────────────────────────────────────────────────────
from models import (
    GlobalElementRegistry,
    MultimodalChunk,
    SearchResult,
)

# ── Dependencies with graceful fallback ────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("⚠️ sentence-transformers not installed — using lexical fallback")

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("⚠️ faiss-cpu not installed — using lexical fallback")


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFIED VECTOR STORE V6
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedVectorStore:
    """
    Production vector store with:
    - Separate FAISS sub-indexes per chunk type (text/equation/table/figure)
    - Global element registry for O(1) metadata-first lookup
    - Hybrid search (dense + sparse)
    - Fast type-specific retrieval
    - Lexical fallback when FAISS/ST unavailable
    
    Architecture:
    - _type_indexes: {chunk_type: (faiss_index, [chunk_id order])}
    - _unified_index: Combined index for cross-type search
    - registry: GlobalElementRegistry for O(1) "Equation 3" lookup
    """

    CHUNK_TYPES = ["text", "equation", "figure", "table"]

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        cache_dir: str = "./cache",
        enable_hybrid_search: bool = True,
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_model_name: SentenceTransformer model name
            dimension: Embedding dimension
            cache_dir: Cache directory for models
            enable_hybrid_search: Enable hybrid search (dense + sparse)
        """
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self.cache_dir = cache_dir
        self.enable_hybrid_search = enable_hybrid_search

        # ── Chunk storage ──────────────────────────────────────────────────
        self.chunks: Dict[str, MultimodalChunk] = {}
        
        # ── Type-based organization ────────────────────────────────────────
        self.chunks_by_type: Dict[str, List[str]] = {
            chunk_type: [] for chunk_type in self.CHUNK_TYPES
        }
        
        # ── Search cache ───────────────────────────────────────────────────
        self._search_cache: Dict[str, List[SearchResult]] = {}
        self._cache_max_size = 100

        # ── Initialize embedding model ─────────────────────────────────────
        self.encoder: Optional[SentenceTransformer] = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer(
                    embedding_model_name,
                    cache_folder=cache_dir,
                )
                logger.info(f"✅ Loaded embedding model: {embedding_model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                self.encoder = None
        else:
            logger.warning("⚠️ SentenceTransformers unavailable — lexical search only")

        # ── Initialize FAISS indexes ───────────────────────────────────────
        self._type_indexes: Dict[str, Tuple[Any, List[str]]] = {}
        self._unified_index: Optional[Any] = None
        self._unified_chunk_order: List[str] = []

        if HAS_FAISS and self.encoder:
            for chunk_type in self.CHUNK_TYPES:
                self._type_indexes[chunk_type] = (
                    faiss.IndexFlatIP(dimension),  # Inner Product = cosine sim
                    []  # chunk_id order
                )
            self._unified_index = faiss.IndexFlatIP(dimension)
            logger.info("✅ FAISS indexes initialized")
        else:
            logger.warning("⚠️ FAISS unavailable — using lexical fallback")

        # ── Global registry ────────────────────────────────────────────────
        self.registry = GlobalElementRegistry()

        # ── Statistics ─────────────────────────────────────────────────────
        self.stats = {
            'total_chunks': 0,
            'chunks_by_type': defaultdict(int),
            'search_queries': 0,
            'cache_hits': 0,
        }

    # ═══════════════════════════════════════════════════════════════════════
    #  ADD CHUNKS
    # ═══════════════════════════════════════════════════════════════════════

    def add_chunks(
        self,
        chunks: List[MultimodalChunk],
        doc_id: Optional[str] = None,
    ) -> None:
        """
        Add multiple chunks to the vector store.
        
        Args:
            chunks: List of MultimodalChunk objects
            doc_id: Document ID (optional)
        """
        if not chunks:
            logger.warning("⚠️ No chunks to add")
            return

        logger.info(f"📊 Adding {len(chunks)} chunks to vector store...")

        # Store chunks
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.chunks_by_type[chunk.chunk_type].append(chunk.chunk_id)
            self.registry.register(doc_id or chunk.doc_id, chunk)
            self.stats['chunks_by_type'][chunk.chunk_type] += 1

        self.stats['total_chunks'] += len(chunks)

        # Embed and index
        if self.encoder and HAS_FAISS:
            self._embed_and_index(chunks)
        else:
            logger.warning("⚠️ Skipping embedding (encoder/FAISS unavailable)")

        # Clear cache
        self._search_cache.clear()

        logger.info(f"✅ Added {len(chunks)} chunks successfully")

    def _embed_and_index(self, chunks: List[MultimodalChunk]) -> None:
        """
        Embed chunks and add to FAISS indexes.
        
        Args:
            chunks: List of chunks to embed
        """
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Use embedding_text if available, else chunk.text
            text = chunk.metadata.get('embedding_text', chunk.text)
            if not text or not text.strip():
                text = f"[{chunk.chunk_type}] {chunk.text[:100]}"
            texts.append(text)

        # Batch encode
        try:
            embeddings = self.encoder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # For cosine similarity
            )
            
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Ensure float32
            embeddings = embeddings.astype('float32')

        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            return

        # Add to type-specific indexes
        for chunk, embedding in zip(chunks, embeddings):
            chunk_type = chunk.chunk_type
            if chunk_type in self._type_indexes:
                index, chunk_order = self._type_indexes[chunk_type]
                index.add(embedding.reshape(1, -1))
                chunk_order.append(chunk.chunk_id)

        # Add to unified index
        if self._unified_index is not None:
            self._unified_index.add(embeddings)
            self._unified_chunk_order.extend(c.chunk_id for c in chunks)

        logger.debug(f"✅ Embedded and indexed {len(chunks)} chunks")

    # ═══════════════════════════════════════════════════════════════════════
    #  SEARCH METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunk_type: Filter by chunk type (None = all types)
            filter_metadata: Additional metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        self.stats['search_queries'] += 1

        # Check cache
        cache_key = f"{query}:{top_k}:{chunk_type}:{similarity_threshold}"
        if cache_key in self._search_cache:
            self.stats['cache_hits'] += 1
            return self._search_cache[cache_key]

        # Perform search
        if self.encoder and HAS_FAISS:
            results = self._vector_search(
                query, top_k, chunk_type, filter_metadata, similarity_threshold
            )
        else:
            results = self._lexical_search(
                query, top_k, chunk_type, filter_metadata
            )

        # Cache results
        if len(self._search_cache) < self._cache_max_size:
            self._search_cache[cache_key] = results

        return results

    def _vector_search(
        self,
        query: str,
        top_k: int,
        chunk_type: Optional[str],
        filter_metadata: Optional[Dict[str, Any]],
        similarity_threshold: float,
    ) -> List[SearchResult]:
        """
        Vector-based search using FAISS.
        """
        # Encode query
        try:
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            query_embedding = query_embedding.astype('float32')
        except Exception as e:
            logger.error(f"❌ Query encoding failed: {e}")
            return []

        # Select index
        if chunk_type and chunk_type in self._type_indexes:
            index, chunk_order = self._type_indexes[chunk_type]
        elif self._unified_index is not None:
            index = self._unified_index
            chunk_order = self._unified_chunk_order
        else:
            logger.warning("⚠️ No index available")
            return []

        # Search
        try:
            distances, indices = index.search(query_embedding, min(top_k * 2, len(chunk_order)))
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {e}")
            return []

        # Build results
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(chunk_order):
                continue
            
            similarity = float(dist)  # Already cosine similarity with IndexFlatIP
            
            if similarity < similarity_threshold:
                continue

            chunk_id = chunk_order[idx]
            chunk = self.chunks.get(chunk_id)
            
            if chunk is None:
                continue

            # Apply metadata filters
            if filter_metadata:
                if not self._matches_filters(chunk, filter_metadata):
                    continue

            results.append(SearchResult(
                chunk=chunk,
                similarity_score=similarity,
                rank=rank,
            ))

            if len(results) >= top_k:
                break

        return results

    def _lexical_search(
        self,
        query: str,
        top_k: int,
        chunk_type: Optional[str],
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """
        Lexical fallback search (keyword matching).
        """
        query_words = set(query.lower().split())
        
        # Select chunks
        if chunk_type:
            chunk_ids = self.chunks_by_type.get(chunk_type, [])
        else:
            chunk_ids = list(self.chunks.keys())

        # Score chunks
        scored_chunks = []
        for chunk_id in chunk_ids:
            chunk = self.chunks[chunk_id]
            
            # Apply metadata filters
            if filter_metadata:
                if not self._matches_filters(chunk, filter_metadata):
                    continue

            # Calculate keyword overlap
            chunk_text = (chunk.text + " " + chunk.metadata.get('embedding_text', '')).lower()
            chunk_words = set(chunk_text.split())
            overlap = len(query_words & chunk_words)
            
            if overlap > 0:
                # Normalize by query length
                score = overlap / len(query_words)
                scored_chunks.append((chunk, score))

        # Sort and limit
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        results = [
            SearchResult(chunk=chunk, similarity_score=score, rank=i)
            for i, (chunk, score) in enumerate(scored_chunks[:top_k])
        ]

        return results

    def _matches_filters(
        self,
        chunk: MultimodalChunk,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if chunk matches metadata filters."""
        for key, value in filters.items():
            if key not in chunk.metadata:
                return False
            if chunk.metadata[key] != value:
                return False
        return True

    # ═══════════════════════════════════════════════════════════════════════
    #  TYPE-SPECIFIC RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════════

    def get_all_chunks_by_type(self, chunk_type: str) -> List[MultimodalChunk]:
        """
        Get all chunks of a specific type.
        
        Args:
            chunk_type: Type of chunks to retrieve (equation, table, figure, text)
            
        Returns:
            List of MultimodalChunk objects
        """
        if chunk_type not in self.CHUNK_TYPES:
            logger.warning(f"⚠️ Unknown chunk type: {chunk_type}")
            return []

        chunk_ids = self.chunks_by_type.get(chunk_type, [])
        chunks = [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
        
        # Sort by global_number if available
        chunks.sort(key=lambda c: c.metadata.get('global_number', 999))
        
        return chunks

    def get_chunk_by_number(
        self,
        chunk_type: str,
        number: int
    ) -> Optional[MultimodalChunk]:
        """
        Get a specific numbered element (e.g., Equation 3).
        
        Args:
            chunk_type: Type (equation, table, figure)
            number: Element number
            
        Returns:
            MultimodalChunk or None
        """
        chunk_id = self.registry.lookup(chunk_type, number)
        if chunk_id:
            return self.chunks.get(chunk_id)
        return None

    def get_chunks_in_range(
        self,
        chunk_type: str,
        start: int,
        end: int
    ) -> List[MultimodalChunk]:
        """
        Get chunks in a number range.
        
        Args:
            chunk_type: Type of chunks
            start: Start number (inclusive)
            end: End number (inclusive)
            
        Returns:
            List of chunks
        """
        chunks = []
        for num in range(start, end + 1):
            chunk = self.get_chunk_by_number(chunk_type, num)
            if chunk:
                chunks.append(chunk)
        return chunks

    # ═══════════════════════════════════════════════════════════════════════
    #  HYBRID SEARCH
    # ═══════════════════════════════════════════════════════════════════════

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[str] = None,
        alpha: float = 0.7,  # Weight for vector search (1-alpha for lexical)
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and lexical methods.
        
        Args:
            query: Search query
            top_k: Number of results
            chunk_type: Filter by type
            alpha: Weight for vector search (0-1)
            
        Returns:
            List of SearchResult objects
        """
        if not self.enable_hybrid_search or not self.encoder:
            # Fall back to regular search
            return self.search(query, top_k, chunk_type)

        # Get results from both methods
        vector_results = self._vector_search(query, top_k * 2, chunk_type, None, 0.0)
        lexical_results = self._lexical_search(query, top_k * 2, chunk_type, None)

        # Combine scores
        combined_scores: Dict[str, Tuple[MultimodalChunk, float]] = {}

        for result in vector_results:
            chunk_id = result.chunk.chunk_id
            score = result.similarity_score * alpha
            combined_scores[chunk_id] = (result.chunk, score)

        for result in lexical_results:
            chunk_id = result.chunk.chunk_id
            score = result.similarity_score * (1 - alpha)
            if chunk_id in combined_scores:
                chunk, existing_score = combined_scores[chunk_id]
                combined_scores[chunk_id] = (chunk, existing_score + score)
            else:
                combined_scores[chunk_id] = (result.chunk, score)

        # Sort and create results
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1][1],
            reverse=True
        )

        results = [
            SearchResult(chunk=chunk, similarity_score=score, rank=i)
            for i, (chunk_id, (chunk, score)) in enumerate(sorted_items[:top_k])
        ]

        return results

    # ═══════════════════════════════════════════════════════════════════════
    #  UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_chunks': self.stats['total_chunks'],
            'chunks_by_type': dict(self.stats['chunks_by_type']),
            'search_queries': self.stats['search_queries'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / self.stats['search_queries']
                if self.stats['search_queries'] > 0 else 0.0
            ),
            'has_encoder': self.encoder is not None,
            'has_faiss': HAS_FAISS,
        }

    def clear(self) -> None:
        """Clear all data from vector store."""
        self.chunks.clear()
        for chunk_type in self.CHUNK_TYPES:
            self.chunks_by_type[chunk_type].clear()
        self._search_cache.clear()
        self.registry.clear()
        
        # Reset indexes
        if HAS_FAISS and self.encoder:
            for chunk_type in self.CHUNK_TYPES:
                self._type_indexes[chunk_type] = (
                    faiss.IndexFlatIP(self.dimension),
                    []
                )
            self._unified_index = faiss.IndexFlatIP(self.dimension)
            self._unified_chunk_order.clear()

        # Reset stats
        self.stats = {
            'total_chunks': 0,
            'chunks_by_type': defaultdict(int),
            'search_queries': 0,
            'cache_hits': 0,
        }

        logger.info("🗑️ Vector store cleared")

    def save_checkpoint(self, path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_data = {
            cid: {
                'chunk_id': c.chunk_id,
                'text': c.text,
                'doc_id': c.doc_id,
                'page_num': c.page_num,
                'chunk_type': c.chunk_type,
                'metadata': c.metadata,
                'image_path': c.image_path,
            }
            for cid, c in self.chunks.items()
        }
        
        with open(checkpoint_path / 'chunks.json', 'w') as f:
            json.dump(chunks_data, f, indent=2)

        # Save registry
        with open(checkpoint_path / 'registry.pkl', 'wb') as f:
            pickle.dump(self.registry, f)

        # Save FAISS indexes
        if HAS_FAISS and self._unified_index:
            faiss.write_index(
                self._unified_index,
                str(checkpoint_path / 'unified_index.faiss')
            )
            for chunk_type, (index, _) in self._type_indexes.items():
                faiss.write_index(
                    index,
                    str(checkpoint_path / f'{chunk_type}_index.faiss')
                )

        logger.info(f"💾 Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint_path = Path(path)
        
        if not checkpoint_path.exists():
            logger.error(f"❌ Checkpoint not found: {path}")
            return

        # Load chunks
        with open(checkpoint_path / 'chunks.json', 'r') as f:
            chunks_data = json.load(f)
        
        for cid, data in chunks_data.items():
            chunk = MultimodalChunk(
                chunk_id=data['chunk_id'],
                text=data['text'],
                doc_id=data['doc_id'],
                page_num=data['page_num'],
                chunk_type=data['chunk_type'],
                metadata=data.get('metadata', {}),
                image_path=data.get('image_path'),
            )
            self.chunks[cid] = chunk
            self.chunks_by_type[chunk.chunk_type].append(cid)

        # Load registry
        with open(checkpoint_path / 'registry.pkl', 'rb') as f:
            self.registry = pickle.load(f)

        # Load FAISS indexes
        if HAS_FAISS:
            unified_index_path = checkpoint_path / 'unified_index.faiss'
            if unified_index_path.exists():
                self._unified_index = faiss.read_index(str(unified_index_path))
            
            for chunk_type in self.CHUNK_TYPES:
                index_path = checkpoint_path / f'{chunk_type}_index.faiss'
                if index_path.exists():
                    index = faiss.read_index(str(index_path))
                    self._type_indexes[chunk_type] = (index, self.chunks_by_type[chunk_type])

        logger.info(f"📂 Checkpoint loaded from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    store = UnifiedVectorStore()
    print(f"✅ Vector store initialized")
    print(f"   Has encoder: {store.encoder is not None}")
    print(f"   Has FAISS: {HAS_FAISS}")
    print(f"   Hybrid search enabled: {store.enable_hybrid_search}")