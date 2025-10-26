"""
FAISS vector store helper with a backwards-compatible FaissStore wrapper.

This file exposes:
- VectorStore: primary implementation that uses faiss when available, else a simple numpy fallback.
- FaissStore: thin wrapper with the methods app/main.py expects:
    - add(vectors, metadatas)
    - load()
    - save()
    - search(query_vector, k) -> list of metadata dicts (each includes 'score')
The wrapper keeps behavior stable whether faiss is installed or not.
"""
import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

# Try to import faiss; if unavailable we fall back to a numpy brute-force store
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # will be handled below

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store that persists an index and associated metadata.
    Uses faiss if available; otherwise keeps vectors in-memory and uses cosine similarity.
    """

    def __init__(self, embedding_dim: int = 384, index_path: str = "models/faiss.index",
                 metadata_path: str = "models/faiss_meta.json", vectors_path: str = "models/faiss_vectors.npy"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.vectors_path = vectors_path
        self.index = None  # faiss index or None
        self.metadata: List[Dict] = []
        self._vectors: Optional[np.ndarray] = None  # numpy array of shape (n, dim) used for fallback

        # Attempt to load persisted store
        self.load()

        # If no index and faiss available, create a faiss index
        if self.index is None and faiss is not None:
            logger.info(f"Creating new FAISS IndexFlatL2 with dim={self.embedding_dim}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            # if vectors were loaded into _vectors, add them to the faiss index
            if self._vectors is not None and self._vectors.shape[0] > 0:
                self.index.add(self._vectors.astype("float32"))

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        """
        Add documents and embeddings.
        embeddings: (n_docs, dim)
        documents: list of metadata dicts (len == n_docs)
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings must have shape (n, {self.embedding_dim})")

        n = embeddings.shape[0]
        if n != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        embeddings = embeddings.astype("float32")

        if faiss is not None and self.index is not None:
            try:
                self.index.add(embeddings)
            except Exception as e:
                logger.error(f"Faiss add failed: {e}; falling back to memory store")
                # fall through to memory fallback
                self._add_memory(embeddings, documents)
                return
        else:
            self._add_memory(embeddings, documents)
            return

        # store metadata
        self.metadata.extend(documents)
        # keep vectors in memory for persistence in case process restarts
        if self._vectors is None:
            self._vectors = embeddings.copy()
        else:
            self._vectors = np.vstack([self._vectors, embeddings])
        logger.info(f"Added {n} docs (faiss). Total metadata: {len(self.metadata)}")

    def _add_memory(self, embeddings: np.ndarray, documents: List[Dict]):
        """Fallback: store embeddings in a numpy array and metadata list"""
        if self._vectors is None:
            self._vectors = embeddings.copy()
        else:
            self._vectors = np.vstack([self._vectors, embeddings])
        self.metadata.extend(documents)
        logger.info(f"Added {embeddings.shape[0]} docs (memory). Total metadata: {len(self.metadata)}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar documents. Returns a list of metadata dicts augmented with 'score'.
        Score is between 0 and 1 (higher is better).
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        q = query_embedding.astype("float32")

        if faiss is not None and self.index is not None and getattr(self.index, "ntotal", 0) > 0:
            k = min(k, int(self.index.ntotal))
            D, I = self.index.search(q, k)
            distances = D[0].tolist()
            indices = I[0].tolist()
            results = []
            for dist, idx in zip(distances, indices):
                if 0 <= idx < len(self.metadata):
                    # convert L2 distance to a score in (0,1]
                    score = float(1.0 / (1.0 + float(dist)))
                    m = dict(self.metadata[idx])
                    m["score"] = score
                    results.append(m)
            return results

        # Fallback: brute-force cosine similarity against self._vectors
        if self._vectors is None or len(self.metadata) == 0:
            logger.warning("Vector store empty")
            return []

        # normalize
        vecs = self._vectors.astype("float32")
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        v_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        sims = (v_norm @ q_norm.T).ravel()  # cosine similarity
        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            m = dict(self.metadata[int(idx)])
            m["score"] = score
            results.append(m)
        return results

    def save(self) -> None:
        """
        Persist index and metadata. If faiss present and index built, write faiss index;
        also persist vectors and metadata for fallback.
        """
        try:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            # Save faiss index if available
            if faiss is not None and self.index is not None:
                try:
                    faiss.write_index(self.index, self.index_path)
                    logger.info(f"Saved faiss index to {self.index_path}")
                except Exception as e:
                    logger.error(f"Failed to save faiss index: {e}")
            # Save vectors (numpy) for fallback
            if self._vectors is not None:
                try:
                    np.save(self.vectors_path, self._vectors)
                    logger.info(f"Saved {self._vectors.shape[0]} vectors to {self.vectors_path}")
                except Exception as e:
                    logger.error(f"Failed to save vectors: {e}")
            # Save metadata
            try:
                with open(self.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved metadata to {self.metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
        except Exception as e:
            logger.error(f"Error during save: {e}")

    def load(self) -> bool:
        """
        Attempt to load a persisted store (faiss index, vectors, metadata).
        Returns True if any persistence was loaded.
        """
        loaded_any = False
        try:
            if faiss is not None and os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded faiss index ({self.index_path}) with ntotal={self.index.ntotal}")
                    loaded_any = True
                except Exception as e:
                    logger.error(f"Failed to read faiss index: {e}")
                    self.index = None
            # load vectors numpy if exists
            if os.path.exists(self.vectors_path):
                try:
                    self._vectors = np.load(self.vectors_path)
                    logger.info(f"Loaded vectors from {self.vectors_path} shape={self._vectors.shape}")
                    loaded_any = True
                except Exception as e:
                    logger.error(f"Failed to load vectors: {e}")
                    self._vectors = None
            # load metadata
            if os.path.exists(self.metadata_path):
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    logger.info(f"Loaded metadata from {self.metadata_path} (count={len(self.metadata)})")
                    loaded_any = True
                except Exception as e:
                    logger.error(f"Failed to load metadata: {e}")
                    self.metadata = []
        except Exception as e:
            logger.error(f"Error during load: {e}")
        return loaded_any

    def clear(self) -> None:
        """Clear the store in memory (and on disk removed only when save called)."""
        self.index = None
        self._vectors = None
        self.metadata = []
        if faiss is not None:
            try:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            except Exception:
                self.index = None
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict:
        total = 0
        if self._vectors is not None:
            total = int(self._vectors.shape[0])
        elif self.index is not None and hasattr(self.index, "ntotal"):
            total = int(self.index.ntotal)
        return {"total_vectors": total, "embedding_dim": self.embedding_dim, "total_documents": len(self.metadata)}


# Backwards-compatible wrapper class expected by app/main.py
class FaissStore:
    """
    Thin wrapper around VectorStore exposing the API app expects:
      - add(vectors, metadatas)
      - load(), save()
      - search(query_vector, k) -> returns list of metadata dicts with 'score'
    """
    def __init__(self, dim: int = 384, index_path: str = "models/faiss.index", meta_path: str = "models/faiss_meta.json"):
        self._store = VectorStore(embedding_dim=dim, index_path=index_path, metadata_path=meta_path, vectors_path=meta_path.replace(".json", ".npy"))
        self.dim = dim

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        # accept shapes (n, dim)
        self._store.add_documents(vectors, metadatas)

    def save(self):
        self._store.save()

    def load(self):
        return self._store.load()

    def search(self, query_vector: np.ndarray, k: int = 6) -> List[Dict]:
        """
        Return a list of metadata dicts with a 'score' key. This matches expected shape by app/main.py.
        """
        # underlying VectorStore.search already returns list of metadata dicts with score
        return self._store.search(query_vector, k=k)


# convenience factory (legacy compatibility)
_global_vector_store = None


def get_vector_store(embedding_dim: int = 384) -> VectorStore:
    """
    Return a shared VectorStore instance (legacy compatibility).
    """
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = VectorStore(embedding_dim=embedding_dim)
    return _global_vector_store