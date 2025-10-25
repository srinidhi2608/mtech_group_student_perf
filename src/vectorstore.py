"""
FAISS vector store helper to persist index and metadata.
"""
import os
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS-based vector store for document embeddings.
    """
    
    def __init__(self, embedding_dim: int = 384, index_path: str = "models/faiss.index", 
                 metadata_path: str = "models/faiss_meta.json"):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        
        # Try to load existing index and metadata
        self.load()
        
        # If no index exists, create a new one
        if self.index is None:
            logger.info(f"Creating new FAISS index with dimension {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            embeddings: Array of embeddings (n_docs, embedding_dim)
            documents: List of document metadata dictionaries
        """
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, documents)
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return [], []
        
        # Ensure query embedding is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # Get corresponding documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return distances[0].tolist(), results
    
    def save(self) -> None:
        """
        Save FAISS index and metadata to disk.
        """
        try:
            # Ensure models directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_path} ({len(self.metadata)} documents)")
                
                return True
            else:
                logger.info("No existing index found, will create new one")
                return False
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def clear(self) -> None:
        """
        Clear the vector store.
        """
        logger.info("Clearing vector store")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "total_documents": len(self.metadata)
        }

# Global instance to avoid recreating the vector store
_global_vector_store = None

def get_vector_store(embedding_dim: int = 384) -> VectorStore:
    """
    Get or create a global vector store instance.
    
    Args:
        embedding_dim: Dimension of embeddings
        
    Returns:
        VectorStore instance
    """
    global _global_vector_store
    
    if _global_vector_store is None:
        _global_vector_store = VectorStore(embedding_dim)
    
    return _global_vector_store
