"""
Embeddings wrapper using sentence-transformers.
"""
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingsModel:
    """
    Wrapper for sentence-transformers embedding model.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        logger.info(f"Loading embeddings model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embeddings model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim

# Global instance to avoid reloading the model multiple times
_global_embeddings_model = None

def get_embeddings_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingsModel:
    """
    Get or create a global embeddings model instance.
    
    Args:
        model_name: Name of the sentence-transformers model
        
    Returns:
        EmbeddingsModel instance
    """
    global _global_embeddings_model
    
    if _global_embeddings_model is None or _global_embeddings_model.model_name != model_name:
        _global_embeddings_model = EmbeddingsModel(model_name)
    
    return _global_embeddings_model
