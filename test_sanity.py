"""
Basic sanity tests for PDF ingestion, embeddings, and vector store.
"""
import os
import sys
import tempfile
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_ingest import chunk_text, extract_text_from_pdf
from src.embeddings import EmbeddingsModel
from src.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chunk_text():
    """Test text chunking functionality."""
    logger.info("Testing chunk_text...")
    text = "This is a test sentence. " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    
    assert len(chunks) > 0, "Should produce at least one chunk"
    assert all(len(chunk) <= 110 for chunk in chunks), "Chunks should respect max size"
    logger.info(f"✓ chunk_text passed: Created {len(chunks)} chunks")

def test_embeddings_model():
    """Test embeddings model initialization and text embedding."""
    logger.info("Testing EmbeddingsModel...")
    
    try:
        model = EmbeddingsModel(model_name="all-MiniLM-L6-v2")
        
        # Test single text embedding
        embedding = model.embed_text("This is a test sentence")
        assert embedding.shape[0] == model.get_embedding_dim(), "Embedding dimension should match"
        logger.info(f"✓ Single embedding shape: {embedding.shape}")
        
        # Test batch embedding
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = model.embed_texts(texts)
        assert embeddings.shape[0] == len(texts), "Should have one embedding per text"
        assert embeddings.shape[1] == model.get_embedding_dim(), "Embedding dimension should match"
        logger.info(f"✓ Batch embeddings shape: {embeddings.shape}")
        
        logger.info("✓ EmbeddingsModel passed")
    except Exception as e:
        logger.error(f"✗ EmbeddingsModel failed: {e}")
        raise

def test_vector_store():
    """Test vector store functionality."""
    logger.info("Testing VectorStore...")
    
    try:
        # Create temporary index files
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_faiss.index")
            meta_path = os.path.join(tmpdir, "test_faiss_meta.json")
            
            # Initialize vector store
            store = VectorStore(
                embedding_dim=384,
                index_path=index_path,
                metadata_path=meta_path
            )
            
            # Test adding documents
            import numpy as np
            embeddings = np.random.rand(5, 384).astype('float32')
            documents = [{"text": f"Document {i}", "source": "test"} for i in range(5)]
            
            store.add_documents(embeddings, documents)
            assert store.get_stats()["total_vectors"] == 5, "Should have 5 vectors"
            logger.info(f"✓ Added 5 documents")
            
            # Test search
            query_embedding = np.random.rand(384).astype('float32')
            distances, results = store.search(query_embedding, k=3)
            assert len(results) == 3, "Should return 3 results"
            logger.info(f"✓ Search returned {len(results)} results")
            
            # Test save/load
            store.save()
            assert os.path.exists(index_path), "Index file should be created"
            assert os.path.exists(meta_path), "Metadata file should be created"
            
            # Load in new store
            store2 = VectorStore(
                embedding_dim=384,
                index_path=index_path,
                metadata_path=meta_path
            )
            assert store2.get_stats()["total_vectors"] == 5, "Loaded store should have 5 vectors"
            logger.info(f"✓ Save/load works correctly")
        
        logger.info("✓ VectorStore passed")
    except Exception as e:
        logger.error(f"✗ VectorStore failed: {e}")
        raise

def run_all_tests():
    """Run all sanity tests."""
    logger.info("=" * 60)
    logger.info("Running sanity tests...")
    logger.info("=" * 60)
    
    tests = [
        test_chunk_text,
        test_embeddings_model,
        test_vector_store,
    ]
    
    failed = []
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            failed.append((test_func.__name__, str(e)))
            logger.error(f"Test {test_func.__name__} failed!")
    
    logger.info("=" * 60)
    if failed:
        logger.error(f"Failed tests: {len(failed)}/{len(tests)}")
        for name, error in failed:
            logger.error(f"  - {name}: {error}")
        return False
    else:
        logger.info(f"All tests passed! ({len(tests)}/{len(tests)})")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
