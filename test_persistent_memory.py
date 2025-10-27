"""
Test script for persistent memory functionality.
"""
import os
import sys
import tempfile
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.persistent_memory import SQLiteMemory, SemanticMemory
from src.embeddings import get_embeddings_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sqlite_memory():
    """Test SQLiteMemory functionality."""
    logger.info("=" * 60)
    logger.info("Testing SQLiteMemory...")
    logger.info("=" * 60)
    
    try:
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        # Initialize SQLiteMemory
        memory = SQLiteMemory(db_path=db_path)
        logger.info("✓ SQLiteMemory initialized")
        
        # Save some messages
        memory.save_message("Student_1", "user", "What is machine learning?")
        memory.save_message("Student_1", "assistant", "Machine learning is a subset of AI...")
        memory.save_message("Student_1", "user", "Can you explain deep learning?")
        memory.save_message("Student_1", "assistant", "Deep learning uses neural networks...")
        logger.info("✓ Saved 4 messages for Student_1")
        
        # Save messages for another student
        memory.save_message("Student_2", "user", "What is Python?")
        memory.save_message("Student_2", "assistant", "Python is a programming language...")
        logger.info("✓ Saved 2 messages for Student_2")
        
        # Retrieve messages for Student_1
        messages = memory.get_last_n("Student_1", n=10)
        assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"
        assert messages[0]["role"] == "user", "First message should be from user"
        assert messages[1]["role"] == "assistant", "Second message should be from assistant"
        logger.info(f"✓ Retrieved {len(messages)} messages for Student_1")
        
        # Retrieve messages for Student_2
        messages = memory.get_last_n("Student_2", n=10)
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        logger.info(f"✓ Retrieved {len(messages)} messages for Student_2")
        
        # Test limit
        messages = memory.get_last_n("Student_1", n=2)
        assert len(messages) == 2, f"Expected 2 messages with limit, got {len(messages)}"
        logger.info("✓ Message limit works correctly")
        
        # Clean up
        os.unlink(db_path)
        logger.info("✓ SQLiteMemory tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ SQLiteMemory tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semantic_memory():
    """Test SemanticMemory functionality."""
    logger.info("=" * 60)
    logger.info("Testing SemanticMemory...")
    logger.info("=" * 60)
    
    try:
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_semantic.index")
            meta_path = os.path.join(tmpdir, "test_semantic.json")
            
            # Initialize embedder
            logger.info("Loading embeddings model...")
            embedder = get_embeddings_model("all-MiniLM-L6-v2")
            logger.info("✓ Embeddings model loaded")
            
            # Initialize SemanticMemory
            memory = SemanticMemory(
                embedder=embedder,
                index_path=index_path,
                meta_path=meta_path
            )
            logger.info("✓ SemanticMemory initialized")
            
            # Add some memories
            memory.add_memory("Student_1", "user", "What is machine learning?")
            memory.add_memory("Student_1", "assistant", "Machine learning is a subset of artificial intelligence...")
            memory.add_memory("Student_1", "user", "How does deep learning work?")
            memory.add_memory("Student_1", "assistant", "Deep learning uses neural networks with multiple layers...")
            logger.info("✓ Added 4 memories for Student_1")
            
            # Add memories for another student
            memory.add_memory("Student_2", "user", "What is Python programming?")
            memory.add_memory("Student_2", "assistant", "Python is a high-level programming language...")
            logger.info("✓ Added 2 memories for Student_2")
            
            # Retrieve similar memories for Student_1
            results = memory.retrieve_similar("Student_1", "explain neural networks", k=2)
            assert len(results) > 0, "Should retrieve at least one result"
            assert results[0]["student_id"] == "Student_1", "Results should be for Student_1"
            logger.info(f"✓ Retrieved {len(results)} similar memories for Student_1")
            for i, r in enumerate(results, 1):
                logger.info(f"  {i}. {r['role']}: {r['text'][:60]}... (score: {r.get('score', 0):.3f})")
            
            # Retrieve for Student_2
            results = memory.retrieve_similar("Student_2", "programming languages", k=2)
            assert len(results) > 0, "Should retrieve at least one result"
            assert results[0]["student_id"] == "Student_2", "Results should be for Student_2"
            logger.info(f"✓ Retrieved {len(results)} similar memories for Student_2")
            
            # Test persistence
            logger.info("Testing persistence...")
            memory2 = SemanticMemory(
                embedder=embedder,
                index_path=index_path,
                meta_path=meta_path
            )
            assert len(memory2.metadata) == 6, f"Expected 6 metadata entries after reload, got {len(memory2.metadata)}"
            logger.info("✓ Persistence works correctly")
            
            logger.info("✓ SemanticMemory tests passed")
            return True
            
    except Exception as e:
        logger.error(f"✗ SemanticMemory tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all persistent memory tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Running Persistent Memory Tests")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Test SQLiteMemory
    results.append(("SQLiteMemory", test_sqlite_memory()))
    
    # Test SemanticMemory
    results.append(("SemanticMemory", test_semantic_memory()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
