"""
Simple unit tests for persistent memory that don't require model downloads.
"""
import os
import sys
import tempfile
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.persistent_memory import SQLiteMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sqlite_memory_basic():
    """Test basic SQLiteMemory functionality."""
    logger.info("Testing SQLiteMemory basic operations...")
    
    try:
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        # Initialize SQLiteMemory
        memory = SQLiteMemory(db_path=db_path)
        
        # Save some messages
        memory.save_message("Student_1", "user", "What is machine learning?")
        memory.save_message("Student_1", "assistant", "Machine learning is a subset of AI...")
        memory.save_message("Student_1", "user", "Can you explain deep learning?")
        memory.save_message("Student_1", "assistant", "Deep learning uses neural networks...")
        
        # Save messages for another student
        memory.save_message("Student_2", "user", "What is Python?")
        memory.save_message("Student_2", "assistant", "Python is a programming language...")
        
        # Test retrieval for Student_1
        messages = memory.get_last_n("Student_1", n=10)
        assert len(messages) == 4, f"Expected 4 messages for Student_1, got {len(messages)}"
        assert messages[0]["role"] == "user", f"First message should be user, got {messages[0]['role']}"
        assert messages[1]["role"] == "assistant", f"Second message should be assistant, got {messages[1]['role']}"
        assert "machine learning" in messages[0]["text"].lower(), "First message should mention machine learning"
        
        # Test retrieval for Student_2
        messages = memory.get_last_n("Student_2", n=10)
        assert len(messages) == 2, f"Expected 2 messages for Student_2, got {len(messages)}"
        assert messages[0]["role"] == "user", "First message should be user"
        assert "python" in messages[0]["text"].lower(), "Message should mention Python"
        
        # Test limit
        messages = memory.get_last_n("Student_1", n=2)
        assert len(messages) == 2, f"Expected 2 messages with limit, got {len(messages)}"
        # Should get the last 2 messages (most recent)
        assert "deep learning" in messages[0]["text"].lower() or "deep learning" in messages[1]["text"].lower(), \
            "Should include recent messages about deep learning"
        
        # Test non-existent student
        messages = memory.get_last_n("NonExistent", n=10)
        assert len(messages) == 0, "Non-existent student should return empty list"
        
        # Clean up
        os.unlink(db_path)
        
        logger.info("✓ SQLiteMemory basic tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ SQLiteMemory basic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sqlite_memory_metadata():
    """Test SQLiteMemory with metadata."""
    logger.info("Testing SQLiteMemory with metadata...")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        memory = SQLiteMemory(db_path=db_path)
        
        # Save messages with metadata
        memory.save_message("Student_1", "user", "Test question", 
                          metadata={"context": "testing", "score": 0.95})
        memory.save_message("Student_1", "assistant", "Test answer",
                          metadata={"model": "test-model"})
        
        # Retrieve and check metadata
        messages = memory.get_last_n("Student_1", n=10)
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0]["metadata"].get("context") == "testing", "Metadata should be preserved"
        assert messages[1]["metadata"].get("model") == "test-model", "Metadata should be preserved"
        
        # Clean up
        os.unlink(db_path)
        
        logger.info("✓ SQLiteMemory metadata tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ SQLiteMemory metadata tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all unit tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Running Persistent Memory Unit Tests")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Test SQLiteMemory basic
    results.append(("SQLiteMemory Basic", test_sqlite_memory_basic()))
    
    # Test SQLiteMemory with metadata
    results.append(("SQLiteMemory Metadata", test_sqlite_memory_metadata()))
    
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
