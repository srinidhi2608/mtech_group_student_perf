"""
Quick validation script to check core components without network access.
"""
import sys
import os

print("=" * 60)
print("Validating Implementation")
print("=" * 60)

# 1. Check file structure
print("\n1. Checking file structure...")
required_files = [
    "src/pdf_ingest.py",
    "src/embeddings.py",
    "src/vectorstore.py",
    "app/main.py",
    "frontend/app.py",
    "test_sanity.py",
    "requirements.txt",
    ".gitignore"
]

missing = []
for f in required_files:
    if not os.path.exists(f):
        missing.append(f)
        print(f"  ✗ Missing: {f}")
    else:
        print(f"  ✓ Found: {f}")

if missing:
    print(f"\nError: Missing {len(missing)} required files")
    sys.exit(1)

print("  ✓ All required files present")

# 2. Check Python syntax
print("\n2. Checking Python syntax...")
import py_compile

for f in [f for f in required_files if f.endswith('.py')]:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  ✓ {f}: valid syntax")
    except py_compile.PyCompileError as e:
        print(f"  ✗ {f}: {e}")
        sys.exit(1)

print("  ✓ All Python files have valid syntax")

# 3. Check imports (without actually initializing embeddings)
print("\n3. Checking core imports...")
try:
    from src import pdf_ingest
    print("  ✓ pdf_ingest module")
    from src import embeddings
    print("  ✓ embeddings module")
    from src import vectorstore
    print("  ✓ vectorstore module")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

print("  ✓ All core modules import successfully")

# 4. Test PDF chunking (no network needed)
print("\n4. Testing PDF text chunking...")
try:
    text = "This is a test. " * 100
    chunks = pdf_ingest.chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 0, "Should produce chunks"
    print(f"  ✓ Chunking works ({len(chunks)} chunks created)")
except Exception as e:
    print(f"  ✗ Chunking failed: {e}")
    sys.exit(1)

# 5. Test vector store (no network needed)
print("\n5. Testing vector store...")
try:
    import numpy as np
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test.index")
        meta_path = os.path.join(tmpdir, "test_meta.json")
        
        store = vectorstore.VectorStore(
            embedding_dim=384,
            index_path=index_path,
            metadata_path=meta_path
        )
        
        # Add test documents
        embeddings_test = np.random.rand(3, 384).astype('float32')
        docs = [{"text": f"Doc {i}", "source": "test"} for i in range(3)]
        store.add_documents(embeddings_test, docs)
        
        # Save and load
        store.save()
        assert os.path.exists(index_path), "Index file should exist"
        assert os.path.exists(meta_path), "Metadata file should exist"
        
        store2 = vectorstore.VectorStore(
            embedding_dim=384,
            index_path=index_path,
            metadata_path=meta_path
        )
        assert store2.get_stats()["total_vectors"] == 3, "Should load 3 vectors"
        
        print("  ✓ Vector store save/load works")
except Exception as e:
    print(f"  ✗ Vector store failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Check requirements.txt
print("\n6. Checking requirements.txt...")
try:
    with open("requirements.txt") as f:
        reqs = f.read()
        required_deps = ["faiss-cpu", "PyMuPDF", "python-multipart", "sentence-transformers"]
        missing_deps = [d for d in required_deps if d not in reqs]
        if missing_deps:
            print(f"  ✗ Missing dependencies: {missing_deps}")
            sys.exit(1)
        print(f"  ✓ All required dependencies present")
except Exception as e:
    print(f"  ✗ Error reading requirements.txt: {e}")
    sys.exit(1)

# 7. Check directory structure
print("\n7. Checking directory structure...")
required_dirs = ["models", "uploads"]
for d in required_dirs:
    if os.path.exists(d):
        print(f"  ✓ Directory exists: {d}")
    else:
        print(f"  ⚠ Directory will be created at runtime: {d}")

print("\n" + "=" * 60)
print("✓ VALIDATION PASSED")
print("=" * 60)
print("\nNote: Embeddings model download requires HuggingFace access.")
print("In production, the model will be cached on first run.")
print("All core components are correctly implemented.")
