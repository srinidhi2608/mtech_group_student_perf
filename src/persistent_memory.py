"""
Persistent memory module for storing and retrieving student-specific chat history and semantic memories.

Provides two main classes:
- SQLiteMemory: Stores raw chat turns (user/assistant messages) with timestamps
- SemanticMemory: Stores and retrieves semantic QA pairs using embeddings
"""
import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SQLiteMemory:
    """
    Stores chat history in SQLite database.
    Schema: student_id, role (user/assistant), text, metadata (JSON), ts (timestamp)
    """
    
    def __init__(self, db_path: str = "models/chat_history.db"):
        """
        Initialize SQLite memory storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_db()
        logger.info(f"SQLiteMemory initialized at {db_path}")
    
    def _init_db(self):
        """Create the chat_history table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata TEXT,
                        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Create index for efficient student_id queries (using id for ordering)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_student_id ON chat_history(student_id, id DESC)
                """)
                conn.commit()
                logger.info("Chat history table initialized")
        except Exception as e:
            logger.error(f"Failed to initialize chat_history table: {e}")
            raise
    
    def save_message(self, student_id: str, role: str, text: str, metadata: Optional[Dict] = None):
        """
        Save a chat message to the database.
        
        Args:
            student_id: Unique identifier for the student
            role: Either 'user' or 'assistant'
            text: Message content
            metadata: Optional metadata dictionary
        """
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_history (student_id, role, text, metadata)
                    VALUES (?, ?, ?, ?)
                """, (student_id, role, text, metadata_json))
                conn.commit()
                logger.debug(f"Saved message for {student_id}: {role}")
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
    
    def get_last_n(self, student_id: str, n: int = 12) -> List[Dict]:
        """
        Retrieve the last N messages for a student.
        
        Args:
            student_id: Unique identifier for the student
            n: Number of recent messages to retrieve
        
        Returns:
            List of message dictionaries with keys: role, text, metadata, ts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, text, metadata, ts
                    FROM chat_history
                    WHERE student_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (student_id, n))
                
                rows = cursor.fetchall()
                # Reverse to get chronological order
                messages = []
                for row in reversed(rows):
                    metadata = json.loads(row[2]) if row[2] else {}
                    messages.append({
                        "role": row[0],
                        "text": row[1],
                        "metadata": metadata,
                        "ts": row[3]
                    })
                logger.debug(f"Retrieved {len(messages)} messages for {student_id}")
                return messages
        except Exception as e:
            logger.error(f"Failed to retrieve messages: {e}")
            return []
    
    def clear_student_history(self, student_id: str):
        """Clear all chat history for a specific student."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history WHERE student_id = ?", (student_id,))
                conn.commit()
                logger.info(f"Cleared history for {student_id}")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")


class SemanticMemory:
    """
    Stores and retrieves semantic QA pairs using embeddings.
    Uses embedder and FAISS for similarity search.
    """
    
    def __init__(self, embedder, index_path: str = "models/semantic_memory.index", 
                 meta_path: str = "models/semantic_memory.json"):
        """
        Initialize semantic memory.
        
        Args:
            embedder: EmbeddingsModel instance for generating embeddings
            index_path: Path to FAISS index file
            meta_path: Path to metadata JSON file
        """
        self.embedder = embedder
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedding_dim = embedder.get_embedding_dim() if embedder else 384
        self.metadata: List[Dict] = []
        self._vectors: Optional[np.ndarray] = None
        self.index = None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
        
        # Try to import and initialize FAISS
        self._init_faiss()
        
        # Load existing data
        self._load()
        
        logger.info(f"SemanticMemory initialized (dim={self.embedding_dim})")
    
    def _init_faiss(self):
        """Initialize FAISS index if available."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("FAISS index initialized for semantic memory")
        except Exception as e:
            logger.warning(f"FAISS not available for semantic memory: {e}")
            self.index = None
    
    def _load(self):
        """Load existing semantic memory from disk."""
        try:
            # Load metadata
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.metadata)} semantic memory entries")
            
            # Load FAISS index if available
            if self.index is not None and os.path.exists(self.index_path):
                try:
                    import faiss
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded FAISS semantic memory index ({self.index.ntotal} vectors)")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
            
            # Load vectors as backup
            vectors_path = self.meta_path.replace(".json", ".npy")
            if os.path.exists(vectors_path):
                self._vectors = np.load(vectors_path)
                logger.info(f"Loaded semantic memory vectors ({self._vectors.shape[0]} vectors)")
        except Exception as e:
            logger.error(f"Error loading semantic memory: {e}")
    
    def _save(self):
        """Save semantic memory to disk."""
        try:
            # Save metadata
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            # Save FAISS index if available
            if self.index is not None and hasattr(self.index, "ntotal") and self.index.ntotal > 0:
                try:
                    import faiss
                    faiss.write_index(self.index, self.index_path)
                    logger.debug("Saved FAISS semantic memory index")
                except Exception as e:
                    logger.warning(f"Failed to save FAISS index: {e}")
            
            # Save vectors as backup
            if self._vectors is not None and self._vectors.shape[0] > 0:
                vectors_path = self.meta_path.replace(".json", ".npy")
                np.save(vectors_path, self._vectors)
                logger.debug("Saved semantic memory vectors")
        except Exception as e:
            logger.error(f"Error saving semantic memory: {e}")
    
    def add_memory(self, student_id: str, role: str, text: str, metadata: Optional[Dict] = None):
        """
        Add a QA memory entry.
        
        Args:
            student_id: Unique identifier for the student
            role: Either 'user' or 'assistant'
            text: Message content
            metadata: Optional metadata dictionary
        """
        if not self.embedder or not text.strip():
            return
        
        try:
            # Generate embedding
            embedding = self.embedder.embed_text(text).astype("float32")
            
            # Create metadata entry
            meta = {
                "student_id": student_id,
                "role": role,
                "text": text,
                "ts": datetime.now().isoformat()
            }
            if metadata:
                meta.update(metadata)
            
            # Add to FAISS index if available
            if self.index is not None:
                self.index.add(embedding.reshape(1, -1))
            
            # Add to numpy vectors for fallback
            if self._vectors is None:
                self._vectors = embedding.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, embedding.reshape(1, -1)])
            
            # Add metadata
            self.metadata.append(meta)
            
            # Save to disk
            self._save()
            
            logger.debug(f"Added semantic memory for {student_id}: {role}")
        except Exception as e:
            logger.error(f"Failed to add semantic memory: {e}")
    
    def retrieve_similar(self, student_id: str, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar QA pairs for a student.
        
        Args:
            student_id: Unique identifier for the student
            query: Query text to search for
            k: Number of results to return
        
        Returns:
            List of metadata dictionaries with 'score' added
        """
        if not self.embedder or not query.strip():
            return []
        
        # Filter metadata by student_id
        student_metadata = [(i, m) for i, m in enumerate(self.metadata) if m.get("student_id") == student_id]
        
        if not student_metadata:
            logger.debug(f"No semantic memories found for {student_id}")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query).astype("float32").reshape(1, -1)
            
            # Search using FAISS if available
            if self.index is not None and hasattr(self.index, "ntotal") and self.index.ntotal > 0:
                # Get more results than k to filter by student_id
                search_k = min(len(self.metadata), max(k * 5, 20))
                D, I = self.index.search(query_embedding, search_k)
                
                results = []
                for dist, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(self.metadata):
                        meta = self.metadata[int(idx)]
                        if meta.get("student_id") == student_id:
                            score = float(1.0 / (1.0 + float(dist)))
                            result = dict(meta)
                            result["score"] = score
                            results.append(result)
                            if len(results) >= k:
                                break
                
                logger.debug(f"Retrieved {len(results)} semantic memories for {student_id}")
                return results
            
            # Fallback: use cosine similarity on numpy vectors
            elif self._vectors is not None:
                # Get indices for this student
                student_indices = [i for i, m in student_metadata]
                student_vectors = self._vectors[student_indices]
                
                # Normalize for cosine similarity
                q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
                v_norm = student_vectors / (np.linalg.norm(student_vectors, axis=1, keepdims=True) + 1e-12)
                
                # Compute similarities
                similarities = (v_norm @ q_norm.T).ravel()
                
                # Get top k
                top_k_local = min(k, len(similarities))
                top_indices = np.argsort(similarities)[::-1][:top_k_local]
                
                results = []
                for local_idx in top_indices:
                    global_idx = student_indices[int(local_idx)]
                    meta = dict(self.metadata[global_idx])
                    meta["score"] = float(similarities[int(local_idx)])
                    results.append(meta)
                
                logger.debug(f"Retrieved {len(results)} semantic memories for {student_id} (fallback)")
                return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve semantic memories: {e}")
        
        return []
