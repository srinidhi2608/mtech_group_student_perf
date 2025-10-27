# Persistent Per-Student Memory Feature

## Overview

The chat endpoint now supports persistent per-student memory, allowing the system to maintain conversation history and semantic memories for each student. This enables more contextual and personalized responses.

## Features

### 1. SQLiteMemory - Chat History Storage
- Stores raw chat turns (user and assistant messages) in a SQLite database
- Maintains chronological order of conversations
- Supports metadata for each message
- Per-student isolation

### 2. SemanticMemory - QA Semantic Search
- Stores question-answer pairs with embeddings
- Enables semantic search for similar previous QA exchanges
- Uses FAISS for efficient similarity search
- Per-student memory isolation

## API Changes

### ChatRequest Model
The `/chat` endpoint now accepts an optional `student_id` parameter:

```json
{
  "query": "What is machine learning?",
  "top_k": 8,
  "use_llm": true,
  "student_id": "Student_1"
}
```

**Parameters:**
- `query` (string, required): The user's question
- `top_k` (integer, optional): Number of documents to retrieve (default: 8)
- `use_llm` (boolean, optional): Whether to use LLM for answer generation (default: true)
- `student_id` (string, optional): Unique identifier for the student (default: "anonymous")

### Response Format
The response now includes additional fields when memory is available:

```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is...",
  "retrieved": [...],
  "retrieved_qa": [...],
  "chat_history": [...]
}
```

**Response Fields:**
- `query`: The original query
- `answer`: The generated answer (or null if use_llm is false)
- `retrieved`: List of retrieved document chunks from vectorstore
- `retrieved_qa` (optional): List of similar previous QA pairs for this student
- `chat_history` (optional): Recent conversation history for this student

## Backward Compatibility

The implementation maintains full backward compatibility:
- Requests without `student_id` default to "anonymous"
- If persistent memory modules cannot be imported, the endpoint works as before
- Existing response fields (`query`, `answer`, `retrieved`) remain unchanged
- New fields (`retrieved_qa`, `chat_history`) are only added when memory is available

## Usage Examples

### Example 1: First Chat (No History)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "student_id": "Student_1",
    "use_llm": true
  }'
```

Response:
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved": [...],
  "chat_history": []
}
```

### Example 2: Follow-up Chat (With History)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you explain deep learning?",
    "student_id": "Student_1",
    "use_llm": true
  }'
```

Response:
```json
{
  "query": "Can you explain deep learning?",
  "answer": "Deep learning is a subset of machine learning...",
  "retrieved": [...],
  "retrieved_qa": [
    {
      "role": "user",
      "text": "What is machine learning?",
      "score": 0.85,
      ...
    }
  ],
  "chat_history": [
    {
      "role": "user",
      "text": "What is machine learning?",
      "ts": "2025-10-27 18:00:00"
    },
    {
      "role": "assistant",
      "text": "Machine learning is...",
      "ts": "2025-10-27 18:00:01"
    }
  ]
}
```

### Example 3: Without LLM (Document Retrieval Only)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "student_id": "Student_1",
    "use_llm": false,
    "top_k": 5
  }'
```

### Example 4: Anonymous Chat (No Student ID)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "use_llm": true
  }'
```

This will use "anonymous" as the student_id internally.

## Storage Details

### SQLite Database
- **Location**: `models/chat_history.db`
- **Schema**:
  ```sql
  CREATE TABLE chat_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_id TEXT NOT NULL,
      role TEXT NOT NULL,
      text TEXT NOT NULL,
      metadata TEXT,
      ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )
  ```
- **Index**: `idx_student_id` on (student_id, id DESC)

### Semantic Memory Files
- **FAISS Index**: `models/semantic_memory.index`
- **Metadata**: `models/semantic_memory.json`
- **Vectors**: `models/semantic_memory.npy` (fallback)

## Testing

### Unit Tests
Run unit tests for the persistent memory module:
```bash
python test_persistent_memory_unit.py
```

### Smoke Test Steps

1. **Start the server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **First chat without student_id (backward compatibility)**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query":"What is machine learning?","use_llm":false}'
   ```
   Verify: Should return `retrieved` documents

3. **First chat with student_id**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query":"What is machine learning?","student_id":"Student_1","use_llm":false}'
   ```
   Verify: Should return `retrieved` documents, empty or missing `chat_history`

4. **Second chat with same student_id**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query":"Can you explain neural networks?","student_id":"Student_1","use_llm":false}'
   ```
   Verify: Should return `retrieved` documents, `chat_history` with previous conversation

5. **Third chat with related question**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query":"Tell me more about ML algorithms","student_id":"Student_1","use_llm":false}'
   ```
   Verify: Should return `retrieved_qa` with similar previous questions

6. **Chat with different student**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query":"What is Python?","student_id":"Student_2","use_llm":false}'
   ```
   Verify: Should not include Student_1's history or QA

## Implementation Notes

### Defensive Coding
- If `src.persistent_memory` cannot be imported, the endpoint falls back to the original behavior
- If embedder is not available, memory features are disabled
- All memory operations are wrapped in try-except blocks with logging
- Failures in memory operations do not crash the endpoint

### Prompt Composition
The enhanced prompt includes (when available):
1. Relevant prior student Q&A (top 3 similar QA pairs)
2. Relevant documents from vectorstore (top_k chunks)
3. Recent conversation history (last 6 messages)
4. Current user question

### Performance Considerations
- Chat history retrieval uses indexed queries (O(log n))
- Semantic search uses FAISS for efficient similarity search
- Memory operations are non-blocking and fast
- Graceful degradation if memory is unavailable

## Limitations

1. **Timestamp Precision**: SQLite timestamps have second-level precision; ordering uses auto-increment ID
2. **Memory Size**: No automatic cleanup of old messages (consider adding TTL in production)
3. **Concurrent Access**: SQLite handles concurrent reads well, but consider PostgreSQL for high concurrency
4. **Embedding Model**: Requires sentence-transformers model to be available

## Future Enhancements

1. Add conversation summarization for very long histories
2. Implement automatic memory cleanup/archival
3. Add user-triggered memory reset
4. Support multi-turn conversation context window
5. Add conversation analytics and insights
6. Implement hybrid search (keyword + semantic)

## Error Handling

All memory operations include comprehensive error handling:
- Import failures: Log warning and continue without memory
- Initialization failures: Log warning and set memory to None
- Runtime errors: Log error and return previous behavior
- Database errors: Log error and return empty results

No memory-related errors will cause the chat endpoint to fail.
