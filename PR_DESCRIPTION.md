# Enhanced Chat Endpoint - Persistent Per-Student Memory

## Summary of Changes

This PR enhances the `/chat` endpoint with persistent per-student memory capabilities, enabling contextual conversations while maintaining full backward compatibility.

## Files Modified

### 1. New Files Created

#### `src/persistent_memory.py` (NEW)
Complete implementation of persistent memory modules:
- **SQLiteMemory**: Stores chat history in SQLite database
  - Schema: student_id, role, text, metadata, timestamp
  - Efficient indexing for per-student retrieval
  - Fixed ordering using auto-increment ID for consistency
- **SemanticMemory**: Stores and retrieves QA pairs using embeddings
  - Uses sentence-transformers for embeddings
  - FAISS-based similarity search
  - Per-student memory isolation

#### `PERSISTENT_MEMORY.md` (NEW)
Comprehensive documentation including:
- Feature overview and architecture
- API changes and request/response formats
- Usage examples with curl commands
- Smoke test steps
- Implementation notes and limitations

#### `test_persistent_memory_unit.py` (NEW)
Unit tests for SQLiteMemory:
- Basic operations (save/retrieve)
- Metadata handling
- Per-student isolation
- Message ordering

#### `test_smoke.sh` (NEW)
Shell script for smoke testing the enhanced chat endpoint:
- Backward compatibility verification
- Memory persistence validation
- Multi-student isolation testing

### 2. Modified Files

#### `app/main.py`
**Changes:**
1. Added defensive imports for persistent_memory module (lines 20-29)
2. Initialized sqlite_memory and semantic_memory at module level (lines 79-95)
3. Updated ChatRequest model with optional `student_id` field (line 139)
4. Enhanced chat endpoint (lines 141-248):
   - Derives student_id (defaults to 'anonymous')
   - Retrieves recent chat history from SQLiteMemory
   - Retrieves student-specific QA from SemanticMemory
   - Performs existing vectorstore search unchanged
   - Composes enhanced prompt with all contexts
   - Calls existing LLM path unchanged
   - Persists new messages to SQLiteMemory
   - Adds QA pairs to SemanticMemory
   - Returns enhanced response with optional fields

**Unchanged:**
- All other endpoints (`/upload_pdf`, `/student_predict`, `/student_recommendations`)
- Core LLM invocation logic
- Existing response fields

## Key Features

### 1. Persistent Chat History
- SQLite-based storage: `models/chat_history.db`
- Chronological conversation tracking
- Metadata support for each message
- Efficient per-student queries

### 2. Semantic Memory
- Embedding-based QA storage
- Similarity search for related previous questions
- FAISS index for performance: `models/semantic_memory.index`
- Fallback to numpy-based search if FAISS unavailable

### 3. Enhanced Prompting
The chat endpoint now constructs richer prompts including:
1. Relevant prior student Q&A (top 3 similar exchanges)
2. Relevant documents from vectorstore (existing functionality)
3. Recent conversation history (last 6 messages)
4. Current user question

### 4. Backward Compatibility
- Requests without `student_id` use 'anonymous' as default
- Graceful fallback if memory modules unavailable
- Existing response fields preserved
- New fields added only when memory is available

### 5. Defensive Implementation
- Try-except wrappers around all memory operations
- Logging for debugging without crashing
- Module-level import guards
- No failures propagate to break the endpoint

## API Usage

### Request Format (NEW)
```json
{
  "query": "What is machine learning?",
  "top_k": 8,
  "use_llm": true,
  "student_id": "Student_1"
}
```

### Response Format (ENHANCED)
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is...",
  "retrieved": [...],
  "retrieved_qa": [...],
  "chat_history": [...]
}
```

## Testing

### Run Unit Tests
```bash
python test_persistent_memory_unit.py
```

### Run Smoke Tests (requires running server)
```bash
# Terminal 1: Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Run smoke tests
./test_smoke.sh
```

## Smoke Test Steps

### 1. Start the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test Backward Compatibility
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","use_llm":false}'
```
**Expected**: Response with `retrieved` field, works as before

### 3. First Chat with Student ID
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","student_id":"Student_1","use_llm":false}'
```
**Expected**: Response with `retrieved` field, empty or missing `chat_history`

### 4. Second Chat (Should Include History)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"Can you explain neural networks?","student_id":"Student_1","use_llm":false}'
```
**Expected**: Response includes `chat_history` with previous conversation

### 5. Related Question (Should Include Similar QA)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me more about ML algorithms","student_id":"Student_1","use_llm":false}'
```
**Expected**: Response includes `retrieved_qa` with semantically similar previous questions

### 6. Different Student
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Python?","student_id":"Student_2","use_llm":false}'
```
**Expected**: No Student_1 history; fresh conversation for Student_2

## Implementation Notes

### Storage Files
All persistent memory files are stored in the `models/` directory:
- `models/chat_history.db` - SQLite database for chat history
- `models/semantic_memory.index` - FAISS index for semantic search
- `models/semantic_memory.json` - Metadata for semantic memories
- `models/semantic_memory.npy` - Numpy backup of embeddings

### Error Handling
All memory operations include comprehensive error handling:
- Import failures: Log warning, continue without memory
- Initialization failures: Set memory to None, work as before
- Runtime errors: Log error, return empty results
- No memory errors crash the endpoint

### Performance
- Indexed SQLite queries for fast history retrieval
- FAISS for efficient semantic search
- Fallback to numpy-based search if FAISS unavailable
- Non-blocking memory operations

## Limitations & Future Work

### Current Limitations
1. No automatic memory cleanup (consider TTL in production)
2. SQLite timestamps have second precision (uses ID for ordering)
3. No conversation summarization for very long histories
4. Requires sentence-transformers model download

### Potential Enhancements
1. Add conversation summarization
2. Implement automatic memory archival
3. Add user-triggered memory reset
4. Support configurable context window size
5. Add conversation analytics
6. Implement hybrid search (keyword + semantic)
7. Add memory export/import functionality

## Migration Guide

### No Migration Needed!
The implementation is fully backward compatible. Existing code will work without changes.

### To Enable Memory Features
Simply include `student_id` in your chat requests:
```python
response = requests.post("http://localhost:8000/chat", json={
    "query": "Your question",
    "student_id": "unique_student_id",
    "use_llm": True
})
```

## Dependencies

No new dependencies added. The implementation uses existing dependencies:
- `sqlite3` (Python standard library)
- `sentence-transformers` (already in requirements.txt)
- `faiss-cpu` (already in requirements.txt)
- `numpy` (already in requirements.txt)

## Security & Privacy

### Considerations
1. Student IDs should be anonymized/hashed if PII
2. Chat history stored locally in SQLite
3. No automatic data retention policy (implement TTL as needed)
4. Consider encryption for sensitive conversations

### Recommendations for Production
1. Use PostgreSQL instead of SQLite for better concurrency
2. Implement data retention policies (GDPR compliance)
3. Add authentication/authorization for student_id access
4. Consider encryption at rest for chat_history.db
5. Add rate limiting per student_id

## Code Review Checklist

- [x] Backward compatibility maintained
- [x] No changes to other endpoints
- [x] Defensive error handling implemented
- [x] Unit tests added and passing
- [x] Documentation complete
- [x] Smoke test script provided
- [x] Code follows repository style
- [x] No security vulnerabilities introduced
- [x] Logging added for debugging
- [x] Memory operations are non-blocking

## Author Notes

This implementation prioritizes:
1. **Backward Compatibility**: Zero breaking changes
2. **Defensive Coding**: Graceful degradation on errors
3. **Clean Architecture**: Separate memory modules
4. **Testability**: Unit tests and smoke tests
5. **Documentation**: Comprehensive usage guide

The feature enables richer, more contextual conversations while maintaining the simplicity and reliability of the existing chat endpoint.
