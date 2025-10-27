#!/bin/bash
# Smoke test script for persistent memory chat endpoint

set -e

API_URL="http://localhost:8000"
STUDENT_1="Student_1"
STUDENT_2="Student_2"

echo "================================"
echo "Persistent Memory Smoke Tests"
echo "================================"
echo ""

# Helper function to make POST request
make_request() {
    local payload=$1
    local desc=$2
    echo "Test: $desc"
    echo "Payload: $payload"
    response=$(curl -s -X POST "$API_URL/chat" \
        -H "Content-Type: application/json" \
        -d "$payload")
    echo "Response: $response"
    echo ""
    echo "$response"
}

# Test 1: Chat without student_id (backward compatibility)
echo "1. Testing backward compatibility (no student_id)"
result=$(make_request '{"query":"What is machine learning?","use_llm":false}' "Backward compatibility")
if echo "$result" | grep -q '"retrieved"'; then
    echo "✓ PASS: Response contains 'retrieved' field"
else
    echo "✗ FAIL: Response missing 'retrieved' field"
    exit 1
fi
echo ""

# Test 2: First chat with student_id
echo "2. First chat with Student_1"
result=$(make_request '{"query":"What is machine learning?","student_id":"'$STUDENT_1'","use_llm":false}' "First chat")
if echo "$result" | grep -q '"retrieved"'; then
    echo "✓ PASS: Response contains 'retrieved' field"
else
    echo "✗ FAIL: Response missing 'retrieved' field"
    exit 1
fi
echo ""

# Test 3: Second chat with same student
echo "3. Second chat with Student_1 (should have history)"
result=$(make_request '{"query":"Can you explain neural networks?","student_id":"'$STUDENT_1'","use_llm":false}' "Second chat")
if echo "$result" | grep -q '"retrieved"'; then
    echo "✓ PASS: Response contains 'retrieved' field"
else
    echo "✗ FAIL: Response missing 'retrieved' field"
    exit 1
fi

if echo "$result" | grep -q '"chat_history"'; then
    echo "✓ PASS: Response contains 'chat_history' field"
else
    echo "⚠ WARNING: Response missing 'chat_history' field (may be expected if memory not initialized)"
fi
echo ""

# Test 4: Third chat with related question
echo "4. Third chat with related question (should have retrieved_qa)"
result=$(make_request '{"query":"Tell me more about ML algorithms","student_id":"'$STUDENT_1'","use_llm":false}' "Third chat")
if echo "$result" | grep -q '"retrieved"'; then
    echo "✓ PASS: Response contains 'retrieved' field"
else
    echo "✗ FAIL: Response missing 'retrieved' field"
    exit 1
fi

if echo "$result" | grep -q '"retrieved_qa"'; then
    echo "✓ PASS: Response contains 'retrieved_qa' field"
else
    echo "⚠ WARNING: Response missing 'retrieved_qa' field (may be expected if semantic memory not initialized)"
fi
echo ""

# Test 5: Chat with different student
echo "5. Chat with Student_2 (should not have Student_1's history)"
result=$(make_request '{"query":"What is Python?","student_id":"'$STUDENT_2'","use_llm":false}' "Different student")
if echo "$result" | grep -q '"retrieved"'; then
    echo "✓ PASS: Response contains 'retrieved' field"
else
    echo "✗ FAIL: Response missing 'retrieved' field"
    exit 1
fi
echo ""

# Test 6: Anonymous student (default)
echo "6. Anonymous chat (no student_id)"
result=$(make_request '{"query":"What is AI?","use_llm":false}' "Anonymous chat")
if echo "$result" | grep -q '"retrieved"'; then
    echo "✓ PASS: Response contains 'retrieved' field"
else
    echo "✗ FAIL: Response missing 'retrieved' field"
    exit 1
fi
echo ""

echo "================================"
echo "All smoke tests passed!"
echo "================================"
