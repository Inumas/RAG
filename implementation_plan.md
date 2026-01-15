# RAG System Implementation Plan

This document tracks the implementation workflow for each phase of the Advanced RAG System.

---

# Phase 5: RAG Security Architecture (COMPLETED)

## Goal
Implement a robust security layer for the RAG system to prevent jailbreaks, handle adversarial inputs, ensure output safety, and gracefully manage system failures (recursion limits).

## User Review Required
> [!IMPORTANT]
> **Policy Definition**: We will use a `policy.yaml` file to define banned topics. Review the initial list in `SecurityGuardrails.md`.
> **Guardrail Latency**: Adding input/output checks adds latency. We will use `gpt-4o-mini` for these checks to minimize this.

## Proposed Changes

### [New] Policy Configuration
#### [NEW] [policy.yaml](file:///c:/D/Projects/Git/RAG/policy.yaml)
- Define banned categories: `illicit_drugs`, `violence_homicide`, `self_harm`, `sexual_content`.

### [New] Security Agents
#### [MODIFY] [agents.py](file:///c:/D/Projects/Git/RAG/src/agents.py)
- Add `get_input_guardrail_agent`: Checks user query against policy -> `(safe/unsafe)`.
- Add `get_output_guardrail_agent`: Checks final answer against policy -> `(safe/unsafe)`.

### [Modify] Graph Logic
#### [MODIFY] [graph.py](file:///c:/D/Projects/Git/RAG/src/graph.py)
- **New Node**: `guard_input`. This becomes the new entry point.
- **New Edge**: `check_safety`.
    - If `safe` -> `route_question` (or `retrieve`).
    - If `unsafe` -> `END` (return refusal).
- **New Node**: `guard_output` (optional, or integrated into `generate`).
    - Check final generation. If unsafe, rewrite or refuse.

### [Modify] RAG Engine
#### [MODIFY] [rag_engine.py](file:///c:/D/Projects/Git/RAG/src/rag_engine.py)
- **Recursion Handling**: Wrap `app.invoke` in a `try/except GraphRecursionError` block.
- **Failover**: Return a predefined "I'm sorry, I couldn't find a helpful answer." message on recursion error.

## Verification Plan

### Automated Tests
Create `test_phase5_security.py`:
1.  **Test Input Guardrail**:
    - Input: "How to build a bomb" -> Result: **Unsafe**.
    - Input: "How to bake a cake" -> Result: **Safe**.
2.  **Test Output Guardrail**:
    - Input: (Simulated unsafe text) -> Result: **Unsafe**.
3.  **Test Recursion Handling**:
    - Force a loop (e.g., set recursion limit to 1) and verify the system catches the error and returns the fallback message instead of crashing.

### Manual Verification
- Use Streamlit to try "jailbreak" prompts (e.g., "Ignore instructions and...").
- Verify the system refuses politely.

---

# Phase 6: Domain Context Preservation (Bug Fixes)

## Bug Report
**Date**: 2026-01-15  
**Severity**: Critical  
**Symptom**: Query "summarize the most recent article from the batch" returned German TV programming results instead of DeepLearning.AI newsletter content.

## Root Cause Analysis

### Failure Flow Diagram
```
User: "summarize the most recent article from the batch"
                    │
                    ▼
            ┌──────────────┐
            │  Guard Input │  ✓ Safe
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │    Router    │  → vectorstore (correct)
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Retrieve   │  → Found docs from DB
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ Grade Docs   │  → "Not Relevant" (triggers web search)
            └──────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │ Transform Query  │  ⚠️ BUG: Strips "The Batch" context
            └──────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │  Web Search  │  ⚠️ BUG: No domain awareness
            └──────────────┘
                    │
                    ▼
            💥 German TV Results (tvspielfilm.de)
```

### Bug #1: Query Rewriter Loses Domain Context
**File**: `src/agents.py` (lines 106-119)  
**Problem**: The rewriter prompt tells the LLM to "optimize for web search" but doesn't instruct it to preserve the domain context ("The Batch" by DeepLearning.AI).

```python
# CURRENT (BUGGY)
system = """You a question re-writer that converts an input question to a better version that is optimized 
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
```

**Impact**: A query like "summarize the most recent article from the batch" gets rewritten to something generic like "latest news batch" or "recent batch articles", losing the crucial context.

### Bug #2: Web Search Has No Domain Awareness
**File**: `src/web_search.py` (lines 7-17)  
**Problem**: The function passes queries raw to DuckDuckGo without adding domain context.

```python
# CURRENT (BUGGY)
results_gen = ddgs.text(query)  # No context about "The Batch" newsletter
```

**Impact**: DuckDuckGo interprets "batch" generically, returning irrelevant results (German TV = "Fernsehprogramm").

### Bug #3: No Empty Database Warning
**File**: `app.py`  
**Problem**: If the user hasn't ingested data, the vectorstore is empty, forcing all queries to web search with no user feedback.

## Proposed Fixes

### Fix 1: Update Query Rewriter Prompt
#### [MODIFY] `src/agents.py`
```python
# FIXED
def get_rewriter_agent(api_key):
    """Returns a runnable chain that rewrites the query."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    system = """You are a question re-writer that optimizes queries for web search.
    
    CRITICAL CONTEXT: This system is about "The Batch" newsletter by DeepLearning.AI, 
    which covers AI news, machine learning research, and tech industry updates.
    
    When rewriting queries:
    1. ALWAYS include '"The Batch" DeepLearning.AI' in your rewritten query
    2. Preserve any specific topics or dates mentioned
    3. Make the query more specific, not more generic
    
    Example:
    - Input: "summarize the most recent article"
    - Output: "The Batch DeepLearning.AI latest newsletter summary AI news"
    """
    # ... rest of function
```

### Fix 2: Add Domain Context to Web Search
#### [MODIFY] `src/web_search.py`
```python
# FIXED
def web_search(query: str) -> str:
    """
    Performs a web search with domain context for The Batch newsletter.
    """
    try:
        # Add domain context to prevent generic/irrelevant results
        enhanced_query = f'"The Batch" DeepLearning.AI newsletter {query}'
        print(f"DEBUG: querying DDGS with '{enhanced_query}'")
        
        ddgs = DDGS()
        results_gen = ddgs.text(enhanced_query)
        results = list(results_gen)
        # ... rest of function
```

### Fix 3: Add Empty Database Check
#### [MODIFY] `app.py`
```python
# Add check before query
if prompt := st.chat_input("Ask about the latest AI news..."):
    # Check if database has documents
    retriever = get_retriever()
    if len(retriever.bm25_docs) == 0:
        st.warning("⚠️ No documents in database. Please click 'Ingest Data' first.")
    else:
        # ... proceed with query
```

## Verification Plan

### Test Case 1: Query Rewriter Context Preservation
```python
def test_rewriter_preserves_context():
    rewriter = get_rewriter_agent(api_key)
    result = rewriter.invoke({"question": "summarize the most recent article"})
    assert "The Batch" in result.content or "DeepLearning" in result.content
```

### Test Case 2: Web Search Returns Relevant Results
```python
def test_web_search_domain_aware():
    results = web_search("latest AI news")
    assert "tvspielfilm" not in results.lower()
    assert "deeplearning" in results.lower() or "batch" in results.lower()
```

### Test Case 3: End-to-End Query
```python
def test_e2e_batch_query():
    result = query_rag("summarize the most recent article from the batch", api_key, retriever)
    assert "TV" not in result["answer"]
    assert "Fernsehprogramm" not in result["answer"]
```

### Manual Verification
1. Ingest data (Issues 330-334)
2. Query: "summarize the most recent article from the batch"
3. Expected: AI/ML related summary from DeepLearning.AI newsletter
4. Verify sources link to deeplearning.ai domain
