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

---

# Phase 7: Infinite Loop Prevention (COMPLETED)

## Bug Report
**Date**: 2026-01-15  
**Severity**: Critical  
**Symptom**: System stuck in infinite loop - Answer Grader kept rejecting, no retry limit enforced.

## Root Cause
1. **Tight Inner Loop**: `"not supported" → generate` edge bypassed retry counter
2. **No Generation Limit**: No cap on how many times `generate()` could be called

## Fixes Implemented
- Added `generation_count` to GraphState (hard limit: 5)
- Changed `"not supported"` edge to route through `transform_query` (increments retry)
- Added dual limit check in `grade_generation` function
- Added `flush=True` to all print statements for real-time debugging

---

# Phase 8: Comprehensive Logging System (COMPLETED)

## Goal
Implement a structured logging system that captures every action in the RAG pipeline for debugging, analysis, and traceability.

## Architecture Design

### Core Components
```
┌─────────────────────────────────────────────────────────────┐
│                      RAGLogger (Singleton)                   │
├─────────────────────────────────────────────────────────────┤
│  session_id: str          - Unique ID per user query        │
│  events: List[LogEvent]   - All events in session           │
│  query_history: List[str] - Track all query transformations │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         LogEvent                             │
├─────────────────────────────────────────────────────────────┤
│  session_id: str                                             │
│  event_type: EventType (enum)                                │
│  timestamp: ISO 8601 string                                  │
│  step_number: int                                            │
│  data: Dict[str, Any]                                        │
│  duration_ms: Optional[float]                                │
└─────────────────────────────────────────────────────────────┘
```

### Event Types (20+)
| Category | Events |
|----------|--------|
| **Session** | `session_start`, `session_end` |
| **Input** | `user_input`, `safety_check`, `routing_decision` |
| **Retrieval** | `retrieval_start`, `vector_search`, `bm25_search`, `hybrid_merge`, `rerank`, `retrieval_end` |
| **Documents** | `document_grading`, `document_grade_result` |
| **Query** | `query_transform` |
| **Web Search** | `web_search_start`, `web_search_strategy`, `web_search_result`, `web_search_end` |
| **Generation** | `generation_start`, `generation_end` |
| **Grading** | `hallucination_check`, `answer_check` |
| **Control** | `retry_increment`, `limit_reached` |
| **Errors** | `error`, `warning` |

### Log File Structure
```json
{
  "summary": {
    "session_id": "a1b2c3d4",
    "original_query": "summarize the most recent article",
    "final_query": "The Batch DeepLearning.AI latest article",
    "query_transformations": 2,
    "total_steps": 15,
    "total_duration_ms": 2340,
    "success": true,
    "source_type": "vectorstore",
    "source_count": 3
  },
  "query_history": [
    "summarize the most recent article",
    "The Batch DeepLearning.AI latest article summary",
    "The Batch DeepLearning.AI latest article"
  ],
  "events": [
    {
      "session_id": "a1b2c3d4",
      "event_type": "session_start",
      "timestamp": "2026-01-15T14:30:22.123456",
      "step_number": 1,
      "data": {...}
    },
    ...
  ]
}
```

## Files Created/Modified

### [NEW] `src/logger.py`
- `EventType` enum with 20+ event types
- `LogEvent` dataclass for structured events
- `RAGLogger` singleton class with methods:
  - `start_session()` / `end_session()`
  - `log_event()`, `log_query_transform()`, `log_retrieval()`
  - `log_document_grades()`, `log_web_search()`, `log_generation()`
  - `log_grading_result()`, `log_retry()`, `log_limit_reached()`
  - `log_error()`

### [NEW] `src/log_viewer.py`
- CLI tool for viewing and analyzing logs
- Commands: `--list`, `--summary`, `--verbose`, `<session_id>`

### [MODIFY] `src/graph.py`
- Import logger and time module
- Add timing and logging to all nodes:
  - `guard_input()`: Logs safety check result
  - `retrieve()`: Logs document retrieval with previews
  - `generate()`: Logs context size, response length, duration
  - `grade_documents()`: Logs relevance grades
  - `transform_query()`: Logs query transformation with history
  - `web_search_node()`: Logs search strategy and results
  - `grade_generation_v_documents_and_question()`: Logs grading decisions
  - `route_question()`: Logs routing decision

### [MODIFY] `src/rag_engine.py`
- Start logging session at query start
- End logging session with summary at query end
- Log errors with full context

## Log Output Location
```
logs/
├── 2026-01-15_14-30-22_a1b2c3d4.json
├── 2026-01-15_14-35-10_e5f6g7h8.json
└── ...
```

## Usage

### Programmatic
```python
from logger import get_logger, EventType

logger = get_logger()
session_id = logger.start_session("What is AI?")
logger.log_event(EventType.ROUTING_DECISION, {"route": "vectorstore"})
logger.end_session("AI is...", success=True)
```

### CLI Log Viewer
```powershell
# Show latest log
python src/log_viewer.py

# List all logs
python src/log_viewer.py --list

# Show summary of all sessions
python src/log_viewer.py --summary

# View specific session
python src/log_viewer.py a1b2c3d4
```

## Verification Plan

### Test 1: Log File Creation
1. Start Streamlit app
2. Submit a query
3. Check `logs/` folder for new JSON file

### Test 2: Log Content Validation
1. Open the log file
2. Verify `summary` contains all required fields
3. Verify `query_history` tracks transformations
4. Verify `events` have timestamps and step numbers

### Test 3: Log Viewer CLI
```powershell
python src/log_viewer.py --list    # Should show log files
python src/log_viewer.py           # Should display latest log
```

---

# Phase 9: Metadata-Aware Retrieval (PLANNING)

## Problem Statement
**Date**: 2026-01-15  
**Severity**: High  
**Symptom**: Query "What's the recent post?" returned Issue 231 instead of Issue 334.

### Root Cause
Vector/semantic search matches on **word similarity**, not **temporal metadata**:
- Query "recent" matched documents containing the word "recent" in content
- Issue 334 exists in DB but doesn't contain "recent" prominently
- No mechanism to prioritize by `issue_number` or `publish_date`

### Log Evidence
```json
"documents": [
  { "source": "issue-200", "content": "...recent blog post..." },  // Word match
  { "source": "issue-116", "content": "...leaked documents..." },
  { "source": "issue-231", "content": "...Mickey Mouse..." }       // Selected
]
// Issue 334 never retrieved!
```

## Proposed Solution

### Architecture: Temporal Query Detection + Metadata Boost

```
User Query: "What's the recent article?"
                │
                ▼
    ┌─────────────────────────┐
    │  Temporal Query Detector │  ← NEW
    │  (recent, latest, newest,│
    │   last, this week, etc.) │
    └─────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
    TEMPORAL         NON-TEMPORAL
        │               │
        ▼               ▼
┌───────────────┐  ┌───────────────┐
│ Metadata-First│  │ Hybrid Search │
│ Retrieval     │  │ (unchanged)   │
│               │  │               │
│ 1. Get max    │  │ Vector + BM25 │
│    issue_num  │  │ + Rerank      │
│ 2. Filter docs│  │               │
│ 3. Then rerank│  │               │
└───────────────┘  └───────────────┘
        │               │
        └───────┬───────┘
                ▼
           Final Docs
```

### Temporal Keywords (Expanded)
```python
TEMPORAL_KEYWORDS = [
    # Recency words
    "recent", "latest", "newest", "last", "current", "new", "just",
    # Time periods  
    "today", "yesterday", "this week", "last week", "this month", 
    "last month", "this year", "last year", "past week", "past month",
    # Freshness
    "fresh", "up to date", "up-to-date", "most recent",
    # Specific years (dynamic consideration)
    "2026", "2025", 
    # Months
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december"
]
```

## Proposed Changes

### [MODIFY] `src/retrieval.py`

#### 1. Add Temporal Query Detection
```python
def _is_temporal_query(self, query: str) -> bool:
    """Detect if query is asking about recency."""
    TEMPORAL_KEYWORDS = [
        # Recency words
        "recent", "latest", "newest", "last", "current", "new", "just",
        # Time periods  
        "today", "yesterday", "this week", "last week", "this month", 
        "last month", "this year", "last year", "past week", "past month",
        # Freshness
        "fresh", "up to date", "up-to-date", "most recent",
        # Specific years
        "2026", "2025", 
        # Months
        "january", "february", "march", "april", "may", "june", 
        "july", "august", "september", "october", "november", "december"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in TEMPORAL_KEYWORDS)
```

#### 2. Add Metadata-First Retrieval (Wider Pool + Semantic Rerank)
```python
def _get_docs_from_latest_issues(self, n_issues: int = 20) -> List[Document]:
    """
    Get ALL documents from the N most recent issues.
    Returns a wider pool for semantic reranking to find relevant content.
    """
    all_docs = self.bm25_docs
    
    # Filter to docs with issue_number
    docs_with_issue = [d for d in all_docs if d.metadata.get("issue_number")]
    
    if not docs_with_issue:
        return []
    
    # Find the top N unique issue numbers
    issue_numbers = set(d.metadata.get("issue_number") for d in docs_with_issue)
    top_issues = sorted(issue_numbers, reverse=True)[:n_issues]
    
    # Get ALL docs from those issues (not just one per issue)
    result = [d for d in docs_with_issue if d.metadata.get("issue_number") in top_issues]
    
    logger.info(f"Temporal retrieval: {len(result)} docs from issues {min(top_issues)}-{max(top_issues)}")
    return result
```

#### 3. Modify `retrieve()` Method (Smarter Approach)
```python
def retrieve(self, query: str, mode: str = "hybrid") -> List[Document]:
    """
    Main retrieval with temporal awareness.
    
    For temporal queries: 
    1. Get wide pool from recent issues (ensures recency)
    2. Semantic rerank within pool (ensures relevance to specific question)
    
    This handles queries like "latest article about transformers" correctly.
    """
    
    if self._is_temporal_query(query):
        logger.info("Temporal query detected - using recency-weighted retrieval")
        
        # Step 1: Get ALL docs from latest 20 issues (wide pool)
        recent_docs = self._get_docs_from_latest_issues(n_issues=20)
        
        if not recent_docs:
            # Fallback to hybrid if no issue metadata
            logger.info("No issue metadata found, falling back to hybrid search")
            candidates = self.hybrid_search(query, k=10)
            return self.rerank(query, candidates, top_n=3)
        
        # Step 2: Semantic rerank to find most relevant within recent docs
        return self.rerank(query, recent_docs, top_n=3)
    
    # Default: Hybrid search (unchanged)
    candidates = self.hybrid_search(query, k=10)
    return self.rerank(query, candidates, top_n=3)
```

### [MODIFY] `src/ingestion.py` (Add publish_date extraction)

Extract publish date from HTML for future use:
```python
def scrape_article(url):
    # ... existing code ...
    
    # NEW: Extract publish date
    publish_date = None
    date_meta = soup.find('meta', property='article:published_time')
    if date_meta:
        publish_date = date_meta.get('content')
    else:
        # Fallback: try other common meta tags
        date_meta = soup.find('meta', {'name': 'date'}) or \
                    soup.find('meta', {'name': 'publish-date'}) or \
                    soup.find('time', {'datetime': True})
        if date_meta:
            publish_date = date_meta.get('content') or date_meta.get('datetime')
    
    # Add to document metadata
    for doc in content_chunks:
        doc.metadata["publish_date"] = publish_date or ""
```

## Verification Plan

### Test 1: Temporal Detection
```python
def test_temporal_detection():
    retriever = HybridRetriever()
    # Should detect
    assert retriever._is_temporal_query("What's the recent article?") == True
    assert retriever._is_temporal_query("latest news in ML") == True
    assert retriever._is_temporal_query("what happened last week") == True
    assert retriever._is_temporal_query("news from january") == True
    assert retriever._is_temporal_query("2026 AI predictions") == True
    
    # Should NOT detect
    assert retriever._is_temporal_query("What is AI?") == False
    assert retriever._is_temporal_query("How does RAG work?") == False
```

### Test 2: Latest Issues Retrieval (Wide Pool)
```python
def test_latest_issues_pool():
    retriever = HybridRetriever()
    pool = retriever._get_docs_from_latest_issues(n_issues=5)
    
    # Should return MULTIPLE docs (all sections from top 5 issues)
    assert len(pool) > 5  # More than just 1 doc per issue
    
    # All should be from top issues
    issue_nums = set(d.metadata.get("issue_number") for d in pool)
    assert 334 in issue_nums  # Must include latest
    assert min(issue_nums) >= 330  # All from recent issues
```

### Test 3: Temporal + Topic Query
```bash
# Query: "What did the latest article say about AI agents?"
# Expected: 
#   1. Pool from issues 315-334 (recent)
#   2. Reranked to find AI agent content
#   3. Returns relevant section from Issue 334 (AI agents of 2026)
```

### Test 4: End-to-End Query
```bash
# Query: "What's the most recent article?"
# Expected: Returns content from Issue 334, NOT Issue 231
```

## User Feedback Incorporated

| Question | User Response | Action |
|----------|---------------|--------|
| Temporal keywords sufficient? | Add: "last week", "this year", months | ✅ Added 20+ keywords |
| Add `publish_date` extraction? | Yes | ✅ Will implement |
| Top 5 issues sufficient? | Not sure | ✅ Changed to wide pool (20 issues) + semantic rerank |

## Ready for Implementation

> [!NOTE]
> **User Approved**: Proceeding with implementation.
> - Expanded temporal keywords ✅
> - Wide pool (20 issues) + semantic rerank ✅  
> - publish_date extraction ✅

---

# Phase 10: Comprehensive Site Ingestion (PLANNING)

## Problem Statement
**Date**: 2026-01-15  
**Severity**: Critical  
**Symptom**: System only ingests ~334 issue pages, missing 94% of website content.

### Discovery
Website analysis via sitemap revealed:
```
sitemap-0.xml: 4,677 Batch URLs (597 issues, 4,080 articles)
sitemap-1.xml: 785 Batch URLs
TOTAL: ~5,462 URLs available
CURRENTLY INGESTED: ~334 (6%)
MISSING: ~5,128 (94%)
```

### Content Types Missing
| Type | Example URL | Count |
|------|-------------|-------|
| **Data Points** | `/the-batch/china-and-nvidia-make-a-deal/` | ~500+ |
| **Andrew's Letters** | `/the-batch/build-with-andrew/` | ~50+ |
| **Individual Articles** | `/the-batch/openai-fine-tuned-gpt-5/` | ~3,500+ |
| **Interviews** | `/the-batch/sharon-zhao-of-amd/` | ~100+ |

## Proposed Solution

### Architecture: Sitemap-Based Discovery

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Discover URLs from Sitemap                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ sitemap-0.xml + sitemap-1.xml                       │    │
│  │ → Filter: /the-batch/* only                         │    │
│  │ → Exclude: /tag/, /about/, main page                │    │
│  │ → Result: ~5,400 article URLs                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  Step 2: Classify Content Type                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ /issue-\d+/        → type: "issue"                  │    │
│  │ Contains "Data Points" → type: "data_points"        │    │
│  │ Contains "Andrew"  → type: "letter"                 │    │
│  │ Other              → type: "article"                │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  Step 3: Scrape & Index                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ scrape_article(url)                                 │    │
│  │ → Extract: title, content, publish_date, type      │    │
│  │ → Chunk by sections                                 │    │
│  │ → Index to ChromaDB                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Proposed Changes

### [NEW] `src/ingestion.py` - Sitemap Discovery

```python
def get_all_batch_urls_from_sitemap():
    """
    Discover all Batch URLs from sitemap.
    Returns list of article URLs excluding non-content pages.
    """
    import requests
    from bs4 import BeautifulSoup
    
    SITEMAP_URLS = [
        'https://www.deeplearning.ai/sitemap-0.xml',
        'https://www.deeplearning.ai/sitemap-1.xml',
    ]
    
    EXCLUDE_PATTERNS = ['/tag/', '/about/', '/subscribe']
    
    all_urls = []
    for sitemap_url in SITEMAP_URLS:
        response = requests.get(sitemap_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'xml')
        
        for url_tag in soup.find_all('url'):
            loc = url_tag.find('loc').text
            if '/the-batch/' in loc and not any(ex in loc for ex in EXCLUDE_PATTERNS):
                # Skip main page
                if loc.rstrip('/') != 'https://www.deeplearning.ai/the-batch':
                    all_urls.append(loc)
    
    return list(set(all_urls))  # Deduplicate

def classify_content_type(url):
    """Classify URL into content type."""
    if '/issue-' in url:
        return "issue"
    # Will be refined based on page content during scraping
    return "article"
```

### [MODIFY] `src/ingestion.py` - Enhanced scrape_article

```python
def scrape_article(url):
    """Scrapes article with enhanced metadata extraction."""
    # ... existing scraping code ...
    
    # NEW: Detect content type from page
    content_type = "article"
    if '/issue-' in url:
        content_type = "issue"
    elif soup.find(string=lambda t: t and "Data Points" in t):
        content_type = "data_points"
    elif soup.find(string=lambda t: t and "Andrew" in t and "Letter" in t):
        content_type = "letter"
    
    # Add to metadata
    for doc in content_chunks:
        doc.metadata["content_type"] = content_type
        doc.metadata["url"] = url
```

### [MODIFY] `src/ingestion.py` - New load_data modes

```python
def load_data(mode="sitemap", start_issue=None, end_issue=None, max_articles=None):
    """
    Load data with multiple modes:
    - "sitemap": Discover and ingest ALL content from sitemap
    - "issues": Ingest issue range (existing behavior)
    - "recent": Ingest most recent N articles
    """
    if mode == "sitemap":
        urls = get_all_batch_urls_from_sitemap()
        if max_articles:
            urls = urls[:max_articles]
        print(f"Found {len(urls)} URLs to ingest")
        return ingest_urls(urls)
    
    elif mode == "issues":
        # Existing issue range logic
        ...
    
    elif mode == "recent":
        urls = get_all_batch_urls_from_sitemap()
        # Sort by date (from sitemap lastmod) and take most recent
        ...
```

### [MODIFY] `app.py` - Enhanced UI

```python
# Sidebar ingestion options
ingestion_mode = st.radio(
    "Ingestion Mode",
    ["Recent Articles (Fast)", "Full Site (Slow)", "Issue Range", "Custom URL"]
)

if ingestion_mode == "Recent Articles (Fast)":
    num_articles = st.slider("Number of articles", 10, 100, 50)
    
elif ingestion_mode == "Full Site (Slow)":
    st.warning("⚠️ This will ingest ~5,400 articles. May take 30+ minutes.")
    
elif ingestion_mode == "Custom URL":
    custom_url = st.text_input("Article URL")
```

## Implementation Phases

### Phase 10a: Core Sitemap Discovery (Quick Win)
1. Add `get_all_batch_urls_from_sitemap()`
2. Add `ingest_urls()` helper
3. Test with 50 article limit

### Phase 10b: Enhanced Metadata
1. Add content type detection
2. Extract publish_date for all articles
3. Add lastmod from sitemap

### Phase 10c: UI Enhancement
1. Add ingestion mode selector
2. Add progress bar for bulk ingestion
3. Add "ingest single URL" option

## Verification Plan

### Test 1: Sitemap Discovery
```python
urls = get_all_batch_urls_from_sitemap()
assert len(urls) > 5000
assert any('/china-and-nvidia/' in u for u in urls)
```

### Test 2: Article Ingestion
```python
docs = scrape_article("https://www.deeplearning.ai/the-batch/china-and-nvidia-make-a-deal/")
assert len(docs) > 0
assert docs[0].metadata["content_type"] == "data_points"
```

### Test 3: Query Recent Data Points
```
Query: "What did Nvidia and China agree on?"
Expected: Answer about H200 chip imports from the Jan 12, 2026 article
```

## User Review Required

> [!IMPORTANT]
> **Please Review:**
> 1. Should we ingest ALL ~5,400 articles or limit to recent N?
> 2. Estimated time: ~2 seconds per article = ~3 hours for full site
> 3. Should we add incremental ingestion (only new articles)?
>
> **Approve to proceed with Phase 10a (quick win with 50 articles)?**
