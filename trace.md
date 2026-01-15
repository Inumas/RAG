# Debug Trace Log

This file documents all debugging attempts, thought processes, and fixes for reference when changes don't work as expected.

---

## Session: 2026-01-15 - Metadata-Aware Retrieval (Phase 9)

### Original Problem
**Query**: "What's the recent post in the newsletter?"  
**Expected**: Issue 334 (most recent)  
**Actual**: Issue 231 (contains word "recent" in content)

### Root Cause Analysis
Vector/semantic search matches on **word similarity**, not **temporal metadata**:
- Query "recent" matched documents containing the word "recent" in content
- Issue 334 exists in DB but doesn't contain "recent" prominently
- No mechanism to prioritize by `issue_number` or `publish_date`

---

## Attempt 1: Basic Temporal Detection

### Changes Made
**File**: `src/retrieval.py`

1. Added `TEMPORAL_KEYWORDS` list (35 keywords)
2. Added `_is_temporal_query()` method
3. Added `_get_docs_from_latest_issues()` method
4. Modified `retrieve()` to route temporal queries to metadata-first retrieval

### Result: PARTIAL SUCCESS
- Temporal detection worked
- But returned wrong issues (318, 327) via semantic reranking
- Cross-encoder matched "recent" word in content, not actual recency

### Lesson Learned
> Semantic reranking doesn't understand temporal metadata. Need separate handling for "pure recency" vs "topic + recency" queries.

---

## Attempt 2: Pure Recency Detection

### Changes Made
**File**: `src/retrieval.py`

Added `_is_pure_recency_query()` method using string replacement:

```python
def _is_pure_recency_query(self, query: str) -> bool:
    remaining = query.lower()
    for word in TEMPORAL_KEYWORDS + filler_words:
        remaining = remaining.replace(word, " ")
    remaining = " ".join(remaining.split()).strip()
    return len(remaining) < 5
```

### Result: FAILED
Test output:
```
FAIL: "What's the most recent article?" -> False (expected True)
```

### Debug Trace
```
"What's the most recent article?"
       |
       v Remove "recent"
"What's the most   article?"    <- "most" left behind!
       |
       v Remove "what" (before "what's" in list)
" 's the most   article?"       <- "'s" left behind!
       |
       v Remove "a" 
" 's   most    rticle?"         <- "a" matched inside "article"!
       |
       v Final
"'s most rticle" (len=14)       <- NOT < 5, returns False!
```

### Lesson Learned
> String replacement order matters! Shorter strings match inside longer ones.
> - "what" replaced before "what's" leaves "'s"
> - "a" matches inside "article" leaves "rticle"

---

## Attempt 3: Tokenization-Based Detection

### Changes Made
**File**: `src/retrieval.py`

Rewrote `_is_pure_recency_query()` using regex tokenization:

```python
def _is_pure_recency_query(self, query: str) -> bool:
    import re
    tokens = re.findall(r'\b\w+\b', query.lower())
    # Filter out temporal and filler words
    remaining_tokens = [t for t in tokens if t not in temporal_words and t not in filler_set]
    return len(remaining_tokens) == 0
```

### Result: PARTIAL SUCCESS
```
FAIL: "What's the most recent article?" -> False (expected True)
Remaining tokens: ['s']
```

### Debug Trace
Regex `\b\w+\b` splits "what's" into ["what", "s"] because apostrophe is not alphanumeric.

### Lesson Learned
> Contractions split on apostrophes. Need to handle fragments like 's, 't, 're, etc.

---

## Attempt 4: Add Contraction Fragments

### Changes Made
Added contraction fragments to filler_words:
```python
"s", "t", "re", "ve", "ll", "d"  # what's -> s, don't -> t, etc.
```

### Result: SUCCESS for basic queries
```
PASS: "What's the most recent article?" -> True
PASS: "Show me the latest post" -> True
```

But Streamlit app still failing...

---

## Attempt 5: Streamlit Cache Issue

### Problem
Old `HybridRetriever` was cached by `@st.cache_resource` decorator.

### Solution
Restarted Streamlit to clear cache.

### Result: PARTIAL - Still returning old issues

Log showed retrieval from old code path. Cache wasn't fully cleared.

---

## Attempt 6: End-to-End Test Failure

### Problem
Query: "What's the most recent article about? and what is the issue number?"

Log analysis (`2026-01-15_14-52-29_cb9eb85a.json`):
1. **110 seconds retrieval** - Unacceptable!
2. **Returned issues 326, 332, 331** - Not 334
3. **All docs marked irrelevant** - Triggered web search
4. **Web search returned garbage** - Wiktionary "what"

### Debug Trace
```
Query: "What's the most recent article about? and what is the issue number?"
                                        ^              ^
                                     "about"        "number"
                                        |
            These words AREN'T in filler list!
                        |
         _is_pure_recency_query() returns FALSE
                        |
         Goes to SLOW path: rerank 991 docs (110 sec!)
                        |
         Cross-encoder picks wrong docs
```

### Lesson Learned
> Filler word list was incomplete. Common question structure words like "about", "and", "number" should be filtered.

---

## Attempt 7: Comprehensive Filler Words + Speed Fix

### Changes Made
**File**: `src/retrieval.py`

1. Expanded filler_words:
```python
filler_words = [
    # Added: who, where, when, how, why, and, or, could, wondering
    # Added: number, title, topic, subject, cover, covering
    ...
]
```

2. Added speed limit for reranking:
```python
MAX_RERANK_POOL = 50  # Cross-encoder can't handle 991 docs
if len(recent_docs) > MAX_RERANK_POOL:
    # Pre-filter with BM25
    temp_bm25 = BM25Okapi(recent_corpus)
    recent_docs = temp_bm25.get_top_n(tokenized_query, recent_docs, n=MAX_RERANK_POOL)
```

3. Filter out short header chunks:
```python
top_issue_docs = [d for d in sorted_docs 
                 if d.metadata.get("issue_number") == max_issue 
                 and len(d.page_content) > 100]  # Filter short headers
```

### Result: SUCCESS
```
Query: What's the most recent article about? and what is the issue number?
  Temporal: True, Pure Recency: True

Retrieved 3 docs in 0.02 seconds  <- Was 110 seconds!
  1. Issue 334: "Happy 2026! Will this be the year we finally achieve AGI?..."
  2. Issue 334: "followed by being asked to carry out the task..."
  3. Issue 334: "that they might achieve AGI within a few quarters..."
```

---

## Key Lessons Summary

| Attempt | Issue | Lesson |
|---------|-------|--------|
| 1 | Semantic rerank found wrong docs | Separate pure recency from topic+recency |
| 2 | String replacement order | Don't use `.replace()` for word filtering |
| 3 | Apostrophe splits contractions | Handle fragments: 's, 't, 're |
| 4 | Cache persisted old code | Restart Streamlit fully |
| 5 | Filler list incomplete | Include ALL question structure words |
| 6 | 991 docs = 110 seconds | Cap reranking pool at 50 docs |
| 7 | Short header chunks | Filter by content length > 100 |

---

---

## Attempt 8: Skip Grading for Temporal Queries

### Problem
Query: "What is the latest article about?"

Log analysis:
1. Retrieval: PERFECT ✅ - Got Issue 334 in 17ms
2. Document Grading: REJECTED ALL ❌ - `relevant_count: 0`
3. Triggered web search → garbage results

### Root Cause
The Document Grader expects explicit topic match:
```
Query: "What is the article ABOUT?"
Grader looks for: "This article is about AGI"
Doc contains: "Happy 2026! Will this be the year we finally achieve AGI?"
Result: "Not relevant" (no explicit topic statement)
```

### Fix
For temporal queries, skip grading entirely - the docs ARE relevant by definition (we found the latest issue).

### Changes Made
**File**: `src/graph.py`

1. Added `temporal_retrieval: bool` to GraphState
2. Modified `retrieve()` to detect temporal queries and set flag:
```python
is_temporal = retriever._is_temporal_query(question)
return {..., "temporal_retrieval": is_temporal}
```

3. Modified `grade_documents()` to skip grading when flag is set:
```python
if state.get("temporal_retrieval", False):
    print("---TEMPORAL QUERY: SKIPPING GRADING---")
    return {"documents": documents, "web_search": False}
```

**File**: `src/rag_engine.py`
- Added `temporal_retrieval: False` to initial inputs

### Result
PENDING - Testing...

---

## Files Modified in This Session

| File | Changes |
|------|---------|
| `src/retrieval.py` | Added temporal detection, pure recency detection, metadata retrieval |
| `src/ingestion.py` | Added publish_date extraction |
| `task.md` | Added Phase 9 |
| `implementation_plan.md` | Added Phase 9 documentation |
| `Claude.md` | Added trace.md reference |

---

## Attempt 9: Restrict Web Search to Empty DB Only
**Date**: 2026-01-15
**Goal**: Make responses faster and more accurate by using local data exclusively when DB has content.

**Changes**:
1. Modified `route_question()` in `src/graph.py`:
   - Check if DB has data (`retriever.bm25_docs`) BEFORE calling router agent
   - If DB has data → ALWAYS route to vectorstore
   - Only use router agent (which might choose web_search) if DB is empty

2. Changed `transform_query` edge:
   - Was: `transform_query → web_search`
   - Now: `transform_query → retrieve` (retry with better query)

3. Updated `decide_to_generate()`:
   - If max retries reached AND no relevant docs, generate anyway (don't infinite loop)

4. Updated `grade_documents()`:
   - If no relevant docs BUT max retries reached → use original docs anyway
   - Only set `web_search_flag` if retries remaining

**Expected Result**:
- Queries always hit local vectorstore first
- No web search unless DB is completely empty
- Faster responses (no DuckDuckGo latency)
- More accurate results (no German TV websites)

### Result: SUCCESS ✅

**Log Evidence** (`2026-01-15_17-13-59_94f437c4.json`):
```json
"routing_decision": {
    "route": "vectorstore",
    "reason": "database_has_data",   ← NEW CODE!
    "doc_count": 15923,
    "duration_ms": 0.03              ← Instant routing!
}
```

- Query: "How to Test for Artificial General Intelligence"
- **NO web_search events** in entire log
- Found correct AGI article from DB
- Total time: 14.8 seconds (all local)

**Lesson**: When you have a comprehensive local database, web search adds latency and noise. Prioritize local data.

---

## How to Use This File

When a fix doesn't work:
1. Add a new "Attempt N" section
2. Document the changes made
3. Show the actual result vs expected
4. Include debug traces (logs, outputs)
5. Write down the lesson learned
6. Reference this for similar issues in future
