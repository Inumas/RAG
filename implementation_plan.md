# Phase 4: Control Loop & Orchestration (LangGraph) Implementation Plan

## Goal
Migrate the current conditional logic into a formal **LangGraph StateGraph**. This enables true cyclic behavior (loops) for self-correction: if an answer is hallucinated, the system can loop back to re-generate or re-search without deeper nesting of function calls.

## User Review Required
> [!IMPORTANT]
> **Dependency**: This requires installing `langgraph`.
> **State Management**: The app will need to maintain a state object `GraphState` containing keys like `{"question": str, "generation": str, "documents": List[str]}`.

## Proposed Changes

### [Modify] Dependencies
#### [MODIFY] [requirements.txt](file:///c:/D/Projects/Git/RAG/requirements.txt)
- Add `langgraph`

### [New] Graph Logic
#### [NEW] [graph.py](file:///c:/D/Projects/Git/RAG/src/graph.py)
This will be the core orchestrator.
- **State Definition**: `TypedDict` with keys `question`, `generation`, `documents`.
- **Nodes**:
    - `retrieve`: Calls retriever.
    - `grade_documents`: Uses `Grader` agent to filter docs.
    - `generate`: Calls LLM to answer.
    - `web_search`: Calls `web_search` tool.
- **Edges (Conditional Logic)**:
    - `decide_to_generate`: After grading, go to `generate` OR `web_search` (if no docs).
    - `grade_generation_v_documents_and_question`:
        1. Check **Hallucination** (Groundedness).
        2. Check **Answer Quality** (Addressing question).
        3. If Bad -> Loop back to `generate` or `web_search`.

### [Modify] Agents
#### [MODIFY] [agents.py](file:///c:/D/Projects/Git/RAG/src/agents.py)
- Add **Hallucination Grader**: `(generation, docs) -> yes/no`.
- Add **Answer Grader**: `(generation, question) -> yes/no`.

### [Modify] RAG Engine
#### [MODIFY] [rag_engine.py](file:///c:/D/Projects/Git/RAG/src/rag_engine.py)
- Replace the custom `query_rag` logic with `graph.compile().invoke(inputs)`.

## Verification Plan

### Automated Tests
Create `test_phase4_graph.py`:
1. **Test Hallucination Grader**:
    - Input: {docs: "Sky is blue", generation: "Sky is green"} -> Score: **No**.
2. **Test Answer Grader**:
    - Input: {question: "1+1?", generation: "Photosynthesis is..."} -> Score: **No**.
3. **Test Full Graph**:
    - Run a query known to be difficult (requiring search) and assert the final state contains a valid answer and used "web_search" node.

### Manual Verification
- Use Streamlit UI.
- Watch console logs (I will add print statements in nodes) to see the "Thinking..." process:
    - "Retrieving..." -> "Grading..." -> "Generating..." -> "Checking Hallucination..." -> "Done".
