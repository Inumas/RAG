# Advanced RAG System Implementation Roadmap

- [x] **Phase 1: Intelligent Ingestion** <!-- id: 1 -->
    - [x] Implement metadata-aware chunking (extract page numbers, section headers, headings) <!-- id: 1.1 -->
    - [x] Update `ingestion.py` to store granular metadata <!-- id: 1.2 -->
    - [x] Verify chunks contain context headers <!-- id: 1.3 -->

- [x] **Phase 2: Multi-Stage Retrieval Funnel** <!-- id: 2 -->
    - [x] Implement Hybrid Search (Vector + Keyword/BM25) <!-- id: 2.1 -->
    - [x] Implement Re-ranking mechanism (Cross-Encoder) <!-- id: 2.2 -->
    - [x] Create retrieval supervisor to select strategy <!-- id: 2.3 -->

- [x] **Phase 3: Agentic Workflow Core** <!-- id: 3 -->
    - [x] **Prerequisite**: Setup Web Search Tool (DuckDuckGo/Tavily) <!-- id: 3.0 -->
    - [x] **Planner Agent**: Implement query decomposition and source routing (Internal vs. Web) <!-- id: 3.1 -->
    - [x] **Query Rewriter**: Implement sub-question to search query transformation <!-- id: 3.2 -->
    - [x] **Document Grader**: Implement relevance grading for retrieved chunks <!-- id: 3.4 -->
    - [x] **Distillation Agent**: Implement evidence compression/synthesis from top results <!-- id: 3.3 -->

- [x] **Phase 4: Control Loop & Orchestration (LangGraph)** <!-- id: 4 -->
    - [x] **Hallucination Grader**: Check if generation is grounded in docs <!-- id: 4.1 -->
    - [x] **Answer Grader**: Check if generation addresses the question <!-- id: 4.2 -->
    - [x] **Graph Construction**: Implement StateGraph with nodes (Retrieve, Grade, Generate, Search) <!-- id: 4.3 -->
    - [x] **Execution Loop**: Manage state limits (max retries) to prevent infinite loops <!-- id: 4.4 -->

- [x] **Phase 5: RAG Security Architecture** <!-- id: 5 -->
    - [x] **Policy Configuration**: Create `policy.yaml` for banned topics <!-- id: 5.1 -->
    - [x] **Input Guardrails**: Detect malicious/jailbreak attempts (Input Gate) <!-- id: 5.2 -->
    - [x] **Output Guardrails**: Moderate final answer (Output Gate) <!-- id: 5.3 -->
    - [x] **System Robustness**: Handle recursion limits (GraphRecursionError) gracefully <!-- id: 5.4 -->

- [x] **Phase 6: Domain Context Preservation (Bug Fixes)** <!-- id: 6 -->
    - [x] **Query Rewriter Fix**: Update prompt to preserve "The Batch" DeepLearning.AI context <!-- id: 6.1 -->
    - [x] **Web Search Enhancement**: Add domain context to search queries <!-- id: 6.2 -->
    - [x] **Empty DB Handling**: Warn user if vectorstore is empty before querying <!-- id: 6.3 -->
    - [x] **Verification**: Test with "summarize the most recent article from the batch" <!-- id: 6.4 -->

- [x] **Phase 7: Infinite Loop Prevention (Critical Bug Fix)** <!-- id: 7 -->
    - [x] **Retry Counter**: Add retry_count to GraphState to track loop iterations <!-- id: 7.1 -->
    - [x] **Web Search Fallback**: Multi-strategy search (site-restrict → domain context → raw) <!-- id: 7.2 -->
    - [x] **Answer Grader Fix**: Made grader more lenient to accept partial answers <!-- id: 7.3 -->
    - [x] **Generate Fallback**: Handle empty context gracefully with helpful message <!-- id: 7.4 -->