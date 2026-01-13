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

- [ ] **Phase 4: Control Loop & Orchestration** <!-- id: 4 -->
    - [ ] **Policy Agent**: Implement reflection loop (Unknown answer -> Re-retrieve or Web Search) <!-- id: 4.1 -->
    - [ ] Integrate all agents into a cohesive graph/chain <!-- id: 4.2 -->
    - [ ] Final answer generation with strict citations <!-- id: 4.3 -->
