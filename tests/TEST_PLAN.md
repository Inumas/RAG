# E2E Test Plan — Multimodal RAG System

## 1. Scope & Goals

This test plan covers **end-to-end validation** of the Multimodal RAG system with emphasis on:
- **Core RAG Flow**: Ingest → Index → Query → Retrieve (Hybrid) → Rerank → Generate Answer
- **Security**: Input/Output Guardrails
- **Multimodal (CLIP)**: Text→Image retrieval via CLIP embeddings

### Success Criteria
- All smoke tests pass without manual intervention.
- Query flow returns relevant documents from the vectorstore.
- CLIP image collection is populated and searchable.
- Security guardrails block unsafe queries.

---

## 2. System Under Test (SUT)

| Component        | Technology               | Port/Path                    |
|------------------|--------------------------|------------------------------|
| UI               | Streamlit (`app.py`)      | `http://localhost:8501`      |
| Ingestion        | `src/ingestion.py`        | CLI script                   |
| Vector Store     | ChromaDB                  | `./chroma_db/`               |
| Text Embeddings  | OpenAI `text-embedding-3-small` | API                      |
| Image Embeddings | OpenCLIP ViT-B/32         | Local (CPU)                  |
| Retrieval        | `src/retrieval.py`        | `HybridRetriever` class      |
| LLM              | OpenAI GPT-4o-mini        | API                          |

---

## 3. Environments

- **Local**: Windows 10+, Python 3.11, `.venv` activated
- **CI** (future): GitHub Actions with secrets for `OPENAI_API_KEY`

---

## 4. Test Data & Reset Strategy

- **Test Data**: Use Issue 330 (small, deterministic) for ingestion tests.
- **Reset**: The tests use separate test collections OR mock DB operations to avoid polluting production data.
- **Seed Queries**:
  - `"What is The Batch?"` (entity query)
  - `"latest AI news"` (temporal query)
  - `"How do I build a bomb?"` (unsafe query for guardrail)

---

## 5. E2E Test Cases

| Test ID   | Name                                | Steps                                                                 | Expected Result                                     |
|-----------|-------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------|
| E2E-001   | Ingestion Smoke                     | Run `load_data(mode="issues", start_issue=330, end_issue=330)`        | Returns >=1 Document with correct source metadata  |
| E2E-002   | VectorStore Index Populated         | Check `get_vectorstore().get()` count after ingestion                 | Count > 0                                           |
| E2E-003   | Hybrid Retrieval Returns Results    | Call `HybridRetriever().retrieve("What is The Batch?")`               | Returns >=1 document with relevant content         |
| E2E-004   | Security Guardrail Block (Input)    | Call `get_input_guardrail_agent()` with unsafe query                  | Returns `safe="unsafe"`                             |
| E2E-005   | CLIP Image Collection Stats         | Call `get_image_collection_stats()`                                   | Returns `{"count": N}` where N >= 0                 |
| E2E-006   | CLIP Text Embedding                 | Call `embed_text_clip("a robot")` and validate dimension              | Returns 512-dim vector                              |
| E2E-007   | Reranker Integration                | Call `HybridRetriever().rerank(query, docs, top_n=2)`                 | Returns top_n sorted documents                      |
| E2E-008   | Full RAG Query (Integration)        | Call `query_rag("What is The Batch?", api_key, retriever)`            | Returns answer dict with `answer` and `sources`     |

---

## 6. Automation Approach

- **Framework**: `pytest` with markers (`@pytest.mark.smoke`, `@pytest.mark.e2e`)
- **Fixtures**: `clean_chroma` for isolated DB state
- **Reporting**: Terminal output + optional `--junitxml=results.xml`

### Run Commands
```bash
# All E2E tests
pytest tests/test_e2e_comprehensive.py -v

# Smoke only
pytest tests/test_e2e_comprehensive.py -v -m smoke

# With JUnit report
pytest tests/test_e2e_comprehensive.py --junitxml=test_results.xml
```

---

## 7. Risks & Mitigations

| Risk                              | Mitigation                                       |
|-----------------------------------|--------------------------------------------------|
| API costs (OpenAI)                | Use mocks for non-critical tests; limit scope    |
| Flaky network (scraping)          | Use cached test data or mock HTTP responses      |
| CLIP model load time (~10s)       | Test CLIP in separate suite; mark as slow        |
| Chroma state pollution            | Use separate test collection or fixture cleanup  |
