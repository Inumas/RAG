# The Batch - Multimodal RAG System

## Goal
The goal of this project is to provide a **Multimodal Retrieval-Augmented Generation (RAG)** interface for searching and querying content from **"The Batch"** newsletter (by DeepLearning.AI). 

It allows users to:
- **Ingest** recent or historical newsletter issues.
- **Ask questions** in natural language about the contents.
- **Retrieve** relevant text snippets and **images** associated with the news.
- **Verify** answers with direct links and citations to the original source.
- **Search the Web** automatically when the newsletter doesn't contain the answer (Agentic behavior).

## Tech Stack

This project was built using a modern AI-native stack optimized for speed of development and quality of results.

| Component | Technology | Reason |
|-----------|------------|--------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Chosen for rapid prototyping of data/AI applications with built-in chat UI (`st.chat_message`) and state management, minimizing frontend boilerplate. |
| **Orchestration** | [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) | **LangChain** handles chains/prompts. **LangGraph** manages the state machine, cyclic control flow, and self-correction loops. |
| **Agents** | Custom (`src/agents.py`) | Router, Grader (Docs), Grader (Hallucination), Grader (Answer), Rewriter. |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | A lightweight, open-source vector database runs locally (`PersistentClient`), making it easy to set up without external cloud dependencies for this demo. |
| **LLM** | [OpenAI GPT-4o-mini](https://openai.com/) | Selected for its balance of high reasoning capability, speed, and cost-effectiveness for RAG tasks (`gpt-4o-mini` is used). |
| **Embeddings** | OpenAI `text-embedding-3-small` | State-of-the-art embedding model for semantic search, offering better performance than older Ada models at a lower cost. |
| **Web Search** | [DuckDuckGo](https://duckduckgo.com/) | Privacy-focused web search integration (`ddgs`) to handle queries outside the knowledge base. |
| **Scraping** | [Unstructured.io](https://unstructured.io/) | Robust HTML partitioning to extract clean text, replacing brittle regex/BeautifulSoup logic. Includes **Topic Extraction** (via LLM) and **Date Standardization**. |

## Key Features
- **Semantic Search**: Find relevant articles by meaning, not just keywords.
- **Metadata Filtering**: Filter results by **Topic** (e.g., "Policy", "Generative AI") and **Date Range**.
- **Agentic Workflow**: Auto-corrects queries and uses web search if definitions are missing.
- **Citations**: All answers link back to the specific source article.

## Trade-offs & Design Decisions

### 1. **Streamlit vs. Full-Stack Framework (React/Next.js)**
- **Decision:** Use Streamlit.
- **Trade-off:** Streamlit is extremely fast to build (Python only), but offers less customization than React. The UI is functional and clean but follows Streamlit's rigid layout structure.
- **Benefit:** Allowed focusing entirely on the RAG logic rather than spending days on CSS/State management.

### 2. **Local Vector Store (Chroma) vs. Cloud (Pinecone/Weaviate)**
- **Decision:** Use local ChromaDB persistence.
- **Trade-off:** Data is stored on disk (`./chroma_db`). This is not suitable for horizontal scaling or distributed users but is perfect for a self-contained, single-user application or demo.
- **Benefit:** Zero infrastructure cost and setup.

### 3. **OpenAI vs. Local LLMs (Ollama/Llama 3)**
- **Decision:** Use OpenAI API.
- **Trade-off:** Requires an API key and incurs per-token costs. Data assumes privacy trust with OpenAI.
- **Benefit:** `gpt-4o-mini` provides superior reasoning and context adherence compared to most local models manageable on standard consumer hardware, ensuring high-quality answers.

### 4. **Agentic vs. Linear RAG**
- **Decision:** Implemented an Agentic Workflow with **LangGraph**.
- **Trade-off:** Complexity and slightly higher latency.
- **Benefit:** 
    - **Self-Correction**: The system can "change its mind" and rewrite queries if initial results are poor.
    *   **Hallucination Protection**: Explicit checks ensure the answer is grounded in facts.
    - **Cyclic Flow**: Unlike linear chains, the graph can loop back (`Generate` -> `Bad Grade` -> `Rewrite` -> `Search` -> `Generate`).

### 5. **Security & Robustness**
- **Decision:** Implemented **Input/Output Guardrails** and **Recursion Handling**.
- **Trade-off:** Small latency increase for safety checks.
- **Benefit:** 
    - Prevents jailbreaks (e.g., "Ignore instructions").
    - Filters unsafe content (Drugs, Violence, Porn).
    - Fails gracefully with a polite message if the system loops too many times.

## Architecture Flow (LangGraph)

The system is modeled as a State Graph:

1.  **Ingestion & Metadata**: Articles are scraped using `Unstructured`, and an LLM extracts **Topics** (e.g., "Robotics") and **Timestamps** for filtering.
2.  **Safety Check**: User Query -> **Input Guardrail** (Check `policy.yaml`).
    *   If *Unsafe*: **Refuse** immediately.
    *   If *Safe*: Proceed to Routing.
3.  **Route**: User Query -> Router Agent -> `VectorStore` OR `WebSearch`.
3.  **Retrieve (VectorStore)**: Fetch docs -> **Grade Documents Agent**.
    *   If *Relevant*: Proceed to Generate.
    *   If *Not Relevant*: **Rewrite Query** -> Loop to **Web Search**.
4.  **Generate**: Produce Answer.
5.  **Reflection (Loop)**: 
    *   **Hallucination Grader**: Is answer grounded? -> If No: **Rewrite** -> Loop.
    *   **Answer Grader**: Does it answer the question? -> If No: **Rewrite** -> Loop.
6.  **Final Output**: Verified answer displayed in UI.

## Setup (Windows)

1.  **Create and Activate Virtual Environment**:
    Open a terminal (PowerShell) in the project root and run:
    ```powershell
    # Create virtual environment (if not already created)
    python -m venv .venv
    
    # Activate virtual environment
    .\.venv\Scripts\Activate.ps1
    ```
    *Note: If you get a permission error, you may need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.*

2.  **Install dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
    *Optional: Create a `.env` file in the root with `OPENAI_API_KEY=sk-...` to auto-load your credential.*

3.  **Run the app**:
    ```powershell
    streamlit run app.py
    ```

4.  **Usage**:
    Enter your OpenAI API Key in the sidebar when the app launches in your browser.

## Logging & Debugging

The system includes comprehensive logging for debugging and analysis.

### Log Files
Every query creates a JSON log file in the `logs/` directory:
```
logs/
├── 2026-01-15_14-30-22_a1b2c3d4.json
├── 2026-01-15_14-35-10_e5f6g7h8.json
└── ...
```

### What Gets Logged
| Category | Details |
|----------|---------|
| **Session** | Start/end time, total duration, success/failure |
| **User Input** | Original query, safety check result |
| **Routing** | Decision (vectorstore vs web_search) |
| **Retrieval** | Documents retrieved, sources, relevance grades |
| **Query Transforms** | Full history of query rewrites |
| **Generation** | Context size, response length, attempt count |
| **Grading** | Hallucination check, answer relevance check |
| **Errors** | Exception type, message, context |

### Log Viewer CLI
```powershell
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Show latest log with formatted output
python src/log_viewer.py

# List all available logs
python src/log_viewer.py --list

# Show summary of all sessions
python src/log_viewer.py --summary

# View specific session by ID
python src/log_viewer.py a1b2c3d4

# Verbose mode (all events)
python src/log_viewer.py --verbose
```

### Sample Log Output
```
═══════════════════════════════════════════════════════════════════
📋 SESSION SUMMARY
───────────────────────────────────────────────────────────────────
  Session ID:       a1b2c3d4
  Status:           ✅ SUCCESS
  Duration:         2.3s
  Total Steps:      15
  Query Rewrites:   2
───────────────────────────────────────────────────────────────────
  Original Query:   summarize the most recent article from The Batch
  Final Query:      The Batch DeepLearning.AI newsletter latest...
═══════════════════════════════════════════════════════════════════
```

## Testing
To run the automated test suite (including E2E smoke tests):

1. **Install Test Dependencies** (if not done):
   ```powershell
   pip install pytest
   ```

2. **Run All Tests**:
   ```powershell
   python -m pytest tests/
   ```

3. **Run E2E Smoke Tests Only**:
   ```powershell
   python -m pytest tests/test_e2e_smoke.py -v
   ```

   *Note: Smoke tests perform real ingestion and querying. Ensure your `OPENAI_API_KEY` is set in `.env`.*

## Evaluation

The system has been formally evaluated with the following results:

### Retrieval Metrics (30 queries, k=5)
| Metric | Value |
|--------|-------|
| **Precision@5** | 0.867 |
| **MRR@5** | 0.861 |
| **Hit Rate@5** | 0.900 |
| **nDCG@5** | 0.844 |

### Running the Evaluation
```powershell
# Run retrieval evaluation
python scripts/eval_harness.py

# Auto-label results (requires OPENAI_API_KEY)
python scripts/eval_autolabel.py

# Compute metrics
python scripts/eval_metrics.py
```

### Evaluation Artifacts
- `scripts/eval_queries.json` — 30 evaluation queries
- `scripts/eval_results_labeled.json` — Labeled retrieval results
- `scripts/eval_metrics_output.json` — Computed metrics

## Multimodal (CLIP)

The system supports **true multimodal retrieval** using CLIP image embeddings.

### How It Works
- Images are automatically indexed with CLIP during article ingestion
- Text queries search both text AND image embeddings
- Documents with matching images get boosted in ranking

### Manual Image Indexing (if needed)
```powershell
# Index all existing images (one-time backfill)
python scripts/index_images.py
```

### Image Collection Stats
```powershell
# Check how many images are indexed
python -c "from src.database import get_image_collection_stats; print(get_image_collection_stats())"
```

