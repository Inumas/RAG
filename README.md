# The Batch - Multimodal RAG System

## Goal
The goal of this project is to provide a **Multimodal Retrieval-Augmented Generation (RAG)** interface for searching and querying content from **"The Batch"** newsletter (by DeepLearning.AI). 

It allows users to:
- **Ingest** recent or historical newsletter issues.
- **Ask questions** in natural language about the contents.
- **Retrieve** relevant text snippets and **images** associated with the news.
- **Verify** answers with direct links and citations to the original source.

## Tech Stack

This project was built using a modern AI-native stack optimized for speed of development and quality of results.

| Component | Technology | Reason |
|-----------|------------|--------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Chosen for rapid prototyping of data/AI applications with built-in chat UI (`st.chat_message`) and state management, minimizing frontend boilerplate. |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Provides the framework for chaining LLM calls, managing retrievers, and handling prompt templates (`ChatPromptTemplate`, `RunnablePassthrough`). |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | A lightweight, open-source vector database runs locally (`PersistentClient`), making it easy to set up without external cloud dependencies for this demo. |
| **LLM** | [OpenAI GPT-4o-mini](https://openai.com/) | Selected for its balance of high reasoning capability, speed, and cost-effectiveness for RAG tasks (`gpt-4o-mini` is used). |
| **Embeddings** | OpenAI `text-embedding-3-small` | State-of-the-art embedding model for semantic search, offering better performance than older Ada models at a lower cost. |
| **Scraping** | BeautifulSoup4 | Used (implicitly via `ingestion.py`) to parse HTML content from the newsletter for the RAG knowledge base. |

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

## Architecture Flow

1.  **Ingestion Phase:**
    *   Scrape articles -> Text Splitter (Recursive 1000/200) -> Generate Embeddings -> Store in ChromaDB.
2.  **Query Phase:**
    *   User Question -> Embed Query -> Semantic Search in Chroma (Top 3) -> Retrieve Context + Metadata (Images).
3.  **Generation Phase:**
    *   Context + Question -> LLM -> Answer.
4.  **Display Phase:**
    *   Show Answer + Render Source Images and Links in UI.

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
