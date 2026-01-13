# Phase 3: Agentic Workflow Core Implementation Plan

## Goal
Transform the linear RAG pipeline into an **Agentic Workflow** that can reason about *where* to find information (Internal Knowledge Base vs. Web) and self-correct by grading retrieved documents.

## User Review Required
> [!IMPORTANT]
> **Web Search Tool**: I will use `DuckDuckGoSearchRun` (LangChain Community) as it requires no API key for basic usage. If you prefer Tavily or Google Search, we will need to add those API keys to `.env`.

> [!WARNING]
> **Refactoring**: `rag_engine.py` currently holds the linear logic. I will refactor this into a modular `GraphState` (using LangGraph logic, even if implemented simply first) or just modular functions `src/agents/*.py`.

## Proposed Changes

### [New] Web Search Tool
#### [NEW] [web_search.py](file:///c:/D/Projects/Git/RAG/src/web_search.py)
- Implement `DuckDuckGoSearchRun` wrapper.
- Function `web_search(query: str) -> List[str]`

### [New] Agents Logic
#### [NEW] [agents.py](file:///c:/D/Projects/Git/RAG/src/agents.py)
This file will contain the specific logic for each agent node.
- **`Router`**: Class/Function using specific prompts to decide: `["vector_store", "web_search"]`.
- **`Grader`**: LLM chain that takes (question, document) and outputs JSON `{score: yes/no}`.
- **`Rewriter`**: LLM chain that takes (question) and outputs a "better" query.

### [Modify] RAG Engine
#### [MODIFY] [rag_engine.py](file:///c:/D/Projects/Git/RAG/src/rag_engine.py)
- Update `query_rag` to use the new "Conditional" flow:
    1. **Route** Query.
    2. If **Vector Store**: Retrieve -> **Grade** -> (If bad) **Rewrite** -> Web Search.
    3. If **Web Search**: Search -> **Grade**.
    4. **Generate** Final Answer.

### [Modify] Dependencies
#### [MODIFY] [requirements.txt](file:///c:/D/Projects/Git/RAG/requirements.txt)
- Add `duckduckgo-search`
- Add `langgraph` (Optional but good for future Control Loop) - *For now I will stick to conditional logic to keep it simple unless requested.*

## Verification Plan

### Automated Tests
I will create a specific test script `test_agent_flow.py` to verify each component individually.

1. **Test Router**:
   - `python test_agent_flow.py --component router --query "What is the latest context on LLMs?"` -> Should hit Web/Vector depending on phrasing.
   - `python test_agent_flow.py --component router --query "Summarize the uploaded newsletter"` -> Should hit Vector Store.

2. **Test Grader**:
   - Manually pass a "bad" document and a query to see if it says "no".

3. **Test Web Search**:
   - Verify network connectivity and result parsing.

### Manual Verification
- Run `streamlit run app.py` and ask questions that *require* external knowledge (e.g., "What is the weather today?" or "Who won the super bowl 2024?") to see if it routes to the web.
