from graph import build_graph

def query_rag(question, api_key, retriever):
    """
    Queries the RAG system using the LangGraph Control Flow.
    """
    if not api_key:
        raise ValueError("API Key is required.")

    app = build_graph()
    
    inputs = {
        "question": question,
        "api_key": api_key,
        "retriever": retriever,
        "web_search": False # Initialize
    }
    
    # Run the graph
    # Recursion limit handles the "Execution Loop" max retries implicitly (default is usually 25)
    # config={"recursion_limit": 10}
    
    print(f"--- Invoking Graph for: {question} ---")
    final_state = app.invoke(inputs)
    
    return {
        "answer": final_state.get("generation"),
        "source_documents": final_state.get("documents"),
        # "source_type": ... (We could track this in state if needed, but inferred from docs metadata)
    }
