from graph import build_graph
from langgraph.errors import GraphRecursionError
from logger import get_logger
import hashlib

# Get the global logger
logger = get_logger()

def query_rag(question, api_key, retriever, filters=None):
    """
    Queries the RAG system using the LangGraph Control Flow.
    Includes error handling for recursion limits (GraphRecursionError).
    All actions are logged to a session file for traceability.
    """
    if not api_key:
        raise ValueError("API Key is required.")

    # Start logging session
    api_key_hash = hashlib.md5(api_key[:8].encode()).hexdigest()[:6]  # Safe partial hash for logging
    session_id = logger.start_session(question, metadata={"api_key_hash": api_key_hash})

    app = build_graph()
    
    inputs = {
        "question": question,
        "original_question": question,  # Keep original for reference
        "api_key": api_key,
        "retriever": retriever,
        "web_search": False,
        "safety_status": "unknown",
        "retry_count": 0,  # Initialize retry counter
        "generation_count": 0,  # Initialize generation call counter
        "temporal_retrieval": False,  # Set by retrieve() if temporal query detected
        "filters": filters  # Metadata filters
    }
    
    try:
        print(f"--- Invoking Graph for: {question} ---", flush=True)
        final_state = app.invoke(inputs)
        
        # Check safety status
        if final_state.get("safety_status") == "unsafe":
            answer = final_state.get("generation")
            logger.end_session(answer, success=True, source_type="refusal", source_count=0)
            return {
                "answer": answer,
                "source_documents": [],
                "source_type": "refusal"
            }
        
        answer = final_state.get("generation", "")
        documents = final_state.get("documents", [])
        
        # Determine source type
        source_type = "vectorstore"
        if documents and len(documents) > 0:
            if documents[0].metadata.get("source") == "web_search":
                source_type = "web_search"
        
        # End logging session
        logger.end_session(answer, success=True, source_type=source_type, source_count=len(documents))
        
        # Get the final query (may be rewritten from original)
        final_query = final_state.get("question", question)
        
        return {
            "answer": answer,
            "source_documents": documents,
            "source_type": source_type,
            "session_id": session_id,  # Include session ID for reference
            "final_query": final_query  # Rewritten query for CLIP
        }
        
    except GraphRecursionError:
        print("---GRAPH EXECUTION FAILED: RECURSION LIMIT REACHED---", flush=True)
        logger.log_error(GraphRecursionError("Recursion limit reached"), "graph_execution")
        answer = "I apologize, but I was unable to find a satisfactory answer within my processing limits. The query might be too complex or ambiguous. Please try rephrasing."
        logger.end_session(answer, success=False, source_type="failure_recursion", source_count=0)
        return {
            "answer": answer,
            "source_documents": [],
            "source_type": "failure_recursion",
            "session_id": session_id
        }
    except Exception as e:
        print(f"---GRAPH EXECUTION FAILED: {str(e)}---", flush=True)
        logger.log_error(e, "graph_execution")
        answer = f"An unexpected error occurred: {str(e)}"
        logger.end_session(answer, success=False, source_type="failure_error", source_count=0)
        return {
            "answer": answer,
            "source_documents": [],
            "source_type": "failure_error",
            "session_id": session_id
        }
