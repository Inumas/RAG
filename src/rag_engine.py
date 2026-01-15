from graph import build_graph
from langgraph.errors import GraphRecursionError

def query_rag(question, api_key, retriever):
    """
    Queries the RAG system using the LangGraph Control Flow.
    Includes robust error handling for recursion limits (GraphRecursionError).
    """
    if not api_key:
        raise ValueError("API Key is required.")

    app = build_graph()
    
    inputs = {
        "question": question,
        "original_question": question,  # Keep original for reference
        "api_key": api_key,
        "retriever": retriever,
        "web_search": False,
        "safety_status": "unknown",
        "retry_count": 0,  # Initialize retry counter
        "generation_count": 0  # Initialize generation call counter
    }
    
    try:
        print(f"--- Invoking Graph for: {question} ---")
        final_state = app.invoke(inputs)
        
        # Check safety status
        if final_state.get("safety_status") == "unsafe":
            return {
                "answer": final_state.get("generation"),
                "source_documents": [],
                "source_type": "refusal"
            }
            
        return {
            "answer": final_state.get("generation"),
            "source_documents": final_state.get("documents"),
        }
        
    except GraphRecursionError:
        print("---GRAPH EXECUTION FAILED: RECURSION LIMIT REACHED---")
        return {
            "answer": "I apologize, but I was unable to find a satisfactory answer within my processing limits. The query might be too complex or ambiguous. Please try rephrasing.",
            "source_documents": [],
            "source_type": "failure_recursion"
        }
    except Exception as e:
        print(f"---GRAPH EXECUTION FAILED: {str(e)}---")
        return {
            "answer": f"An unexpected error occurred: {str(e)}",
            "source_documents": [],
            "source_type": "failure_error"
        }
