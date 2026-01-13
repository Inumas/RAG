try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import json

def web_search(query: str) -> str:
    """
    Performs a web search and returns the results as a string.
    Uses direct duckduckgo_search package to bypass langchain issues.
    """
    try:
        print(f"DEBUG: querying DDGS with '{query}'")
        ddgs = DDGS()
        # Try without max_results first to be safe, or inspect the object
        results_gen = ddgs.text(query)
        results = list(results_gen)
        print(f"DEBUG: raw results count: {len(results)}")
        if len(results) > 0:
            print(f"DEBUG: first result keys: {results[0].keys()}")
        
        # Format results nicely
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing web search: {e}"
