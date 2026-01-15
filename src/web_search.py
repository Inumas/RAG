try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import json

# Domain context for The Batch newsletter (without strict quotes for better results)
DOMAIN_CONTEXT = 'The Batch DeepLearning.AI newsletter AI news'
SITE_RESTRICT = 'site:deeplearning.ai'

def web_search(query: str, add_domain_context: bool = True) -> str:
    """
    Performs a web search and returns the results as a string.
    Uses direct duckduckgo_search package to bypass langchain issues.
    
    Args:
        query: The search query
        add_domain_context: If True, adds "The Batch" DeepLearning.AI context to prevent
                           irrelevant results. Set to False for truly generic searches.
    """
    try:
        # Check if query already contains domain context
        query_lower = query.lower()
        has_context = any(term in query_lower for term in [
            "the batch", "deeplearning.ai", "deeplearning ai", "andrew ng"
        ])
        
        ddgs = DDGS()
        
        # Strategy 1: Try site-restricted search first (most accurate)
        if add_domain_context:
            site_query = f'{SITE_RESTRICT} {query}'
            print(f"DEBUG: querying DDGS with site restriction: '{site_query}'")
            results = list(ddgs.text(site_query, max_results=10))
            
            # Strategy 2: If no results, try with domain context keywords
            if len(results) == 0:
                if not has_context:
                    enhanced_query = f'{DOMAIN_CONTEXT} {query}'
                else:
                    enhanced_query = query
                print(f"DEBUG: site search failed, trying: '{enhanced_query}'")
                results = list(ddgs.text(enhanced_query, max_results=10))
            
            # Strategy 3: Last resort - just search the original query
            if len(results) == 0:
                print(f"DEBUG: enhanced search failed, trying original: '{query}'")
                results = list(ddgs.text(query, max_results=10))
        else:
            print(f"DEBUG: querying DDGS with: '{query}'")
            results = list(ddgs.text(query, max_results=10))
        
        print(f"DEBUG: raw results count: {len(results)}")
        if len(results) > 0:
            print(f"DEBUG: first result keys: {results[0].keys()}")
        
        # Format results nicely
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n")
        
        if not formatted_results:
            return "No web search results found. Please try a different query or check the local database."
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing web search: {e}"
