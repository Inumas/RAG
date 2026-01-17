import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from web_search import web_search
from agents import get_router_agent, get_grading_agent, get_rewriter_agent
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def test_web_search(query):
    print(f"Testing Web Search with query: '{query}'")
    result = web_search(query)
    print("\n--- Result ---")
    print(result[:500] + "..." if len(result) > 500 else result)
    print("--------------\n")

def test_router(query):
    print(f"Testing Router with query: '{query}'")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return
    router = get_router_agent(api_key)
    result = router.invoke({"question": query})
    print(f"Route: {result.datasource}")

def test_grader():
    print("Testing Grader...")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return
    grader = get_grading_agent(api_key)
    
    # Positive case
    doc_text = "The Batch is a newsletter by DeepLearning.AI covering AI news."
    question = "Who publishes The Batch?"
    result = grader.invoke({"document": doc_text, "question": question})
    print(f"Grading (Relevant): {result.binary_score} (Expected: yes)")
    
    # Negative case
    doc_text = "The weather in Paris is sunny."
    question = "Who publishes The Batch?"
    result = grader.invoke({"document": doc_text, "question": question})
    print(f"Grading (Irrelevant): {result.binary_score} (Expected: no)")

def test_rewriter(query):
    print(f"Testing Rewriter with query: '{query}'")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return
    rewriter = get_rewriter_agent(api_key)
    result = rewriter.invoke({"question": query})
    print(f"Rewritten Query: {result.content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phase 3 Components")
    parser.add_argument("--component", choices=["search", "router", "grader", "rewriter"], required=True)
    parser.add_argument("--query", type=str, help="Query to test with", default="What is the capital of France?")
    
    args = parser.parse_args()
    
    if args.component == "search":
        test_web_search(args.query)
    elif args.component == "router":
        test_router(args.query)
    elif args.component == "grader":
        test_grader()
    elif args.component == "rewriter":
        test_rewriter(args.query)
