import os
import argparse
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from agents import get_hallucination_grader, get_answer_grader
from graph import build_graph

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class MockRetriever:
    def retrieve(self, query):
        # Return a dummy document
        return [Document(page_content="The sky is blue and the grass is green.")]

def test_hallucination_grader():
    print("\n--- Testing Hallucination Grader ---")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    grader = get_hallucination_grader(api_key)
    docs = "The sky is blue."
    
    # Grounded
    print("Testing Grounded (Expected: yes)")
    res1 = grader.invoke({"documents": docs, "generation": "The sky is blue."})
    print(f"Result: {res1.binary_score}")
    
    # Hallucinated
    print("Testing Hallucinated (Expected: no)")
    res2 = grader.invoke({"documents": docs, "generation": "The sky is red."})
    print(f"Result: {res2.binary_score}")

def test_answer_grader():
    print("\n--- Testing Answer Grader ---")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    grader = get_answer_grader(api_key)
    question = "What color is the sky?"
    
    # Addresses
    print("Testing Addresses Question (Expected: yes)")
    res1 = grader.invoke({"question": question, "generation": "The sky is blue."})
    print(f"Result: {res1.binary_score}")
    
    # Does not address
    print("Testing Does Not Address (Expected: no)")
    res2 = grader.invoke({"question": question, "generation": "I like apples."})
    print(f"Result: {res2.binary_score}")

def test_graph_execution():
    print("\n--- Testing Full Graph Execution ---")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        return

    app = build_graph()
    mock_retriever = MockRetriever()
    
    # Test 1: Simple Retrieval
    print("\nTest 1: Simple Query (should hit retrieve -> grade -> generate)")
    inputs = {
        "question": "What color is the sky?",
        "api_key": api_key,
        "retriever": mock_retriever,
        "web_search": False
    }
    res = app.invoke(inputs)
    print(f"Final Answer: {res.get('generation')}")
    
    # Test 2: Web Search Routing
    print("\nTest 2: Web Query (should hit web_search -> generate)")
    inputs = {
        "question": "What is the current weather in Tokyo?",
        "api_key": api_key,
        "retriever": mock_retriever, # Router should bypass this
        "web_search": False
    }
    res = app.invoke(inputs)
    print(f"Final Answer: {res.get('generation')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["graders", "graph"], default="graders")
    args = parser.parse_args()
    
    if args.test == "graders":
        test_hallucination_grader()
        test_answer_grader()
    elif args.test == "graph":
        test_graph_execution()
