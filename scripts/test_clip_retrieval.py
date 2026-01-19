"""
Test CLIP Multimodal Retrieval

Runs a few test queries to verify CLIP image search is working.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval import HybridRetriever


def test_multimodal_retrieval():
    """Test that multimodal retrieval is working."""
    print("Initializing HybridRetriever with multimodal...")
    retriever = HybridRetriever(enable_multimodal=True)
    
    print(f"Multimodal enabled: {retriever.enable_multimodal}")
    
    if retriever.enable_multimodal:
        # Check image collection count
        count = retriever.image_collection.count()
        print(f"Images indexed: {count}")
    
    # Test queries
    test_queries = [
        "diagram of neural network architecture",
        "chart showing AI progress",
        "What is RAG?",
        "transformer attention mechanism",
    ]
    
    print("\n" + "=" * 60)
    print("TESTING MULTIMODAL RETRIEVAL")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Run retrieval
        docs = retriever.retrieve(query)
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "")[:60]
            image_match = doc.metadata.get("image_match", False)
            image_url = doc.metadata.get("image_url", "")
            
            match_indicator = "🖼️ IMAGE MATCH" if image_match else ""
            has_image = "📷" if image_url else ""
            
            print(f"  [{i+1}] {source}... {has_image} {match_indicator}")
    
    print("\n" + "=" * 60)
    print("Testing CLIP image search directly...")
    print("=" * 60)
    
    if retriever.enable_multimodal:
        image_results = retriever._clip_image_search("neural network diagram", k=5)
        print(f"\nCLIP found {len(image_results)} images for 'neural network diagram':")
        for img in image_results[:5]:
            print(f"  - Score: {img['score']:.3f} | {img['title'][:50]}...")


if __name__ == "__main__":
    test_multimodal_retrieval()
