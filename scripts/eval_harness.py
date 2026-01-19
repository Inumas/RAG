"""
RAG System Evaluation Harness - Retrieval-Only Focus

This script evaluates the retrieval component of the RAG system by:
1. Loading evaluation queries from eval_queries.json
2. Running retrieval for each query at k=3, k=5, k=10
3. Capturing retrieved documents with metadata
4. Outputting results for manual relevance labeling and metric computation

Usage:
    python scripts/eval_harness.py
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval import HybridRetriever
from database import get_vectorstore
from dotenv import load_dotenv

load_dotenv()


def run_retrieval_evaluation(queries: List[Dict], k_values: List[int] = [3, 5, 10]) -> Dict:
    """
    Run retrieval evaluation for all queries at specified k values.
    
    Args:
        queries: List of query dicts with 'id', 'query', 'category'
        k_values: List of k values for top-k retrieval
        
    Returns:
        Evaluation results dict
    """
    print("Initializing HybridRetriever...")
    start_init = time.time()
    retriever = HybridRetriever()
    init_time = time.time() - start_init
    print(f"Retriever initialized in {init_time:.2f}s")
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "k_values": k_values,
            "retriever_init_time_s": round(init_time, 2)
        },
        "queries": []
    }
    
    for i, q in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Evaluating: {q['query'][:50]}...")
        
        query_result = {
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "retrievals": {}
        }
        
        for k in k_values:
            start_time = time.time()
            
            try:
                # Use the retrieve method which includes temporal awareness and reranking
                docs = retriever.retrieve(q["query"], mode="hybrid")
                
                # Get more if k > 3 (retrieve returns top 3 by default)
                if k > 3:
                    # Use hybrid_search + rerank manually for larger k
                    candidates = retriever.hybrid_search(q["query"], k=k*2)
                    docs = retriever.rerank(q["query"], candidates, top_n=k)
                else:
                    docs = docs[:k]
                    
            except Exception as e:
                print(f"  Error for k={k}: {e}")
                docs = []
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract document info for labeling
            doc_results = []
            for rank, doc in enumerate(docs):
                doc_info = {
                    "rank": rank + 1,
                    "content_preview": doc.page_content[:300] if doc.page_content else "",
                    "source": doc.metadata.get("source", ""),
                    "title": doc.metadata.get("title", ""),
                    "topic": doc.metadata.get("topic", ""),
                    "issue_number": doc.metadata.get("issue_number"),
                    "has_image": bool(doc.metadata.get("image_url")),
                    "image_url": doc.metadata.get("image_url", ""),
                    # Placeholder for manual labeling
                    "relevance_label": None  # To be filled: 0=not relevant, 0.5=partial, 1=relevant
                }
                doc_results.append(doc_info)
            
            query_result["retrievals"][f"k{k}"] = {
                "documents": doc_results,
                "count": len(doc_results),
                "duration_ms": round(duration_ms, 2)
            }
        
        results["queries"].append(query_result)
        
    return results


def analyze_multimodal_coverage(results: Dict) -> Dict:
    """
    Analyze how many retrieved documents have associated images.
    This provides baseline data for multimodal impact analysis.
    """
    stats = {
        "total_retrievals": 0,
        "with_image": 0,
        "without_image": 0,
        "by_category": {}
    }
    
    for query in results["queries"]:
        category = query["category"]
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"total": 0, "with_image": 0}
        
        # Use k=5 as the reference
        if "k5" in query["retrievals"]:
            for doc in query["retrievals"]["k5"]["documents"]:
                stats["total_retrievals"] += 1
                stats["by_category"][category]["total"] += 1
                
                if doc.get("has_image"):
                    stats["with_image"] += 1
                    stats["by_category"][category]["with_image"] += 1
                else:
                    stats["without_image"] += 1
    
    stats["image_rate"] = round(stats["with_image"] / max(stats["total_retrievals"], 1), 3)
    
    return stats


def main():
    # Load queries
    queries_path = os.path.join(os.path.dirname(__file__), "eval_queries.json")
    
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found")
        return
    
    with open(queries_path, "r") as f:
        queries = json.load(f)
    
    print(f"Loaded {len(queries)} evaluation queries")
    print("=" * 60)
    
    # Run evaluation
    results = run_retrieval_evaluation(queries, k_values=[3, 5, 10])
    
    # Analyze multimodal coverage
    mm_stats = analyze_multimodal_coverage(results)
    results["multimodal_analysis"] = mm_stats
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Queries evaluated: {len(queries)}")
    print(f"  Multimodal coverage: {mm_stats['image_rate']*100:.1f}% of retrieved docs have images")
    print(f"\nNext step: Manually label relevance in eval_results.json, then run eval_metrics.py")


if __name__ == "__main__":
    main()
