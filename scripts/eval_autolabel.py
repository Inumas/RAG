"""
Auto-labeler for Retrieval Evaluation Results

Uses GPT-4o-mini to automatically label document relevance.
This replaces manual labeling for faster evaluation iteration.

Usage:
    python scripts/eval_autolabel.py
"""

import json
import os
import sys
import time
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class RelevanceScore(BaseModel):
    """Relevance score for a document."""
    score: float = Field(
        ..., 
        description="Relevance score: 1.0=highly relevant, 0.5=partially relevant, 0.0=not relevant"
    )
    reason: str = Field(
        ..., 
        description="Brief reason for the score"
    )


def get_relevance_grader(api_key: str):
    """Returns a chain that grades document relevance to a query."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    structured_llm = llm.with_structured_output(RelevanceScore)
    
    system = """You are a relevance grader for a retrieval system evaluation.

Given a user query and a retrieved document, assess how relevant the document is to answering the query.

Scoring rubric:
- 1.0 (Highly Relevant): Document directly addresses the query topic or contains the answer
- 0.5 (Partially Relevant): Document is tangentially related or provides useful background
- 0.0 (Not Relevant): Document is unrelated to the query

Be generous - if the document contains ANY useful information for the query, give at least 0.5.
For edge cases (off-topic queries, nonsense), if the system returns AI/ML content anyway, that's still 0.0 for the query.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Query: {query}

Document Title: {title}
Document Preview: {content}

Rate the relevance of this document to the query.""")
    ])
    
    return prompt | structured_llm


def auto_label_results(results: Dict, api_key: str, k_to_label: str = "k5") -> Dict:
    """
    Auto-label retrieval results using an LLM.
    
    Args:
        results: Evaluation results from eval_harness.py
        api_key: OpenAI API key
        k_to_label: Which k value to label (default k5)
    
    Returns:
        Results with relevance_label filled in
    """
    grader = get_relevance_grader(api_key)
    
    total_docs = 0
    labeled_docs = 0
    
    for query in results["queries"]:
        query_text = query["query"]
        category = query["category"]
        
        print(f"Labeling: {query_text[:50]}...")
        
        if k_to_label not in query["retrievals"]:
            continue
            
        for doc in query["retrievals"][k_to_label]["documents"]:
            total_docs += 1
            
            try:
                result = grader.invoke({
                    "query": query_text,
                    "title": doc.get("title", ""),
                    "content": doc.get("content_preview", "")[:500]
                })
                
                doc["relevance_label"] = result.score
                doc["relevance_reason"] = result.reason
                labeled_docs += 1
                
            except Exception as e:
                print(f"  Error labeling doc: {e}")
                doc["relevance_label"] = None
            
            # Rate limiting
            time.sleep(0.1)
    
    print(f"\nLabeled {labeled_docs}/{total_docs} documents")
    
    return results


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return
    
    scripts_dir = os.path.dirname(__file__)
    
    # Load results
    results_path = os.path.join(scripts_dir, "eval_results.json")
    if not os.path.exists(results_path):
        print("eval_results.json not found. Run eval_harness.py first.")
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print(f"Loaded {len(results['queries'])} queries for auto-labeling")
    print("=" * 60)
    
    # Auto-label k5 results
    results = auto_label_results(results, api_key, k_to_label="k5")
    
    # Save labeled results
    output_path = os.path.join(scripts_dir, "eval_results_labeled.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print(f"Labeled results saved to: {output_path}")
    print("\nNext step: Run eval_metrics.py with the labeled results")


if __name__ == "__main__":
    main()
