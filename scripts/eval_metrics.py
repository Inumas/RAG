"""
RAG Evaluation Metrics Calculator

Computes retrieval metrics from labeled eval_results.json:
- Precision@K
- Recall@K (if ground truth size known)
- MRR@K (Mean Reciprocal Rank)
- Hit Rate@K
- nDCG@K (if graded relevance available)

Also analyzes existing query logs for generation quality metrics.

Usage:
    python scripts/eval_metrics.py
"""

import json
import os
import sys
from typing import List, Dict, Any
from datetime import datetime
import glob


def compute_precision_at_k(docs: List[Dict], k: int) -> float:
    """Precision@K = (# relevant docs in top-k) / k"""
    relevant = sum(1 for d in docs[:k] if d.get("relevance_label", 0) >= 0.5)
    return relevant / k if k > 0 else 0.0


def compute_mrr(docs: List[Dict]) -> float:
    """MRR = 1 / (rank of first relevant doc), 0 if none found"""
    for i, doc in enumerate(docs):
        if doc.get("relevance_label", 0) >= 0.5:
            return 1.0 / (i + 1)
    return 0.0


def compute_hit_rate(docs: List[Dict]) -> float:
    """Hit Rate = 1 if any relevant doc found, else 0"""
    for doc in docs:
        if doc.get("relevance_label", 0) >= 0.5:
            return 1.0
    return 0.0


def compute_ndcg_at_k(docs: List[Dict], k: int) -> float:
    """
    nDCG@K with graded relevance (0, 0.5, 1).
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = DCG of perfect ranking
    """
    import math
    
    # Get relevance scores
    rels = [d.get("relevance_label", 0) for d in docs[:k]]
    
    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    
    # IDCG (ideal: sort by relevance descending)
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0


def analyze_retrieval_results(results: Dict) -> Dict:
    """Compute aggregate metrics from labeled results."""
    
    metrics = {
        "by_k": {},
        "by_category": {},
        "overall": {}
    }
    
    k_values = results["metadata"]["k_values"]
    
    for k in k_values:
        k_key = f"k{k}"
        precisions = []
        mrrs = []
        hit_rates = []
        ndcgs = []
        
        for query in results["queries"]:
            if k_key not in query["retrievals"]:
                continue
                
            docs = query["retrievals"][k_key]["documents"]
            
            # Check if labeled
            if not any(d.get("relevance_label") is not None for d in docs):
                continue
            
            precisions.append(compute_precision_at_k(docs, k))
            mrrs.append(compute_mrr(docs))
            hit_rates.append(compute_hit_rate(docs))
            ndcgs.append(compute_ndcg_at_k(docs, k))
        
        if precisions:
            metrics["by_k"][k_key] = {
                "precision": round(sum(precisions) / len(precisions), 3),
                "mrr": round(sum(mrrs) / len(mrrs), 3),
                "hit_rate": round(sum(hit_rates) / len(hit_rates), 3),
                "ndcg": round(sum(ndcgs) / len(ndcgs), 3),
                "n_labeled": len(precisions)
            }
    
    # By category analysis
    categories = set(q["category"] for q in results["queries"])
    for category in categories:
        cat_queries = [q for q in results["queries"] if q["category"] == category]
        
        precisions = []
        for q in cat_queries:
            if "k5" in q["retrievals"]:
                docs = q["retrievals"]["k5"]["documents"]
                if any(d.get("relevance_label") is not None for d in docs):
                    precisions.append(compute_precision_at_k(docs, 5))
        
        if precisions:
            metrics["by_category"][category] = {
                "precision_at_5": round(sum(precisions) / len(precisions), 3),
                "n_queries": len(precisions)
            }
    
    return metrics


def analyze_existing_logs(logs_dir: str) -> Dict:
    """
    Analyze existing query logs for generation quality metrics.
    """
    log_files = glob.glob(os.path.join(logs_dir, "*.json"))
    
    stats = {
        "total_sessions": 0,
        "successful_sessions": 0,
        "hallucination_passed": 0,
        "answer_check_passed": 0,
        "query_rewrites": 0,
        "avg_duration_ms": 0,
        "avg_steps": 0,
        "by_source_type": {"vectorstore": 0, "web_search": 0}
    }
    
    durations = []
    steps = []
    
    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log = json.load(f)
            
            summary = log.get("summary", {})
            events = log.get("events", [])
            
            stats["total_sessions"] += 1
            
            if summary.get("success"):
                stats["successful_sessions"] += 1
            
            if summary.get("source_type"):
                src = summary["source_type"]
                if src in stats["by_source_type"]:
                    stats["by_source_type"][src] += 1
            
            stats["query_rewrites"] += summary.get("query_transformations", 0)
            
            if summary.get("total_duration_ms"):
                durations.append(summary["total_duration_ms"])
            if summary.get("total_steps"):
                steps.append(summary["total_steps"])
            
            # Check events for hallucination and answer checks
            for event in events:
                if event.get("event_type") == "hallucination_check":
                    if event.get("data", {}).get("passed"):
                        stats["hallucination_passed"] += 1
                elif event.get("event_type") == "answer_check":
                    if event.get("data", {}).get("passed"):
                        stats["answer_check_passed"] += 1
                        
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    if durations:
        stats["avg_duration_ms"] = round(sum(durations) / len(durations), 0)
    if steps:
        stats["avg_steps"] = round(sum(steps) / len(steps), 1)
    
    # Compute rates
    if stats["total_sessions"] > 0:
        stats["success_rate"] = round(stats["successful_sessions"] / stats["total_sessions"], 3)
        stats["hallucination_pass_rate"] = round(
            stats["hallucination_passed"] / stats["total_sessions"], 3
        )
        stats["answer_pass_rate"] = round(
            stats["answer_check_passed"] / stats["total_sessions"], 3
        )
    
    return stats


def generate_report(retrieval_metrics: Dict, log_stats: Dict, mm_stats: Dict) -> str:
    """Generate a summary report."""
    
    report = []
    report.append("=" * 60)
    report.append("RAG SYSTEM EVALUATION METRICS")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 60)
    
    report.append("\n## RETRIEVAL METRICS")
    report.append("-" * 40)
    
    if retrieval_metrics.get("by_k"):
        for k, m in retrieval_metrics["by_k"].items():
            report.append(f"\n{k.upper()}:")
            report.append(f"  Precision: {m['precision']:.3f}")
            report.append(f"  MRR:       {m['mrr']:.3f}")
            report.append(f"  Hit Rate:  {m['hit_rate']:.3f}")
            report.append(f"  nDCG:      {m['ndcg']:.3f}")
            report.append(f"  (n={m['n_labeled']} labeled queries)")
    else:
        report.append("  No labeled results found. Please label relevance in eval_results.json")
    
    report.append("\n\n## GENERATION QUALITY (from logs)")
    report.append("-" * 40)
    report.append(f"  Total Sessions:        {log_stats['total_sessions']}")
    report.append(f"  Success Rate:          {log_stats.get('success_rate', 0):.1%}")
    report.append(f"  Hallucination Pass:    {log_stats.get('hallucination_pass_rate', 0):.1%}")
    report.append(f"  Answer Relevance Pass: {log_stats.get('answer_pass_rate', 0):.1%}")
    report.append(f"  Avg Duration:          {log_stats['avg_duration_ms']:.0f}ms")
    report.append(f"  Avg Steps:             {log_stats['avg_steps']:.1f}")
    report.append(f"  Total Query Rewrites:  {log_stats['query_rewrites']}")
    
    report.append("\n\n## MULTIMODAL ANALYSIS")
    report.append("-" * 40)
    report.append(f"  Total Retrieved Docs:  {mm_stats.get('total_retrievals', 0)}")
    report.append(f"  With Image:            {mm_stats.get('with_image', 0)}")
    report.append(f"  Image Coverage Rate:   {mm_stats.get('image_rate', 0):.1%}")
    
    if mm_stats.get("by_category"):
        report.append("\n  By Category:")
        for cat, data in mm_stats["by_category"].items():
            rate = data["with_image"] / max(data["total"], 1)
            report.append(f"    {cat}: {rate:.1%} have images")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


def main():
    scripts_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(scripts_dir)
    
    # Load evaluation results (prefer labeled version)
    labeled_path = os.path.join(scripts_dir, "eval_results_labeled.json")
    unlabeled_path = os.path.join(scripts_dir, "eval_results.json")
    
    if os.path.exists(labeled_path):
        results_path = labeled_path
        print("Using labeled results: eval_results_labeled.json")
    elif os.path.exists(unlabeled_path):
        results_path = unlabeled_path
        print("Using unlabeled results: eval_results.json")
    else:
        print("No eval_results found. Run eval_harness.py first.")
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Compute retrieval metrics
    retrieval_metrics = analyze_retrieval_results(results)
    
    # Analyze existing logs
    logs_dir = os.path.join(project_dir, "logs")
    log_stats = analyze_existing_logs(logs_dir)
    
    # Get multimodal stats
    mm_stats = results.get("multimodal_analysis", {})
    
    # Generate report
    report = generate_report(retrieval_metrics, log_stats, mm_stats)
    print(report)
    
    # Save detailed metrics
    metrics_output = {
        "timestamp": datetime.now().isoformat(),
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics_from_logs": log_stats,
        "multimodal_analysis": mm_stats
    }
    
    metrics_path = os.path.join(scripts_dir, "eval_metrics_output.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
