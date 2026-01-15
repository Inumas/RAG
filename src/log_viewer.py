"""
RAG Log Viewer

Utility to read and display RAG pipeline logs.
Can be used from command line or imported as a module.

Usage:
    python log_viewer.py                    # Show latest log
    python log_viewer.py <session_id>       # Show specific session
    python log_viewer.py --list             # List all logs
    python log_viewer.py --summary          # Show summary of all sessions
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def get_log_files() -> List[Path]:
    """Get all log files sorted by modification time (newest first)."""
    log_path = Path(LOG_DIR)
    if not log_path.exists():
        return []
    return sorted(log_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)


def load_log(file_path: Path) -> Optional[Dict]:
    """Load a log file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def format_duration(ms: float) -> str:
    """Format duration in human-readable form."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}min"


def print_separator(char: str = "─", width: int = 80):
    """Print a separator line."""
    print(char * width)


def display_summary(log_data: Dict) -> None:
    """Display session summary."""
    summary = log_data.get("summary", {})
    
    print_separator("═")
    print(f"📋 SESSION SUMMARY")
    print_separator("─")
    
    print(f"  Session ID:       {summary.get('session_id', 'N/A')}")
    print(f"  Status:           {'✅ SUCCESS' if summary.get('success') else '❌ FAILED'}")
    print(f"  Duration:         {format_duration(summary.get('total_duration_ms', 0))}")
    print(f"  Total Steps:      {summary.get('total_steps', 0)}")
    print(f"  Query Rewrites:   {summary.get('query_transformations', 0)}")
    print(f"  Source Type:      {summary.get('source_type', 'N/A')}")
    print(f"  Source Count:     {summary.get('source_count', 0)}")
    
    print_separator("─")
    print(f"  Original Query:   {summary.get('original_query', 'N/A')}")
    if summary.get('final_query') != summary.get('original_query'):
        print(f"  Final Query:      {summary.get('final_query', 'N/A')}")
    
    print_separator("─")
    print(f"  Answer Preview:   {summary.get('answer_preview', 'N/A')[:100]}...")
    print_separator("═")


def display_query_history(log_data: Dict) -> None:
    """Display query transformation history."""
    history = log_data.get("query_history", [])
    
    if len(history) <= 1:
        print("\n📝 QUERY HISTORY: No transformations")
        return
    
    print("\n📝 QUERY TRANSFORMATION HISTORY")
    print_separator("─")
    
    for i, query in enumerate(history):
        if i == 0:
            print(f"  [Original] {query}")
        else:
            print(f"  [Rewrite {i}] {query}")
    
    print_separator("─")


def display_events(log_data: Dict, verbose: bool = False) -> None:
    """Display event timeline."""
    events = log_data.get("events", [])
    
    print("\n⏱️  EVENT TIMELINE")
    print_separator("─")
    
    # Group events for cleaner display
    important_events = [
        "session_start", "user_input", "safety_check", "routing_decision",
        "retrieval_end", "document_grading", "query_transform", 
        "web_search_end", "generation_end", "hallucination_check",
        "answer_check", "retry_increment", "limit_reached", "error", "session_end"
    ]
    
    for event in events:
        event_type = event.get("event_type", "unknown")
        step = event.get("step_number", 0)
        timestamp = event.get("timestamp", "")
        duration = event.get("duration_ms")
        data = event.get("data", {})
        
        # Skip less important events unless verbose
        if not verbose and event_type not in important_events:
            continue
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S.%f")[:-3]
            except:
                time_str = timestamp[:12]
        else:
            time_str = "N/A"
        
        # Format duration
        dur_str = f" ({format_duration(duration)})" if duration else ""
        
        # Event icon
        icons = {
            "session_start": "🚀",
            "session_end": "🏁",
            "user_input": "💬",
            "safety_check": "🛡️",
            "routing_decision": "🔀",
            "retrieval_end": "📚",
            "document_grading": "📊",
            "query_transform": "✏️",
            "web_search_end": "🌐",
            "generation_end": "🤖",
            "hallucination_check": "🔍",
            "answer_check": "✓",
            "retry_increment": "🔄",
            "limit_reached": "⚠️",
            "error": "❌"
        }
        icon = icons.get(event_type, "•")
        
        # Format output
        print(f"  {step:3d} | {time_str} | {icon} {event_type}{dur_str}")
        
        # Show key data for important events
        if event_type == "routing_decision":
            print(f"       └─ route: {data.get('route', 'N/A')}")
        elif event_type == "document_grading":
            relevant = data.get('relevant_count', 0)
            total = data.get('total_documents', 0)
            print(f"       └─ {relevant}/{total} documents relevant")
        elif event_type == "query_transform":
            print(f"       └─ '{data.get('transformed_query', 'N/A')[:50]}...'")
        elif event_type == "generation_end":
            print(f"       └─ {data.get('response_length_chars', 0)} chars")
        elif event_type == "hallucination_check" or event_type == "answer_check":
            passed = "✅" if data.get("passed") else "❌"
            print(f"       └─ {passed} {data.get('result', 'N/A')}")
        elif event_type == "limit_reached":
            print(f"       └─ {data.get('limit_type')}: {data.get('current_value')}/{data.get('max_value')}")
        elif event_type == "error":
            print(f"       └─ {data.get('error_type')}: {data.get('error_message')}")
    
    print_separator("─")


def display_full_log(log_data: Dict, verbose: bool = False) -> None:
    """Display complete log analysis."""
    display_summary(log_data)
    display_query_history(log_data)
    display_events(log_data, verbose)


def list_all_logs() -> None:
    """List all available log files."""
    files = get_log_files()
    
    if not files:
        print("No log files found.")
        return
    
    print("\n📁 AVAILABLE LOG FILES")
    print_separator("═")
    print(f"  {'Date':<12} {'Time':<10} {'Session':<10} {'File'}")
    print_separator("─")
    
    for f in files[:20]:  # Show last 20
        name = f.stem
        parts = name.split("_")
        if len(parts) >= 3:
            date = parts[0]
            time = parts[1].replace("-", ":")
            session = parts[2]
        else:
            date = time = session = "N/A"
        
        print(f"  {date:<12} {time:<10} {session:<10} {f.name}")
    
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")
    
    print_separator("═")


def show_all_summaries() -> None:
    """Show summary of all sessions."""
    files = get_log_files()
    
    if not files:
        print("No log files found.")
        return
    
    print("\n📊 SESSION SUMMARIES")
    print_separator("═")
    print(f"  {'Session':<10} {'Status':<8} {'Duration':<10} {'Steps':<6} {'Rewrites':<8} {'Query'}")
    print_separator("─")
    
    success_count = 0
    total_duration = 0
    
    for f in files[:50]:  # Analyze last 50
        log_data = load_log(f)
        if not log_data:
            continue
        
        summary = log_data.get("summary", {})
        session = summary.get("session_id", "N/A")[:8]
        success = summary.get("success", False)
        duration = summary.get("total_duration_ms", 0)
        steps = summary.get("total_steps", 0)
        rewrites = summary.get("query_transformations", 0)
        query = summary.get("original_query", "N/A")[:30]
        
        status = "✅" if success else "❌"
        
        if success:
            success_count += 1
        total_duration += duration
        
        print(f"  {session:<10} {status:<8} {format_duration(duration):<10} {steps:<6} {rewrites:<8} {query}...")
    
    print_separator("─")
    print(f"  Total: {len(files)} sessions | Success rate: {success_count}/{min(len(files), 50)} | Avg duration: {format_duration(total_duration/max(1, min(len(files), 50)))}")
    print_separator("═")


def find_log_by_session(session_id: str) -> Optional[Path]:
    """Find a log file by session ID."""
    files = get_log_files()
    for f in files:
        if session_id in f.stem:
            return f
    return None


def main():
    """Main entry point for CLI usage."""
    args = sys.argv[1:]
    
    if not args:
        # Show latest log
        files = get_log_files()
        if not files:
            print("No log files found. Run a query first to generate logs.")
            return
        
        log_data = load_log(files[0])
        if log_data:
            print(f"\n📄 Showing latest log: {files[0].name}")
            display_full_log(log_data)
    
    elif args[0] == "--list":
        list_all_logs()
    
    elif args[0] == "--summary":
        show_all_summaries()
    
    elif args[0] == "--verbose" or args[0] == "-v":
        files = get_log_files()
        if files:
            log_data = load_log(files[0])
            if log_data:
                display_full_log(log_data, verbose=True)
    
    elif args[0] == "--help" or args[0] == "-h":
        print(__doc__)
    
    else:
        # Assume it's a session ID
        log_file = find_log_by_session(args[0])
        if log_file:
            log_data = load_log(log_file)
            if log_data:
                display_full_log(log_data)
        else:
            print(f"Log file not found for session: {args[0]}")
            print("Use --list to see available logs.")


if __name__ == "__main__":
    main()
