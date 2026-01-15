"""
RAG Pipeline Logger

Comprehensive logging system for tracing all RAG pipeline actions.
Logs are written as JSON for easy parsing and analysis.

Each session gets a unique ID and all events are timestamped.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading

# Configure file logging
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class EventType(str, Enum):
    """Types of events in the RAG pipeline."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    
    # Input Processing
    USER_INPUT = "user_input"
    SAFETY_CHECK = "safety_check"
    ROUTING_DECISION = "routing_decision"
    
    # Retrieval
    RETRIEVAL_START = "retrieval_start"
    VECTOR_SEARCH = "vector_search"
    BM25_SEARCH = "bm25_search"
    HYBRID_MERGE = "hybrid_merge"
    RERANK = "rerank"
    RETRIEVAL_END = "retrieval_end"
    
    # Document Processing
    DOCUMENT_GRADING = "document_grading"
    DOCUMENT_GRADE_RESULT = "document_grade_result"
    
    # Query Transformation
    QUERY_TRANSFORM = "query_transform"
    
    # Web Search
    WEB_SEARCH_START = "web_search_start"
    WEB_SEARCH_STRATEGY = "web_search_strategy"
    WEB_SEARCH_RESULT = "web_search_result"
    WEB_SEARCH_END = "web_search_end"
    
    # Generation
    GENERATION_START = "generation_start"
    GENERATION_END = "generation_end"
    
    # Grading
    HALLUCINATION_CHECK = "hallucination_check"
    ANSWER_CHECK = "answer_check"
    
    # Loop Control
    RETRY_INCREMENT = "retry_increment"
    LIMIT_REACHED = "limit_reached"
    
    # Errors
    ERROR = "error"
    WARNING = "warning"


@dataclass
class LogEvent:
    """Represents a single log event."""
    session_id: str
    event_type: EventType
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    step_number: int = 0
    
    def to_dict(self) -> Dict:
        result = {
            "session_id": self.session_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "timestamp": self.timestamp,
            "step_number": self.step_number,
            "data": self.data
        }
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result


class RAGLogger:
    """
    Comprehensive logger for RAG pipeline.
    
    Usage:
        logger = RAGLogger()
        session_id = logger.start_session(user_query="What is AI?")
        
        logger.log_event(EventType.SAFETY_CHECK, {"result": "safe"})
        logger.log_event(EventType.ROUTING_DECISION, {"route": "vectorstore"})
        
        # ... more events ...
        
        logger.end_session(final_answer="AI is...", success=True)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global logger access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.step_counter: int = 0
        self.events: List[LogEvent] = []
        self.query_history: List[str] = []  # Track all query transformations
        
        # Setup file handler for persistent logging
        self.log_file: Optional[str] = None
        
        # Also log to standard logger for console output
        self.console_logger = logging.getLogger("RAGPipeline")
        self.console_logger.setLevel(logging.INFO)
        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.console_logger.addHandler(handler)
    
    def start_session(self, user_query: str, metadata: Optional[Dict] = None) -> str:
        """
        Start a new logging session for a user query.
        
        Args:
            user_query: The original user question
            metadata: Optional additional metadata (e.g., user_id, api_key_hash)
            
        Returns:
            session_id: Unique identifier for this session
        """
        self.current_session_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        self.session_start_time = datetime.now()
        self.step_counter = 0
        self.events = []
        self.query_history = [user_query]
        
        # Create session log file
        date_str = self.session_start_time.strftime("%Y-%m-%d")
        time_str = self.session_start_time.strftime("%H-%M-%S")
        self.log_file = os.path.join(LOG_DIR, f"{date_str}_{time_str}_{self.current_session_id}.json")
        
        # Log session start
        self.log_event(EventType.SESSION_START, {
            "original_query": user_query,
            "metadata": metadata or {},
            "log_file": self.log_file
        })
        
        self.log_event(EventType.USER_INPUT, {
            "query": user_query,
            "query_length": len(user_query),
            "word_count": len(user_query.split())
        })
        
        self.console_logger.info(f"[{self.current_session_id}] Session started: '{user_query[:50]}...'")
        
        return self.current_session_id
    
    def log_event(self, event_type: EventType, data: Dict[str, Any], 
                  duration_ms: Optional[float] = None) -> None:
        """
        Log a pipeline event.
        
        Args:
            event_type: Type of event (from EventType enum)
            data: Event-specific data dictionary
            duration_ms: Optional duration in milliseconds
        """
        if not self.current_session_id:
            # Auto-start a session if not started
            self.start_session("Unknown query")
        
        self.step_counter += 1
        
        event = LogEvent(
            session_id=self.current_session_id,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
            duration_ms=duration_ms,
            step_number=self.step_counter
        )
        
        self.events.append(event)
        
        # Console output for important events
        if event_type in [EventType.ROUTING_DECISION, EventType.QUERY_TRANSFORM, 
                          EventType.GENERATION_END, EventType.LIMIT_REACHED, EventType.ERROR]:
            self.console_logger.info(f"[{self.current_session_id}] Step {self.step_counter}: {event_type.value} - {self._summarize_data(data)}")
    
    def log_query_transform(self, original: str, transformed: str, reason: str = "") -> None:
        """Log a query transformation with history tracking."""
        self.query_history.append(transformed)
        self.log_event(EventType.QUERY_TRANSFORM, {
            "original_query": original,
            "transformed_query": transformed,
            "reason": reason,
            "transformation_number": len(self.query_history) - 1,
            "query_history": self.query_history.copy()
        })
    
    def log_retrieval(self, query: str, vector_results: List[Dict], bm25_results: List[Dict],
                      merged_results: List[Dict], reranked_results: List[Dict],
                      duration_ms: float) -> None:
        """Log complete retrieval pipeline with all intermediate results."""
        self.log_event(EventType.RETRIEVAL_START, {"query": query})
        
        self.log_event(EventType.VECTOR_SEARCH, {
            "query": query,
            "result_count": len(vector_results),
            "results": vector_results[:5]  # Limit to first 5 for log size
        })
        
        self.log_event(EventType.BM25_SEARCH, {
            "query": query,
            "result_count": len(bm25_results),
            "results": bm25_results[:5]
        })
        
        self.log_event(EventType.HYBRID_MERGE, {
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "merged_count": len(merged_results),
            "deduped": len(vector_results) + len(bm25_results) - len(merged_results)
        })
        
        self.log_event(EventType.RERANK, {
            "input_count": len(merged_results),
            "output_count": len(reranked_results),
            "results": reranked_results
        })
        
        self.log_event(EventType.RETRIEVAL_END, {
            "final_document_count": len(reranked_results),
            "total_duration_ms": duration_ms
        }, duration_ms=duration_ms)
    
    def log_document_grades(self, grades: List[Dict]) -> None:
        """Log document grading results."""
        relevant_count = sum(1 for g in grades if g.get("relevant"))
        
        self.log_event(EventType.DOCUMENT_GRADING, {
            "total_documents": len(grades),
            "relevant_count": relevant_count,
            "irrelevant_count": len(grades) - relevant_count,
            "grades": grades
        })
    
    def log_web_search(self, query: str, strategy: str, results: List[Dict], 
                       duration_ms: float) -> None:
        """Log web search with strategy and results."""
        self.log_event(EventType.WEB_SEARCH_START, {"query": query})
        
        self.log_event(EventType.WEB_SEARCH_STRATEGY, {
            "strategy": strategy,
            "query_used": query
        })
        
        self.log_event(EventType.WEB_SEARCH_RESULT, {
            "result_count": len(results),
            "results": results[:5]  # Limit for log size
        })
        
        self.log_event(EventType.WEB_SEARCH_END, {
            "total_results": len(results),
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
    
    def log_generation(self, context_length: int, response_length: int, 
                       generation_count: int, duration_ms: float) -> None:
        """Log LLM generation details."""
        self.log_event(EventType.GENERATION_START, {
            "context_length_chars": context_length,
            "generation_attempt": generation_count
        })
        
        self.log_event(EventType.GENERATION_END, {
            "response_length_chars": response_length,
            "generation_attempt": generation_count,
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
    
    def log_grading_result(self, grade_type: str, result: str, details: Dict = None) -> None:
        """Log hallucination or answer grading result."""
        event_type = EventType.HALLUCINATION_CHECK if grade_type == "hallucination" else EventType.ANSWER_CHECK
        self.log_event(event_type, {
            "result": result,
            "passed": result.lower() == "yes",
            "details": details or {}
        })
    
    def log_retry(self, retry_count: int, generation_count: int, reason: str) -> None:
        """Log retry increment."""
        self.log_event(EventType.RETRY_INCREMENT, {
            "retry_count": retry_count,
            "generation_count": generation_count,
            "reason": reason,
            "query_history": self.query_history.copy()
        })
    
    def log_limit_reached(self, limit_type: str, current_value: int, max_value: int) -> None:
        """Log when a limit is reached."""
        self.log_event(EventType.LIMIT_REACHED, {
            "limit_type": limit_type,
            "current_value": current_value,
            "max_value": max_value,
            "total_query_transformations": len(self.query_history) - 1
        })
        self.console_logger.warning(f"[{self.current_session_id}] LIMIT REACHED: {limit_type} ({current_value}/{max_value})")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with full context."""
        self.log_event(EventType.ERROR, {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        })
        self.console_logger.error(f"[{self.current_session_id}] ERROR: {type(error).__name__}: {str(error)}")
    
    def end_session(self, final_answer: str, success: bool, 
                    source_type: str = "vectorstore", source_count: int = 0) -> Dict:
        """
        End the logging session and write to file.
        
        Returns:
            Session summary dictionary
        """
        if not self.session_start_time:
            return {}
            
        end_time = datetime.now()
        total_duration_ms = (end_time - self.session_start_time).total_seconds() * 1000
        
        summary = {
            "session_id": self.current_session_id,
            "original_query": self.query_history[0] if self.query_history else "",
            "final_query": self.query_history[-1] if self.query_history else "",
            "query_transformations": len(self.query_history) - 1,
            "total_steps": self.step_counter,
            "total_duration_ms": total_duration_ms,
            "success": success,
            "source_type": source_type,
            "source_count": source_count,
            "answer_length": len(final_answer),
            "answer_preview": final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
        }
        
        self.log_event(EventType.SESSION_END, summary, duration_ms=total_duration_ms)
        
        # Write all events to file
        self._write_to_file(summary)
        
        self.console_logger.info(
            f"[{self.current_session_id}] Session ended: "
            f"{'SUCCESS' if success else 'FAILED'} | "
            f"{total_duration_ms:.0f}ms | "
            f"{self.step_counter} steps | "
            f"{len(self.query_history)-1} rewrites"
        )
        
        return summary
    
    def _write_to_file(self, summary: Dict) -> None:
        """Write all events to the log file."""
        if not self.log_file:
            return
            
        log_data = {
            "summary": summary,
            "query_history": self.query_history,
            "events": [event.to_dict() for event in self.events]
        }
        
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.console_logger.error(f"Failed to write log file: {e}")
    
    def _summarize_data(self, data: Dict) -> str:
        """Create a brief summary of event data for console output."""
        if "result" in data:
            return f"result={data['result']}"
        if "route" in data:
            return f"route={data['route']}"
        if "transformed_query" in data:
            return f"query='{data['transformed_query'][:30]}...'"
        if "response_length_chars" in data:
            return f"length={data['response_length_chars']}"
        if "limit_type" in data:
            return f"{data['limit_type']}={data.get('current_value')}/{data.get('max_value')}"
        return str(data)[:50]
    
    def get_session_log(self) -> Dict:
        """Get the current session log as a dictionary."""
        return {
            "session_id": self.current_session_id,
            "query_history": self.query_history,
            "events": [event.to_dict() for event in self.events]
        }


# Global logger instance
rag_logger = RAGLogger()


def get_logger() -> RAGLogger:
    """Get the global RAG logger instance."""
    return rag_logger
