import logging
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from database import get_vectorstore
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Temporal keywords for detecting recency queries
TEMPORAL_KEYWORDS = [
    # Recency words
    "recent", "latest", "newest", "last", "current", "new", "just",
    # Time periods  
    "today", "yesterday", "this week", "last week", "this month", 
    "last month", "this year", "last year", "past week", "past month",
    # Freshness
    "fresh", "up to date", "up-to-date", "most recent",
    # Specific years
    "2026", "2025", 
    # Months
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december"
]

class HybridRetriever:
    def __init__(self):
        """Initializes the Hybrid Retriever with Vector Store and BM25."""
        self.vectorstore = get_vectorstore()
        self.bm25 = None
        self.bm25_docs = []
        self.cross_encoder = None
        
        # Load documents and build BM25 index on initialization
        # Note: In a production app with millions of docs, we wouldn't load all into memory.
        # For this demo/batch size, it's acceptable.
        self._build_bm25_index()
        
    def _build_bm25_index(self):
        """Fetches all documents from Chroma and builds BM25 index."""
        try:
            # Chroma get() fetches all if no ids provided? No, strictly it might not.
            # let's try getting all IDs or just all docs.
            # API for Chroma get: collection.get()
            results = self.vectorstore.get() 
            
            if not results['documents']:
                logger.info("No documents found in VectorStore for BM25.")
                return

            self.bm25_docs = []
            tokenized_corpus = []
            
            api_docs = results['documents']
            api_metadatas = results['metadatas']
            
            for i, text in enumerate(api_docs):
                # Reconstruct Document object
                doc = Document(page_content=text, metadata=api_metadatas[i] if api_metadatas else {})
                self.bm25_docs.append(doc)
                
                # Simple tokenization for BM25
                tokens = self._tokenize(text)
                tokenized_corpus.append(tokens)
                
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index built with {len(self.bm25_docs)} documents.")
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase and remove punctuation."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def _is_temporal_query(self, query: str) -> bool:
        """Detect if query is asking about recency/time (robust to typos)."""
        import difflib
        
        query_lower = query.lower()
        
        # 1. Exact match check (fast path)
        if any(kw in query_lower for kw in TEMPORAL_KEYWORDS):
            return True
            
        # 2. Fuzzy match check
        # Split query into words to check against keywords
        query_words = self._tokenize_regex(query)
        
        for word in query_words:
            # Skip short words to avoid false positives
            if len(word) < 4: 
                continue
                
            # Check against single-word temporal keywords
            single_word_kws = [kw for kw in TEMPORAL_KEYWORDS if ' ' not in kw]
            matches = difflib.get_close_matches(word, single_word_kws, n=1, cutoff=0.8)
            if matches:
                logger.debug(f"Fuzzy match: '{word}' -> '{matches[0]}'")
                return True
                
        return False

    def _tokenize_regex(self, text: str) -> List[str]:
        """Regex-based tokenizer for cleaner splitting."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _is_pure_recency_query(self, query: str) -> bool:
        """
        Detect if query is ONLY asking about recency (no specific topic).
        Includes robust cleaning of typos for temporal/filler words.
        """
        import difflib
        
        # Common filler words (generic query words, not topic words)
        filler_words = [
            # Contractions and question starters
            "what's", "whats", "what", "who", "where", "when", "how", "why",
            "the", "is", "are", "was", "were", "it", "its", "it's", "be", "been",
            "a", "an", "in", "from", "about", "of", "to", "for", "and", "or",
            "show", "me", "tell", "give", "find", "get", "can", "you", "could",
            "please", "i", "want", "would", "like", "know", "wondering",
            "we", "have", "us", "our", "do", "does", "did",
            # Document/content words
            "article", "articles", "post", "posts", "news", "newsletter", 
            "batch", "issue", "issues", "content", "summary", "summarize",
            "number", "title", "topic", "subject", "cover", "covering",
            "published", "date", "dated", "release", "released",
            "discuss", "discussed", "discussion", "mention", "mentioned",
            "say", "says", "said", "talk", "talks", "talked",
            # Temporal helpers
            "most",  # "most recent" is a temporal phrase
            # Contraction fragments (what's -> s, don't -> t, etc.)
            "s", "t", "re", "ve", "ll", "d",
        ]
        
        tokens = self._tokenize_regex(query)
        
        # Build sets for efficient checking
        temporal_kws_single = [kw for kw in TEMPORAL_KEYWORDS if ' ' not in kw]
        temporal_kws_multi = [kw for kw in TEMPORAL_KEYWORDS if ' ' in kw]
        
        # Add individual words from multi-word phrases to temporal set
        for kw in temporal_kws_multi:
            temporal_kws_single.extend(kw.split())
            
        temporal_set = set(temporal_kws_single)
        filler_set = set(filler_words)
        
        remaining_tokens = []
        
        for token in tokens:
            # 1. Exact match
            if token in temporal_set or token in filler_set:
                continue
                
            # 2. Fuzzy match (only for longer words)
            if len(token) >= 4:
                # Check temporal
                if difflib.get_close_matches(token, temporal_kws_single, n=1, cutoff=0.8):
                    continue
                # Check filler
                if difflib.get_close_matches(token, filler_words, n=1, cutoff=0.8):
                    continue
            
            # If we get here, it's a potential topic word
            remaining_tokens.append(token)
        
        # If no meaningful tokens remain, it's a pure recency query
        logger.debug(f"Pure recency check: '{query}' -> remaining tokens: {remaining_tokens}")
        return len(remaining_tokens) == 0

    def _get_docs_from_latest_issues(self, n_issues: int = 20) -> List[Document]:
        """
        Get ALL documents from the N most recent issues.
        Returns a wider pool for semantic reranking to find relevant content.
        
        Args:
            n_issues: Number of recent issues to include (default 20)
            
        Returns:
            List of all documents from the most recent N issues
        """
        # Filter to docs with issue_number metadata
        docs_with_issue = [d for d in self.bm25_docs if d.metadata.get("issue_number")]
        
        if not docs_with_issue:
            logger.info("No documents with issue_number metadata found")
            return []
        
        # Find the top N unique issue numbers
        issue_numbers = set(d.metadata.get("issue_number") for d in docs_with_issue)
        top_issues = sorted(issue_numbers, reverse=True)[:n_issues]
        
        # Get ALL docs from those issues (not just one per issue)
        result = [d for d in docs_with_issue if d.metadata.get("issue_number") in top_issues]
        
        logger.info(f"Temporal retrieval: {len(result)} docs from issues {min(top_issues)}-{max(top_issues)}")
        return result

    def _init_cross_encoder(self):
        """Lazy load cross encoder to save startup time if not used."""
        if not self.cross_encoder:
            logger.info("Loading Cross-Encoder model...")
            # Use a lightweight fast model
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def hybrid_search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Document]:
        """Performs hybrid search (Vector + BM25) with optional filtering and deduplicates."""
        
        # 1. Vector Search with Filters
        chroma_filter = self._build_chroma_filter(filters)
        vector_docs = self.vectorstore.similarity_search(query, k=k, filter=chroma_filter)
        
        # 2. BM25 Search
        bm25_docs = []
        if self.bm25:
            tokenized_query = self._tokenize(query)
            # Fetch more candidates for post-filtering
            candidates = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=k*3)
            
            # Post-filter BM25 results
            if filters:
                bm25_docs = [d for d in candidates if self._matches_filters(d, filters)][:k]
            else:
                bm25_docs = candidates[:k]
            
        # 3. Combine and Deduplicate
        seen_content = set()
        unique_docs = []
        
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
                
        return unique_docs

    def _build_chroma_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs a ChromaDB-compatible filter dictionary."""
        import datetime
        
        if not filters:
            return None
            
        where_clauses = []
        
        # Topic filter
        if filters.get("topic") and filters["topic"] != "All":
            where_clauses.append({"topic": filters["topic"]})
            
        # Date filtering (convert string dates to timestamps for comparison)
        if filters.get("start_date"):
            # Start of day
            dt = datetime.datetime.strptime(filters["start_date"], "%Y-%m-%d")
            ts = int(dt.timestamp())
            where_clauses.append({"timestamp": {"$gte": ts}})
            
        if filters.get("end_date"):
            # End of day (approx, or just same timestamp if relying on day granularity)
            dt = datetime.datetime.strptime(filters["end_date"], "%Y-%m-%d")
            # Set to end of day? 
            dt = dt.replace(hour=23, minute=59, second=59)
            ts = int(dt.timestamp())
            where_clauses.append({"timestamp": {"$lte": ts}})
            
        if not where_clauses:
            return None
        if len(where_clauses) == 1:
            return where_clauses[0]
        return {"$and": where_clauses}

    def _matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """In-memory check if a document matches filters (for BM25)."""
        import datetime
        
        if not filters:
            return True
            
        meta = doc.metadata
        
        # Topic check
        if filters.get("topic") and filters["topic"] != "All":
            if meta.get("topic") != filters["topic"]:
                return False
                
        # Date check - using timestamps if available, or fallback to string compare?
        # New ingestion adds 'timestamp'. Check it.
        doc_ts = meta.get("timestamp")
        
        if doc_ts is None:
            # If doc has no timestamp, exclude if filtering by date
            if filters.get("start_date") or filters.get("end_date"):
                return False
                
        if filters.get("start_date"):
             dt = datetime.datetime.strptime(filters["start_date"], "%Y-%m-%d")
             ts = int(dt.timestamp())
             if doc_ts < ts:
                 return False
                 
        if filters.get("end_date"):
             dt = datetime.datetime.strptime(filters["end_date"], "%Y-%m-%d")
             dt = dt.replace(hour=23, minute=59, second=59)
             ts = int(dt.timestamp())
             if doc_ts > ts:
                 return False
            
        return True

    def rerank(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        """Reranks documents using Cross-Encoder."""
        if not docs:
            return []
            
        self._init_cross_encoder()
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [doc for doc, score in scored_docs[:top_n]]

    def retrieve(self, query: str, mode: str = "hybrid", filters: Dict[str, Any] = None) -> List[Document]:
        """
        Main retrieval entry point with temporal awareness and filtering.
        """
        
        # Check for temporal queries first (only if NO explicit filters provided)
        # If user explicitly filters, we should respect that over auto-temporal logic?
        # Let's say explicit filters TAKE PRECEDENCE on constraints, but implicit recency queries 
        # still prioritize recent stuff WITHIN those constraints.
        
        if self._is_temporal_query(query) and not filters:
            logger.info(f"Temporal query detected: '{query}'")
            
            # Get docs from latest issues
            recent_docs = self._get_docs_from_latest_issues(n_issues=20)
            
            if not recent_docs:
                logger.info("No issue metadata found, falling back to hybrid search")
                candidates = self.hybrid_search(query, k=10, filters=filters)
                return self.rerank(query, candidates, top_n=3)
            
            # Check if it's a PURE recency query
            if self._is_pure_recency_query(query):
                logger.info("Pure recency query - returning docs from highest issue number")
                sorted_docs = sorted(recent_docs, 
                                   key=lambda x: x.metadata.get("issue_number", 0),
                                   reverse=True)
                max_issue = sorted_docs[0].metadata.get("issue_number")
                top_issue_docs = [d for d in sorted_docs 
                                 if d.metadata.get("issue_number") == max_issue 
                                 and len(d.page_content) > 100]
                if not top_issue_docs:
                    top_issue_docs = [d for d in sorted_docs if d.metadata.get("issue_number") == max_issue]
                return top_issue_docs[:3]
            
            # TOPIC + RECENCY
            MAX_RERANK_POOL = 50
            if len(recent_docs) > MAX_RERANK_POOL:
                tokenized_query = self._tokenize(query)
                if self.bm25:
                    recent_corpus = [self._tokenize(d.page_content) for d in recent_docs]
                    from rank_bm25 import BM25Okapi
                    temp_bm25 = BM25Okapi(recent_corpus)
                    recent_docs = temp_bm25.get_top_n(tokenized_query, recent_docs, n=MAX_RERANK_POOL)
                else:
                    recent_docs = recent_docs[:MAX_RERANK_POOL]
            return self.rerank(query, recent_docs, top_n=3)
        
        # Standard Search with Filters
        candidates = self.hybrid_search(query, k=10, filters=filters)
        return self.rerank(query, candidates, top_n=3)
