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
        """Detect if query is asking about recency/time."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in TEMPORAL_KEYWORDS)

    def _is_pure_recency_query(self, query: str) -> bool:
        """
        Detect if query is ONLY asking about recency (no specific topic).
        Examples:
            - "What's the most recent article?" → True (pure recency)
            - "Latest article about transformers" → False (has topic)
        """
        import re
        
        # Common filler words (generic query words, not topic words)
        filler_words = [
            # Contractions and question starters
            "what's", "whats", "what", "who", "where", "when", "how", "why",
            "the", "is", "are", "was", "were", "it", "its", "it's", "be", "been",
            "a", "an", "in", "from", "about", "of", "to", "for", "and", "or",
            "show", "me", "tell", "give", "find", "get", "can", "you", "could",
            "please", "i", "want", "would", "like", "know", "wondering",
            # Document/content words
            "article", "articles", "post", "posts", "news", "newsletter", 
            "batch", "issue", "issues", "content", "summary", "summarize",
            "number", "title", "topic", "subject", "cover", "covering",
            # Temporal helpers
            "most",  # "most recent" is a temporal phrase
            # Contraction fragments (what's -> s, don't -> t, etc.)
            "s", "t", "re", "ve", "ll", "d",
        ]
        
        # Tokenize: split on non-alphanumeric, lowercase
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Remove temporal keywords (exact word match)
        temporal_words = set(kw.lower() for kw in TEMPORAL_KEYWORDS if ' ' not in kw)
        # Also add individual words from multi-word temporal phrases
        for kw in TEMPORAL_KEYWORDS:
            if ' ' in kw:
                temporal_words.update(kw.lower().split())
        
        # Filter out temporal and filler words
        filler_set = set(filler_words)
        remaining_tokens = [t for t in tokens if t not in temporal_words and t not in filler_set]
        
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

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Performs hybrid search (Vector + BM25) and deduplicates."""
        
        # 1. Vector Search
        vector_docs = self.vectorstore.similarity_search(query, k=k)
        
        # 2. BM25 Search
        bm25_docs = []
        if self.bm25:
            tokenized_query = self._tokenize(query)
            bm25_docs = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=k)
            
        # 3. Combine and Deduplicate
        # key = source + content hash? or just content?
        # Let's use content as primary key for dedupe to avoid duplicates
        seen_content = set()
        unique_docs = []
        
        for doc in vector_docs + bm25_docs:
            # Use a snippet of content as hash or the unique ID if we have it?
            # We constructed IDs in database.py, but similarity_search might not return them directly in doc.id 
            # unless we ask for it (Langchain update).
            # Let's use page_content for now.
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
                
        return unique_docs

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

    def retrieve(self, query: str, mode: str = "hybrid") -> List[Document]:
        """
        Main retrieval entry point with temporal awareness.
        
        For temporal queries (containing words like "recent", "latest", "last week"):
        - PURE RECENCY ("What's the most recent article?"): Return docs from highest issue#
        - TOPIC+RECENCY ("Latest article about AI"): Pool from recent issues + semantic rerank
        
        This ensures "most recent article" returns Issue 334, not Issue 231.
        """
        
        # Check for temporal queries first
        if self._is_temporal_query(query):
            logger.info(f"Temporal query detected: '{query}'")
            
            # Get docs from latest issues
            recent_docs = self._get_docs_from_latest_issues(n_issues=20)
            
            if not recent_docs:
                # Fallback to hybrid if no issue metadata
                logger.info("No issue metadata found, falling back to hybrid search")
                candidates = self.hybrid_search(query, k=10)
                return self.rerank(query, candidates, top_n=3)
            
            # Check if it's a PURE recency query (no specific topic)
            if self._is_pure_recency_query(query):
                logger.info("Pure recency query - returning docs from highest issue number")
                
                # Sort by issue number descending
                sorted_docs = sorted(recent_docs, 
                                   key=lambda x: x.metadata.get("issue_number", 0),
                                   reverse=True)
                
                # Return docs from the highest issue with ACTUAL CONTENT (not just headers)
                max_issue = sorted_docs[0].metadata.get("issue_number")
                top_issue_docs = [d for d in sorted_docs 
                                 if d.metadata.get("issue_number") == max_issue 
                                 and len(d.page_content) > 100]  # Filter out short header chunks
                
                # If all docs are short headers, fall back to any docs from max issue
                if not top_issue_docs:
                    top_issue_docs = [d for d in sorted_docs if d.metadata.get("issue_number") == max_issue]
                
                logger.info(f"Returning {len(top_issue_docs[:3])} docs from issue {max_issue}")
                return top_issue_docs[:3]
            
            # TOPIC + RECENCY: Semantic rerank within recent docs pool
            # Limit pool size to avoid slow reranking (991 docs = 110+ seconds!)
            MAX_RERANK_POOL = 50  # Cross-encoder on CPU can't handle more
            if len(recent_docs) > MAX_RERANK_POOL:
                # Pre-filter with BM25 to get most relevant subset
                tokenized_query = self._tokenize(query)
                if self.bm25:
                    # Create temp BM25 index for recent docs only
                    recent_corpus = [self._tokenize(d.page_content) for d in recent_docs]
                    from rank_bm25 import BM25Okapi
                    temp_bm25 = BM25Okapi(recent_corpus)
                    recent_docs = temp_bm25.get_top_n(tokenized_query, recent_docs, n=MAX_RERANK_POOL)
                else:
                    recent_docs = recent_docs[:MAX_RERANK_POOL]
                    
            logger.info(f"Topic+recency query - reranking {len(recent_docs)} docs from recent issues")
            return self.rerank(query, recent_docs, top_n=3)
        
        # Default: Hybrid search (unchanged behavior)
        # Step 1: Broad Retrieval (Top 10 candidates)
        candidates = self.hybrid_search(query, k=10)
        
        # Step 2: Rerank (Refine to Top 3)
        final_docs = self.rerank(query, candidates, top_n=3)
        
        return final_docs
