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
        """Main retrieval entry point with supervisor logic (simple for now)."""
        
        # Supervisor Strategy:
        # For now, always do Hybrid -> Rerank as it's the robust default.
        # Future: Check if query is "keyword-heavy" (dates, names) -> favor BM25, etc.
        
        # Step 1: Broad Retrieval (Top 10 candidates)
        candidates = self.hybrid_search(query, k=10)
        
        # Step 2: Rerank (Refine to Top 3)
        final_docs = self.rerank(query, candidates, top_n=3)
        
        return final_docs
