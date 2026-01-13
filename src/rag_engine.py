from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retrieval import HybridRetriever

# Global instance to avoid rebuilding index on every query (if app restarts logic allows)
# In Streamlit, this runs every rerun unless cached, handled later. For now, instantiate.

def query_rag(question, api_key, retriever):
    """Queries the RAG system using Hybrid Search + Re-ranking."""
    if not api_key:
        raise ValueError("API Key is required.")

    # Retriever passed in (dependency injection)
    
    # 1. Retrieve (Hybrid + Rerank)
    docs = retriever.retrieve(question)
    
    # Format context string
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # 2. Generation
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
    
    template = """You are a helpful assistant for "The Batch" newsletter.
    Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    
    If you don't know the answer from the context, say so.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": question})
    
    return {
        "answer": response.content,
        "source_documents": docs
    }
    
    return {
        "answer": response.content,
        "source_documents": docs
    }
