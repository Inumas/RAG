from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from agents import get_router_agent, get_grading_agent, get_rewriter_agent
from web_search import web_search

def query_rag(question, api_key, retriever):
    """
    Queries the RAG system using an Agentic Workflow:
    1. Route (VectorStore vs Web Search)
    2. If VectorStore -> Retrieve -> Grade
       - If Relevant -> Generate
       - If Not Relevant -> Rewrite -> Web Search -> Generate
    3. If Web Search -> Search -> Generate
    """
    if not api_key:
        raise ValueError("API Key is required.")

    # Initialize Agents
    router = get_router_agent(api_key)
    grader = get_grading_agent(api_key)
    rewriter = get_rewriter_agent(api_key)
    
    # Standard Generation Chain
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
    template = """You are a helpful assistant for "The Batch" newsletter.
    Answer the question based ONLY on the following context. 
    If the context contains web search results, use them to answer the question.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    print(f"--- Routing Query: {question} ---")
    route = router.invoke({"question": question})
    
    context_docs = []
    source_type = "vectorstore"
    
    if route.datasource == "web_search":
        print("--- Routing to Web Search ---")
        web_results = web_search(question)
        context_docs = [Document(page_content=web_results, metadata={"source": "web_search"})]
        source_type = "web_search"
    else:
        print("--- Routing to Vector Store ---")
        docs = retriever.retrieve(question)
        
        # Grade Documents
        print("--- Grading Documents ---")
        filtered_docs = []
        for d in docs:
            score = grader.invoke({"document": d.page_content, "question": question})
            if score.binary_score == "yes":
                print("--- Document Relevant ---")
                filtered_docs.append(d)
            else:
                print("--- Document Irrelevant ---")
        
        if filtered_docs:
            print("--- Relevant Documents Found ---")
            context_docs = filtered_docs
        else:
            print("--- No Relevant Documents -> Rewriting Query ---")
            # Rewrite Query
            better_question = rewriter.invoke({"question": question}).content
            print(f"--- Rewritten Query: {better_question} ---")
            
            # Fallback to Web Search
            print("--- Fallback to Web Search ---")
            web_results = web_search(better_question)
            context_docs = [Document(page_content=web_results, metadata={"source": "web_search"})]
            source_type = "web_search_fallback"

    # Generate Answer
    print("--- Generating Answer ---")
    context_text = "\n\n".join([d.page_content for d in context_docs])
    response = chain.invoke({"context": context_text, "question": question})
    
    return {
        "answer": response.content,
        "source_documents": context_docs,
        "source_type": source_type
    }
