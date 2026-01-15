from typing import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from agents import (
    get_router_agent, 
    get_grading_agent, 
    get_rewriter_agent, 
    get_hallucination_grader, 
    get_answer_grader,
    get_input_guardrail_agent
)
from web_search import web_search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sys

# Maximum number of retries before giving up
MAX_RETRIES = 3
MAX_GENERATION_CALLS = 5  # Hard limit on generate calls

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    original_question: str  # Keep original for reference
    generation: str
    documents: List[str]
    api_key: str
    retriever: any  # The retriever object
    web_search: bool
    safety_status: str # "safe" or "unsafe"
    retry_count: int  # Track number of query transform retries
    generation_count: int  # Track number of generate calls (hard limit)

# --- Security Nodes ---

def guard_input(state):
    """
    Check if the input is safe.
    """
    print("---GUARD INPUT---")
    question = state["question"]
    api_key = state["api_key"]
    
    guard = get_input_guardrail_agent(api_key)
    result = guard.invoke({"question": question})
    
    if result.safe == "unsafe":
        print("---UNSAFE INPUT DETECTED---")
        return {"safety_status": "unsafe", "generation": "I cannot assist with that request as it violates our safety policy."}
    else:
        print("---INPUT SAFE---")
        return {"safety_status": "safe"}

# --- Core Nodes ---

def retrieve(state):
    """
    Retrieve documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]
    documents = retriever.retrieve(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer. Tracks call count to prevent infinite loops.
    """
    generation_count = state.get("generation_count", 0) + 1
    print(f"---GENERATE (call {generation_count}/{MAX_GENERATION_CALLS})---", flush=True)
    
    # HARD LIMIT: If we've called generate too many times, return a fallback
    if generation_count > MAX_GENERATION_CALLS:
        print("---HARD LIMIT REACHED: Returning fallback answer---", flush=True)
        return {
            "generation": "I apologize, but I was unable to find a complete answer after multiple attempts. "
                         "Please try rephrasing your question or ensure data has been ingested into the system.",
            "generation_count": generation_count
        }
    
    question = state["question"]
    original_question = state.get("original_question", question)
    documents = state["documents"]
    api_key = state["api_key"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    # Concatenate docs
    context_text = "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in documents])
    
    # Handle empty context gracefully
    if not context_text.strip() or "No web search results found" in context_text:
        template = """You are a helpful assistant for "The Batch" newsletter by DeepLearning.AI.
        
        The user asked: {question}
        
        Unfortunately, I couldn't find specific information in the database or web search.
        Please provide a helpful response that:
        1. Acknowledges you couldn't find the specific article
        2. Suggests the user try ingesting data first or rephrasing their question
        3. Briefly mentions what "The Batch" newsletter typically covers (AI news, ML research, tech updates)
        
        Answer:"""
    else:
        template = """You are a helpful assistant for "The Batch" newsletter by DeepLearning.AI.
        Answer the question based on the following context. 
        If the context contains web search results, synthesize them into a coherent answer.
        
        Context:
        {context}
        
        Original Question: {original_question}
        Current Question: {question}
        
        Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    generation = chain.invoke({
        "context": context_text, 
        "question": question,
        "original_question": original_question
    })
    return {"documents": documents, "question": question, "generation": generation.content, "generation_count": generation_count}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    api_key = state["api_key"]
    
    grader = get_grading_agent(api_key)
    filtered_docs = []
    web_search_flag = False
    
    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    if not filtered_docs:
        print("---NO RELEVANT DOCS FOUND: SETTING WEB SEARCH FLAG---")
        web_search_flag = True
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search_flag}

def transform_query(state):
    """
    Transform the query to produce a better question.
    Increments retry counter to prevent infinite loops.
    """
    retry_count = state.get("retry_count", 0) + 1
    print(f"---TRANSFORM QUERY (retry {retry_count}/{MAX_RETRIES})---", flush=True)
    
    # If we've exceeded retries, don't bother rewriting - just pass through
    if retry_count > MAX_RETRIES:
        print("---MAX RETRIES EXCEEDED - SKIPPING REWRITE---", flush=True)
        return {"retry_count": retry_count}
    
    question = state["question"]
    documents = state["documents"]
    api_key = state["api_key"]
    
    rewriter = get_rewriter_agent(api_key)
    better_question = rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question.content, "retry_count": retry_count}

def web_search_node(state):
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    
    results = web_search(question)
    docs = [Document(page_content=results, metadata={"source": "web_search"})]
    
    return {"documents": docs, "question": question}

# --- Conditional Edges ---

def check_safety(state):
    """
    Check safety status to decide route.
    """
    status = state["safety_status"]
    if status == "unsafe":
        return "unsafe"
    else:
        return "safe"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---DECIDE TO GENERATE---")
    web_search_flag = state["web_search"]
    
    if web_search_flag:
        print("---DECISION: TRANSFORM QUERY and WEB SEARCH---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    Includes multiple limit checks to prevent infinite loops.
    """
    print("---CHECK HALLUCINATIONS---", flush=True)
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    api_key = state["api_key"]
    retry_count = state.get("retry_count", 0)
    generation_count = state.get("generation_count", 0)
    
    # Check BOTH limits - if either exceeded, accept whatever we have
    if retry_count >= MAX_RETRIES or generation_count >= MAX_GENERATION_CALLS:
        print(f"---LIMIT REACHED (retries={retry_count}, generations={generation_count}) - ACCEPTING ANSWER---", flush=True)
        return "useful"  # Accept the answer to break the loop
    
    hallucination_grader = get_hallucination_grader(api_key)
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---", flush=True)
        # Check answer to question
        print("---CHECK ANSWER TO QUESTION---", flush=True)
        answer_grader = get_answer_grader(api_key)
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---", flush=True)
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---", flush=True)
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---", flush=True)
        return "not supported"

def route_question(state):
    """
    Route question to web search or RAG.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    api_key = state["api_key"]
    router = get_router_agent(api_key)
    source = router.invoke({"question": question})
    
    if source.datasource == "web_search":
        print("---ROUTE: WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE: RAG---")
        return "vectorstore"

def build_graph():
    """Builds the StateGraph."""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("guard_input", guard_input)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Entry Point: Always Guard Input
    workflow.set_entry_point("guard_input")
    
    # Conditional Edge from Guard Input
    # If safe, check router. If unsafe, END.
    def guard_router(state):
        if state.get("safety_status") == "unsafe":
            return "unsafe"
        return route_question(state)

    workflow.add_conditional_edges(
        "guard_input",
        guard_router,
        {
            "unsafe": END,
            "web_search": "web_search",
            "vectorstore": "retrieve",
        }
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("transform_query", "web_search")
    
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "useful": END,               
            "not useful": "transform_query", 
            "not supported": "transform_query",  # FIXED: Go through transform_query to increment retry counter
        },
    )
    
    return workflow.compile()
