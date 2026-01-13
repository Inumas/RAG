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

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]
    api_key: str
    retriever: any  # The retriever object
    web_search: bool
    safety_status: str # "safe" or "unsafe"

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
    Generate answer
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    api_key = state["api_key"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    template = """You are a helpful assistant for "The Batch" newsletter.
    Answer the question based ONLY on the following context. 
    If the context contains web search results, use them to answer the question.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    # Concatenate docs
    context_text = "\n\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in documents])
    
    generation = chain.invoke({"context": context_text, "question": question})
    return {"documents": documents, "question": question, "generation": generation.content}

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
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    api_key = state["api_key"]
    
    rewriter = get_rewriter_agent(api_key)
    better_question = rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question.content}

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
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    api_key = state["api_key"]
    
    hallucination_grader = get_hallucination_grader(api_key)
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check answer to question
        print("---CHECK ANSWER TO QUESTION---")
        answer_grader = get_answer_grader(api_key)
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
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
            "not supported": "generate",     # Potentially map to transform_query to avoid infinite loop
        },
    )
    
    return workflow.compile()
