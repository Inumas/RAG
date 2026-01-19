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
from logger import get_logger, EventType
import sys
import time

# Maximum number of retries before giving up
MAX_RETRIES = 3
MAX_GENERATION_CALLS = 5  # Hard limit on generate calls

# Get the global logger
logger = get_logger()

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
    temporal_retrieval: bool  # Flag for temporal/recency queries - skip grading
    filters: dict  # Metadata filters (topic, date range)

# --- Security Nodes ---

def guard_input(state):
    """
    Check if the input is safe.
    """
    start_time = time.time()
    print("---GUARD INPUT---", flush=True)
    question = state["question"]
    api_key = state["api_key"]
    
    guard = get_input_guardrail_agent(api_key)
    result = guard.invoke({"question": question})
    
    duration_ms = (time.time() - start_time) * 1000
    
    if result.safe == "unsafe":
        print("---UNSAFE INPUT DETECTED---", flush=True)
        logger.log_event(EventType.SAFETY_CHECK, {
            "result": "unsafe",
            "question": question,
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
        return {"safety_status": "unsafe", "generation": "I cannot assist with that request as it violates our safety policy."}
    else:
        print("---INPUT SAFE---", flush=True)
        logger.log_event(EventType.SAFETY_CHECK, {
            "result": "safe",
            "question": question,
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
        return {"safety_status": "safe"}

# --- Core Nodes ---

def retrieve(state):
    """
    Retrieve documents
    """
    start_time = time.time()
    print("---RETRIEVE---", flush=True)
    question = state["question"]
    retriever = state["retriever"]
    filters = state.get("filters")
    
    # Check if this is a temporal query (for skipping grading)
    # Only treat as auto-temporal if NO filters are present. 
    # If filters are present, we depend on them.
    is_temporal = retriever._is_temporal_query(question) and not filters
    if is_temporal:
        print("---TEMPORAL QUERY DETECTED - WILL SKIP GRADING---", flush=True)
    
    documents = retriever.retrieve(question, filters=filters)
    
    duration_ms = (time.time() - start_time) * 1000
    
    # Log retrieval results
    doc_summaries = []
    for i, doc in enumerate(documents):
        doc_summaries.append({
            "index": i,
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", "unknown")
        })
    
    logger.log_event(EventType.RETRIEVAL_END, {
        "query": question,
        "document_count": len(documents),
        "documents": doc_summaries,
        "duration_ms": duration_ms,
        "temporal_retrieval": is_temporal,
        "filters": filters
    }, duration_ms=duration_ms)
    
    return {"documents": documents, "question": question, "temporal_retrieval": is_temporal}

def generate(state):
    """
    Generate answer. Tracks call count to prevent infinite loops.
    """
    start_time = time.time()
    generation_count = state.get("generation_count", 0) + 1
    print(f"---GENERATE (call {generation_count}/{MAX_GENERATION_CALLS})---", flush=True)
    
    # HARD LIMIT: If we've called generate too many times, return a fallback
    if generation_count > MAX_GENERATION_CALLS:
        print("---HARD LIMIT REACHED: Returning fallback answer---", flush=True)
        logger.log_limit_reached("generation_count", generation_count, MAX_GENERATION_CALLS)
        fallback_answer = ("I apologize, but I was unable to find a complete answer after multiple attempts. "
                          "Please try rephrasing your question or ensure data has been ingested into the system.")
        logger.log_generation(0, len(fallback_answer), generation_count, 0)
        return {
            "generation": fallback_answer,
            "generation_count": generation_count
        }
    
    question = state["question"]
    original_question = state.get("original_question", question)
    documents = state["documents"]
    api_key = state["api_key"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    # Concatenate docs with metadata
    context_parts = []
    for d in documents:
        content = d.page_content if hasattr(d, 'page_content') else str(d)
        meta = d.metadata if hasattr(d, 'metadata') else {}
        
        # Add metadata headers if available
        headers = []
        if meta.get('title'):
            headers.append(f"Title: {meta['title']}")
        if meta.get('publish_date'):
            headers.append(f"Date: {meta['publish_date']}")
        if meta.get('issue_number'):
            headers.append(f"Issue: {meta['issue_number']}")
            
        header_str = " | ".join(headers)
        if header_str:
            context_parts.append(f"[{header_str}]\n{content}")
        else:
            context_parts.append(content)
            
    context_text = "\n\n".join(context_parts)
    
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
    
    duration_ms = (time.time() - start_time) * 1000
    logger.log_generation(len(context_text), len(generation.content), generation_count, duration_ms)
    
    return {"documents": documents, "question": question, "generation": generation.content, "generation_count": generation_count}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    For temporal/recency queries, skip grading - docs from latest issues ARE relevant by definition.
    """
    start_time = time.time()
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---", flush=True)
    question = state["question"]
    documents = state["documents"]
    api_key = state["api_key"]
    
    # SKIP GRADING for temporal queries - they found the latest issue, that's what user asked for
    if state.get("temporal_retrieval", False):
        print("---TEMPORAL QUERY: SKIPPING GRADING - DOCS ARE RELEVANT BY DEFINITION---", flush=True)
        grades = [
            {
                "doc_index": i,
                "relevant": True,
                "source": d.metadata.get("source", "unknown"),
                "content_preview": d.page_content[:100] + "...",
                "reason": "temporal_auto_pass"
            }
            for i, d in enumerate(documents)
        ]
        logger.log_document_grades(grades)
        return {"documents": documents, "question": question, "web_search": False}
    
    grader = get_grading_agent(api_key)
    filtered_docs = []
    web_search_flag = False
    grades = []
    
    retry_count = state.get("retry_count", 0)
    
    for i, d in enumerate(documents):
        score = grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        grade_info = {
            "doc_index": i,
            "relevant": grade == "yes",
            "source": d.metadata.get("source", "unknown"),
            "content_preview": d.page_content[:100] + "..."
        }
        grades.append(grade_info)
        
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---", flush=True)
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---", flush=True)
            continue
            
    if not filtered_docs:
        # If max retries reached, use original docs anyway (better than nothing)
        if retry_count >= MAX_RETRIES:
            print(f"---NO RELEVANT DOCS BUT MAX RETRIES ({retry_count}) REACHED - USING ORIGINAL DOCS---", flush=True)
            filtered_docs = documents  # Use all docs, let generate handle it
            web_search_flag = False
        else:
            print("---NO RELEVANT DOCS FOUND: WILL TRANSFORM QUERY AND RE-RETRIEVE---", flush=True)
            web_search_flag = True
    
    duration_ms = (time.time() - start_time) * 1000
    logger.log_document_grades(grades)
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search_flag}

def transform_query(state):
    """
    Transform the query to produce a better question.
    Increments retry counter to prevent infinite loops.
    """
    start_time = time.time()
    retry_count = state.get("retry_count", 0) + 1
    generation_count = state.get("generation_count", 0)
    print(f"---TRANSFORM QUERY (retry {retry_count}/{MAX_RETRIES})---", flush=True)
    
    # Log retry increment
    logger.log_retry(retry_count, generation_count, "query_transform")
    
    # If we've exceeded retries, don't bother rewriting - just pass through
    if retry_count > MAX_RETRIES:
        print("---MAX RETRIES EXCEEDED - SKIPPING REWRITE---", flush=True)
        logger.log_limit_reached("retry_count", retry_count, MAX_RETRIES)
        return {"retry_count": retry_count}
    
    question = state["question"]
    documents = state["documents"]
    api_key = state["api_key"]
    
    rewriter = get_rewriter_agent(api_key)
    better_question = rewriter.invoke({"question": question})
    
    duration_ms = (time.time() - start_time) * 1000
    logger.log_query_transform(question, better_question.content, f"retry_{retry_count}")
    
    return {"documents": documents, "question": better_question.content, "retry_count": retry_count}

def web_search_node(state):
    """
    Web search based on the re-phrased question.
    """
    start_time = time.time()
    print("---WEB SEARCH---", flush=True)
    question = state["question"]
    
    results = web_search(question)
    docs = [Document(page_content=results, metadata={"source": "web_search"})]
    
    duration_ms = (time.time() - start_time) * 1000
    
    # Parse results for logging
    result_count = results.count("Title:") if results else 0
    logger.log_web_search(question, "multi_strategy", [{"content_preview": results[:500]}], duration_ms)
    
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
    Determines whether to generate an answer, or re-transform the query.
    If docs not relevant, re-try with transformed query (back to retrieve).
    """
    print("---DECIDE TO GENERATE---")
    web_search_flag = state["web_search"]
    retry_count = state.get("retry_count", 0)
    
    if web_search_flag:
        # Check if we've already retried too many times
        if retry_count >= MAX_RETRIES:
            print(f"---DECISION: GENERATE (max retries {retry_count} reached, using whatever we have)---")
            return "generate"
        print("---DECISION: TRANSFORM QUERY and RE-RETRIEVE---")
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
        logger.log_limit_reached("combined", retry_count + generation_count, MAX_RETRIES + MAX_GENERATION_CALLS)
        return "useful"  # Accept the answer to break the loop
    
    hallucination_grader = get_hallucination_grader(api_key)
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    logger.log_grading_result("hallucination", grade, {
        "generation_preview": generation[:200] + "...",
        "doc_count": len(documents)
    })
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---", flush=True)
        # Check answer to question
        print("---CHECK ANSWER TO QUESTION---", flush=True)
        answer_grader = get_answer_grader(api_key)
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        
        logger.log_grading_result("answer", grade, {
            "question": question,
            "generation_preview": generation[:200] + "..."
        })
        
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
    PRIORITY: Always use vectorstore if database has data.
    Web search is only for empty database scenarios.
    """
    start_time = time.time()
    print("---ROUTE QUESTION---", flush=True)
    question = state["question"]
    api_key = state["api_key"]
    retriever = state.get("retriever")
    
    # Check if database has data - if yes, ALWAYS use vectorstore
    db_has_data = retriever and hasattr(retriever, 'bm25_docs') and len(retriever.bm25_docs) > 0
    
    if db_has_data:
        duration_ms = (time.time() - start_time) * 1000
        print(f"---ROUTE: VECTORSTORE (DB has {len(retriever.bm25_docs)} docs)---", flush=True)
        logger.log_event(EventType.ROUTING_DECISION, {
            "route": "vectorstore",
            "question": question,
            "reason": "database_has_data",
            "doc_count": len(retriever.bm25_docs),
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
        return "vectorstore"
    
    # Only use router agent if database is empty
    router = get_router_agent(api_key)
    source = router.invoke({"question": question})
    
    duration_ms = (time.time() - start_time) * 1000
    
    if source.datasource == "web_search":
        print("---ROUTE: WEB SEARCH (DB empty)---", flush=True)
        logger.log_event(EventType.ROUTING_DECISION, {
            "route": "web_search",
            "question": question,
            "reason": "database_empty",
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE: RAG---", flush=True)
        logger.log_event(EventType.ROUTING_DECISION, {
            "route": "vectorstore",
            "question": question,
            "duration_ms": duration_ms
        }, duration_ms=duration_ms)
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
    
    # transform_query goes back to retrieve (not web_search) to re-try with better query
    workflow.add_edge("transform_query", "retrieve")
    
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
