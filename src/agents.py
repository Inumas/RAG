from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

# --- Data Models ---

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., 
        description="Given a user question choose to route it to web search or a vectorstore."
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        ..., 
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# --- Agent Functions ---

def get_router_agent(api_key):
    """
    Returns a runnable chain that routes the query.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains newsletters from "The Batch" (DeepLearning.AI), covering AI news, machine learning, and tech updates.
    Use the vectorstore for questions on these topics.
    Otherwise, use web-search."""
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    
    question_router = route_prompt | structured_llm_router
    return question_router

def get_grading_agent(api_key):
    """
    Returns a runnable chain that grades document relevance.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def get_rewriter_agent(api_key):
    """
    Returns a runnable chain that rewrites the query.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    
    question_rewriter = re_write_prompt | llm
    return question_rewriter
