from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from research_agent import ResearchAgent
from synthesis_agent import SynthesisAgent
from relevance_checker import RelevanceChecker
from document_processing.processor import DocumentProcessingUtils

from langchain.schema import Document
from retriever import Retriever
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    files : List[str]
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: Retriever
    
class WorkflowGraph(StateGraph[AgentState]):
    def __init__(self, question: str):
        initial_state: AgentState = {
            "question": question,
            "files": [],
            "documents": [],
            "draft_answer": "",
            "verification_report": "",
            "is_relevant": False,
            "retriever": Retriever(retrievers=[], weights={})
        }
        super().__init__(initial_state)

    