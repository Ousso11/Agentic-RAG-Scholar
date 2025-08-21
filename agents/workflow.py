from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from research_agent import ResearchAgent
from synthesis_agent import SynthesisAgent
from relevance_checker import RelevanceChecker
from document_processing.file_handler import FileHandler
from document_processing.document_processor import DocumentProcessor
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from retriever import RetrieverBuilder
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    files : List[str]
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever
    
class WorkflowGraph(StateGraph[AgentState]):
    def __init__(self): 
        super().__init__()
        self.retriever = RetrieverBuilder()
        self.file_handler = FileHandler()
        self.document_processor = DocumentProcessor()

    def pipeline(self, query, files):
        paths = self.file_handler.process_documents(files)
        documents = self.document_processor.batch_process_files(paths)
        pass
    

    