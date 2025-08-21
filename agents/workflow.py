# workflow_graph.py
from __future__ import annotations
import logging
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import Document

from document_processing.file_handler import FileHandler
from document_processing.document_processor import DocumentProcessor
from agents.retriever import RetrieverBuilder
from agents.relevance_checker import RelevanceChecker
from agents.web_search_agent import WebSearchAgent

logger = logging.getLogger(__name__)

# -------- State --------
class AgentState(TypedDict, total=False):
    question: str
    # input
    files: List[dict]                 # [{type: "url"|"pdf", value: "..."}]
    # artifacts
    paths: List[str]
    documents: List[Document]
    retriever: object
    relevance: str                    # CAN_ANSWER | PARTIAL | NO_MATCH
    web_results: List[dict]
    final_answer: str

# -------- Orchestrator --------
class WorkflowGraph:
    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        chroma_dir: str,
        vector_k: int = 8,
        retriever_weights=(0.5, 0.5),
    ):
        self.file_handler = FileHandler()
        self.doc_processor = DocumentProcessor()
        self.retriever_builder = RetrieverBuilder(
            embedding_model_name=embedding_model_name,
            persist_dir=chroma_dir,
            k=vector_k,
            weights=retriever_weights,
        )
        self.checker = RelevanceChecker(model_name=llm_model_name)
        self.web_agent = WebSearchAgent(llm_model_name)

        # Build the graph
        graph = StateGraph(AgentState)

        graph.add_node("ingest_files", self._ingest_files)
        graph.add_node("build_retriever", self._build_retriever)
        graph.add_node("check_relevance", self._check_relevance)
        graph.add_node("answer_from_docs", self._answer_from_docs)
        graph.add_node("web_search_agent", self._web_search_agent)
        graph.add_node("synthesize_answer", self._synthesize_answer)

        graph.set_entry_point("ingest_files")
        graph.add_edge("ingest_files", "build_retriever")
        graph.add_edge("build_retriever", "check_relevance")

        # Conditional branch based on relevance
        def branch_on_relevance(state: AgentState) -> str:
            label = (state.get("relevance") or "NO_MATCH").upper()
            if label == "CAN_ANSWER":
                return "answer_from_docs"
            # both PARTIAL and NO_MATCH go to the web to enrich
            return "web_search_agent"

        graph.add_conditional_edges("check_relevance", branch_on_relevance, {
            "answer_from_docs": "answer_from_docs",
            "web_search_agent": "web_search_agent",
        })

        # After either path, synthesize
        graph.add_edge("answer_from_docs", "synthesize_answer")
        graph.add_edge("web_search_agent", "synthesize_answer")
        graph.add_edge("synthesize_answer", END)

        self.app = graph.compile()

    # ---- node fns ----
    def _ingest_files(self, state: AgentState) -> AgentState:
        files = state.get("files", [])
        paths = self.file_handler.process_documents(files)
        state["paths"] = [str(p) for p in paths if p]
        # Convert paths -> Documents (list[List[Document]] then flatten)
        chunks_nested = self.doc_processor.batch_process_files(paths)
        docs: List[Document] = []
        for lst in chunks_nested:
            docs.extend(lst or [])
        state["documents"] = docs
        logger.info(f"Ingested: {len(state['documents'])} chunks")
        return state

    def _build_retriever(self, state: AgentState) -> AgentState:
        docs = state.get("documents") or []
        state["retriever"] = self.retriever_builder.build_hybrid_retriever(docs) if docs else None
        return state

    def _check_relevance(self, state: AgentState) -> AgentState:
        retriever = state.get("retriever")
        q = state["question"]
        state["relevance"] = self.checker.check(q, retriever, k=3) if retriever else "NO_MATCH"
        logger.info(f"Relevance: {state['relevance']}")
        return state

    def _answer_from_docs(self, state: AgentState) -> AgentState:
        # Minimal—fetch top docs and stitch a draft. You can replace with a SynthesisAgent.
        retriever = state["retriever"]
        top_docs = retriever.invoke(state["question"]) if retriever else []
        ctx = "\n\n".join(d.page_content for d in top_docs[:5])
        state["final_answer"] = f"Answer based on uploaded documents:\n\n{ctx}"
        return state

    def _web_search_agent(self, state: AgentState) -> AgentState:
        q = state["question"]
        results = self.web_agent.search_papers(q)
        state["web_results"] = results
        return state

    def _synthesize_answer(self, state: AgentState) -> AgentState:
        """
        Combine doc evidence (if any) + web results into a single draft.
        Replace with your SynthesisAgent if you have one.
        """
        parts = []
        if state.get("relevance") == "CAN_ANSWER":
            parts.append(state.get("final_answer", ""))

        if state.get("web_results"):
            top = state["web_results"][:5]
            bullets = "\n".join(f"- {r.get('title','')} — {r.get('link','')}" for r in top)
            parts.append(f"Supplementary web findings:\n{bullets}")

        state["final_answer"] = "\n\n".join(p for p in parts if p) or "No answer found."
        return state

    # ---- public run ----
    def run(self, question: str, files: List[dict]) -> AgentState:
        """
        files: list like [{"type": "url"|"pdf", "value": "<...>"}]
        """
        initial: AgentState = {"question": question, "files": files}
        return self.app.invoke(initial)
