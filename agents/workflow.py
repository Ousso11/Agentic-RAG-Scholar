# workflow_graph.py
from __future__ import annotations
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.schema import Document
import logging
from document_processing.file_handler import FileHandler
from document_processing.document_processor import DocumentProcessor
from agents.retriever import RetrieverBuilder
from agents.synthesis_agent import DocSynthesisAgent
from agents.relevance_checker import RelevanceChecker
from agents.web_search_agent import WebSearchAgent
from config import WEB_SEARCH_AGENT_K, VECTOR_SEARCH_K

# -------- State --------
class AgentState(TypedDict, total=False):
    question: str
    # input
    files: List[dict]                 # [{type: "url"|"pdf", value: "..."}]
    # artifacts
    paths: List[str]
    retriever: object
    documents: List[Document]
    relevance: str                    # CAN_ANSWER | PARTIAL | NO_MATCH
    web_results: List[dict]
    final_answer: str
    citations: List[str]

# -------- Orchestrator --------
class WorkflowGraph:
    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        chroma_dir: str,
        vector_k: int = 8,
        retriever_weights=(0.5, 0.5),
        logger: logging.Logger = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.file_handler = FileHandler(logger=self.logger)
        self.doc_processor = DocumentProcessor(logger=self.logger)
        self.doc_synth_agent = None
        self.retriever_builder = RetrieverBuilder(
            embedding_model_name=embedding_model_name,
            persist_dir=chroma_dir,
            k=vector_k,
            weights=retriever_weights,
            logger=self.logger
        )
        self.relevance_checker = RelevanceChecker(model_name=llm_model_name, logger=self.logger)
        self.web_agent = WebSearchAgent(llm_model_name, logger=self.logger)

        self.build_workflow()
        
    def build_workflow(self):
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
        state["paths"] = self.file_handler.process_documents(files)
        self.logger.info(f"Ingested: {state['paths']}")
        chunks_nested = self.doc_processor.batch_process_files(state["paths"])
        docs: List[Document] = []
        for lst in chunks_nested:
            docs.extend(lst or [])
        state["documents"] = docs
        self.logger.info(f"Ingested: {len(state['documents'])} chunks")
        # for doc in docs:
        #     self.logger.info(f" - {doc.metadata.get('title', 'Untitled')}")
        return state

    def _build_retriever(self, state: AgentState) -> AgentState:
        docs = state.get("documents") or []
        if docs:
            retriever = self.retriever_builder.build_hybrid_retriever(docs)
            state["retriever"] = retriever
            # init doc synthesis agent with same LLM as relevance checker
            self.doc_synth_agent = DocSynthesisAgent(self.relevance_checker.llm, retriever, logger=self.logger)
        else:
            state["retriever"] = None
        return state

    def _check_relevance(self, state: AgentState) -> AgentState:
        retriever = state.get("retriever")
        q = state["question"]
        relevance = self.relevance_checker.check(q, retriever, k=3)
        state["relevance"] = relevance["label"]  # fix: use label not just can_answer
        self.logger.info(f"Relevance: {relevance}")
        return state

    def _answer_from_docs(self, state: AgentState) -> AgentState:
        if not self.doc_synth_agent:
            state["final_answer"] = "No documents available for answering."
            return state

        result = self.doc_synth_agent.run(state["question"])
        state["final_answer"] = result["final_answer"]
        state["citations"] = result.get("citations", [])
        state["doc_results"] = result.get("conversation", [])
        return state


    def _web_search_agent(self, state: AgentState) -> AgentState:
        q = state["question"]
        result = self.web_agent.run(q)
        # store the full output
        state["web_results"] = result.get("conversation", [])
        state["citations"] = result.get("citations", [])
        # also pass along the agent's own final answer
        state["final_answer"] = result.get("final_answer")
        return state

    def _synthesize_answer(self, state: AgentState) -> AgentState:
        parts = []
        if state.get("relevance") == "CAN_ANSWER":
            parts.append(state.get("final_answer", ""))

        if state.get("web_results"):
            # summarize web agent’s results
            parts.append("Web Agent Conversation:")
            for step in state["web_results"]:
                action = step.get("action")
                if action == "search":
                    parts.append(f"- Search: {step.get('query')}")
                elif action == "respond":
                    parts.append(f"- Initial Answer: {step['answer'].get('answer')}")
                elif action == "revise":
                    parts.append(f"- Revised Answer: {step['revised'].get('answer')}")
            if state.get("citations"):
                parts.append("Citations:\n" + "\n".join(f"- {c}" for c in state["citations"]))

        state["final_answer"] = "\n\n".join(p for p in parts if p) or "No answer found."

        return state


    # ---- public run ----
    def run(self, question: str, files: List[dict]) -> AgentState:
        """
        files: list like [{"type": "url"|"pdf", "value": "<...>"}]
        """
        initial: AgentState = {"question": question, "files": files}
        return self.app.invoke(initial)
