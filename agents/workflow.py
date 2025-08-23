# workflow_graph.py
from typing import TypedDict, List, Optional, Dict
import logging

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_ollama import ChatOllama

from document_processing.file_handler import FileHandler
from document_processing.document_processor import DocumentProcessor
from agents.retriever import RetrieverBuilder
from agents.synthesis_agent import DocSynthesisAgent       # <- derived class from BaseReflexAgent (retriever tool)
from agents.relevance_checker import RelevanceChecker
from agents.web_search_agent import WebSearchAgent         # <- derived class from BaseReflexAgent (web search tool)
from config import WEB_SEARCH_AGENT_K, VECTOR_SEARCH_K  # (kept import for config compatibility)


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
    doc_results: List[dict]


# -------- Orchestrator --------
class WorkflowGraph:
    def __init__(
        self,
        llm_model_name: str,
        embedding_model_name: str,
        chroma_dir: str,
        vector_k: int = 8,
        retriever_weights=(0.5, 0.5),
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        # Core utilities
        self.file_handler = FileHandler(logger=self.logger)
        self.doc_processor = DocumentProcessor(logger=self.logger)

        # Single LLM instance shared by all agents (must support tool-calling)
        self.llm = ChatOllama(model=llm_model_name, temperature=0)  # e.g., "llama3.1" or another tool-capable model

        # Builders / checkers
        self.retriever_builder = RetrieverBuilder(
            embedding_model_name=embedding_model_name,
            persist_dir=chroma_dir,
            k=vector_k,
            weights=retriever_weights,
            logger=self.logger,
        )
        self.relevance_checker = RelevanceChecker(model_name=llm_model_name, logger=self.logger)

        # Agents (DocSynthesisAgent is created after retriever exists)
        self.doc_synth_agent: Optional[DocSynthesisAgent] = None
        self.web_agent = WebSearchAgent(llm=self.llm, logger=self.logger)

        self.build_workflow()

    def build_workflow(self):
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

        graph.add_conditional_edges(
            "check_relevance",
            branch_on_relevance,
            {"answer_from_docs": "answer_from_docs", "web_search_agent": "web_search_agent"},
        )

        # After either path, synthesize
        graph.add_edge("answer_from_docs", "synthesize_answer")
        graph.add_edge("web_search_agent", "synthesize_answer")
        graph.add_edge("synthesize_answer", END)

        self.app = graph.compile()

    # ---- node fns ----
    def _ingest_files(self, state: AgentState) -> AgentState:
        files = state.get("files", [])
        state["paths"] = self.file_handler.process_documents(files)
        self.logger.info(f"Ingested paths: {state['paths']}")

        chunks_nested = self.doc_processor.batch_process_files(state["paths"])
        docs: List[Document] = []
        for lst in chunks_nested:
            docs.extend(lst or [])
        state["documents"] = docs
        self.logger.info(f"Ingested {len(state['documents'])} chunks")
        return state

    def _build_retriever(self, state: AgentState) -> AgentState:
        docs = state.get("documents") or []
        if docs:
            retriever = self.retriever_builder.build_hybrid_retriever(docs)
            state["retriever"] = retriever
            # Init DocSynthesisAgent with shared LLM + this retriever
            self.doc_synth_agent = DocSynthesisAgent(llm=self.llm, retriever=retriever, logger=self.logger)
        else:
            state["retriever"] = None
            self.doc_synth_agent = None
        return state

    def _check_relevance(self, state: AgentState) -> AgentState:
        retriever = state.get("retriever")
        q = state["question"]
        relevance = self.relevance_checker.check(q, retriever, k=3)
        state["relevance"] = (relevance or {}).get("label", "NO_MATCH")
        self.logger.info(f"Relevance: {state['relevance']}")
        return state

    def _answer_from_docs(self, state: AgentState) -> AgentState:
        if not self.doc_synth_agent:
            state["final_answer"] = state.get("final_answer", "") or "No documents available for answering."
            state["answer_source"] = "none"
            return state

        result = self.doc_synth_agent.run(state["question"])
        fa = (result.get("final_answer") or "").strip()
        if fa:
            state["final_answer"] = fa
            state["answer_source"] = "docs"
        # only set citations if we actually have some
        cits = result.get("citations") or []
        if cits:
            state["citations"] = cits
        state["doc_results"] = result.get("conversation", [])
        return state

    # --- in _web_search_agent: only overwrite fields when the web agent produced something ---
    def _web_search_agent(self, state: AgentState) -> AgentState:
        q = state["question"]
        result = self.web_agent.run(q)

        convo = result.get("conversation") or []
        if convo:
            state["web_results"] = convo

        fa = (result.get("final_answer") or "").strip()
        if fa:
            state["final_answer"] = fa
            state["answer_source"] = "web"

        cits = result.get("citations") or []
        if cits:
            state["citations"] = cits

        return state

    # --- in _synthesize_answer: prefer existing answer, add trace if present, never fall back to empty ---
    def _synthesize_answer(self, state: AgentState) -> AgentState:
        parts: List[str] = []
        src = state.get("answer_source", "")

        # If docs were sufficient, keep that body
        if state.get("relevance") == "CAN_ANSWER" and src in ("docs", "") and state.get("final_answer"):
            parts.append(state["final_answer"])
            if state.get("citations"):
                parts.append("Citations:\n" + "\n".join(f"- {c}" for c in state["citations"]))

        # If we went to the web (PARTIAL/NO_MATCH), include a compact trace + final
        if state.get("web_results"):
            parts.append("Web Agent Summary:")
            for step in state["web_results"]:
                action = step.get("action")
                if action == "respond":
                    queries = step.get("queries") or []
                    if queries:
                        parts.append(f"- Proposed follow-up queries: {', '.join(queries)}")
                elif action == "revise":
                    decision = step.get("decision", "end")
                    parts.append(f"- Revision decision: {decision}")
            if state.get("final_answer"):
                parts.append("\nWeb Agent Final Answer:\n" + state["final_answer"])
            if state.get("citations"):
                parts.append("Citations:\n" + "\n".join(f"- {c}" for c in state["citations"]))

        # Finalize with a safe fallback:
        synthesized = "\n\n".join(p for p in parts if p).strip()
        if synthesized:
            state["final_answer"] = synthesized
            return state

        # If we got here, keep any non-empty prior answer rather than dropping to "No answer found."
        prior = (state.get("final_answer") or "").strip()
        state["final_answer"] = prior or "No answer found."
        return state
    
    # ---- public run ----
    def run(self, question: str, files: List[dict]) -> AgentState:
        """
        files: list like [{"type": "url"|"pdf", "value": "<...>"}]
        """
        initial: AgentState = {"question": question, "files": files}
        return self.app.invoke(initial)
