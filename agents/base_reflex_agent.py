import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Tuple

from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage


# ------------------ Structured Outputs ------------------
class RespondOutput(BaseModel):
    answer: str
    feedback: str
    search_queries: List[str] = Field(default_factory=list)


class ReviseOutput(BaseModel):
    answer: str
    feedback: str
    action: Literal["reconsider", "end"]
    additional_queries: List[str] = Field(default_factory=list)


# ------------------ BaseReflexAgent ------------------
class BaseReflexAgent(ABC):
    """
    Reflex agent using LangGraph's MessagesState (built-in chat memory).
    Loop: respond -> revise -> (reconsider ? respond : end)
    """

    class State(MessagesState):  # inherits `messages: List[BaseMessage]`
        iteration: int
        draft: Optional[RespondOutput]
        revised: Optional[ReviseOutput]
        feedback: Optional[str]
        queries: List[str]
        trace: List[dict]

    def __init__(self, llm: BaseChatModel, max_iters: int = 3, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.max_iters = max_iters
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # subclass-provided
        self.responder_prompt, self.revision_prompt = self._build_prompts()
        self.external_tools = self._build_tools()

        # Chains
        self.respond_chain = self.responder_prompt | self.llm.with_structured_output(RespondOutput)
        self.revise_chain = self.revision_prompt | self.llm.with_structured_output(ReviseOutput)

        # Graph wiring
        g = StateGraph(self.State)
        g.add_node("respond", self.node_respond)
        g.add_node("revise", self.node_revise)
        g.set_entry_point("respond")
        g.add_edge("respond", "revise")
        g.add_conditional_edges("revise", self._should_continue, {"respond": "respond", "end": END})
        self.graph = g.compile()

    # ----- subclass contract -----
    @abstractmethod
    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        ...

    @abstractmethod
    def _build_tools(self) -> List:
        ...

    # ----- RESPOND -----
    def node_respond(self, state: State) -> State:
        question = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
        if not question:
            raise ValueError("No HumanMessage (question) found in history")
        self.logger.info(f"Responding to question: {question}")
        queries = state.get("queries") or [question]
        prev_feedback = state.get("feedback") or ""

        # 1) Fetch documents
        fetched_blobs: List[str] = []
        for q in queries:
            for tool in self.external_tools:
                try:
                    out = tool.invoke({"query": q})
                except Exception as e:
                    self.logger.exception(f"Tool {getattr(tool, 'name', tool)} failed for query='{q}': {e}")
                    out = f"(tool-error for query='{q}': {e})"
                fetched_blobs.append(str(out) if out is not None else "")
        context = "\n\n---\n\n".join(x for x in fetched_blobs if x).strip()

        # 2) Call LLM
        try:
            draft: RespondOutput = self.respond_chain.invoke({
                "first_instruction": "Answer concisely first, then expand if needed.",
                "question": question,
                "context": context or "(no external context)",
                "feedback": prev_feedback or "(none)",
                "queries": ", ".join(queries),
                "history": state["messages"],  # works since MessagesState ensures BaseMessage[]
            })
        except (ValidationError, Exception) as e:
            self.logger.error(f"Respond chain failed: {e}")
            draft = RespondOutput(answer="", feedback=str(e), search_queries=list(queries))

        self.logger.info(f"Draft answer generated: {draft.answer}, Feedback: {draft.feedback}, Queries: {queries}")

        # record trace
        trace_entry = {
            "iteration": state.get("iteration", 0) + 1,
            "action": "respond",
            "content": draft.answer,
            "queries": draft.search_queries or [],
            "feedback": draft.feedback,
        }

        return {
            **state,
            "draft": draft,
            "messages": state["messages"] + [AIMessage(content=draft.answer)],
            "trace": (state.get("trace") or []) + [trace_entry],
        }

    # ----- REVISE -----
    def node_revise(self, state: State) -> State:
        draft: RespondOutput = state["draft"]
        question = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
        self.logger.info(f"Revising draft answer for question: {question}")
        prior_queries = (draft.search_queries or state.get("queries") or [question])

        try:
            revised: ReviseOutput = self.revise_chain.invoke({
                "first_instruction": "Critique, refine, then decide reconsider/end.",
                "question": question,
                "draft_answer": draft.answer,
                "draft_feedback": draft.feedback or "(none)",
                "prior_queries": ", ".join(prior_queries),
                "history": state["messages"],
            })
        except (ValidationError, Exception) as e:
            self.logger.error(f"Revise chain failed: {e}")
            revised = ReviseOutput(
                answer=draft.answer,
                feedback=str(e),
                action="end",
                additional_queries=[],
            )

        i = state.get("iteration", 0) + 1
        next_queries = list(revised.additional_queries) if revised.action == "reconsider" else []

        self.logger.info(f"Revision iteration {i} complete. Action: {revised.action}, Next queries: {next_queries}")
        self.logger.debug(f"Revised answer: {revised.answer}, Feedback: {revised.feedback}")

        # record trace
        trace_entry = {
            "iteration": i,
            "action": "revise",
            "content": revised.answer,
            "decision": revised.action,
            "feedback": revised.feedback,
            "queries": revised.additional_queries or [],
            "refs": getattr(revised, "citations", []),
        }

        return {
            **state,
            "revised": revised,
            "iteration": i,
            "feedback": revised.feedback,
            "queries": next_queries if next_queries else state.get("queries", []),
            "messages": state["messages"] + [AIMessage(content=f"[Revise decision: {revised.action}] {revised.answer}")],
            "trace": (state.get("trace") or []) + [trace_entry],
        }

    def _should_continue(self, state: State) -> Literal["respond", "end"]:
        if state.get("iteration", 0) >= self.max_iters:
            return "end"
        revised = state.get("revised")
        if revised and getattr(revised, "action", None) == "reconsider":
            return "respond"
        return "end"

    # ----- API -----
    def run(self, question: str):
        init = self.State(
            messages=[HumanMessage(content=question)],
            iteration=0,
            queries=[question],
            feedback="",
            trace=[],
        )
        final = self.graph.invoke(init)

        # Collect final answer
        final_answer = (
            (final.get("revised") and final["revised"].answer)
            or (final.get("draft") and final["draft"].answer)
            or ""
        )

        conversation = final.get("trace", [])
        citations: List[str] = []
        for step in conversation:
            if step.get("refs"):
                citations.extend(step["refs"])

        return {
            "question": question,
            "final_answer": final_answer,
            "iterations": final.get("iteration", 0),
            "messages": final.get("messages", []),     # raw AI/Human messages
            "conversation": conversation,              # structured trace (all rounds!)
            "citations": citations,                    # all refs
        }
