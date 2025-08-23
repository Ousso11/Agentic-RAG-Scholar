import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, TypedDict, Tuple

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


# ------------------ Structured Outputs ------------------
class RespondOutput(BaseModel):
    answer: str
    feedback: str
    search_queries: List[str] = Field(default_factory=list)


class ReviseOutput(BaseModel):
    answer: str
    feedback: str
    action: Literal["reconsider", "end"]
    new_queries: List[str] = Field(default_factory=list)


# ------------------ Abstract Reflex Agent ------------------
class BaseReflexAgent(ABC):
    class State(TypedDict, total=False):
        question: str
        iteration: int
        conversation: List[Dict]
        draft: RespondOutput
        revised: ReviseOutput
        feedback: str
        queries: List[str]

    def __init__(self, llm: BaseChatModel, max_iters: int = 3, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.max_iters = max_iters
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.responder_prompt, self.revision_prompt = self._build_prompts()
        self.external_tools = self._build_tools()  # e.g., retrievers/search

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

    # ----- nodes -----
    def node_respond(self, state: State) -> State:
        """
        Respond node: fetch documents using queries, call LLM once, return answer + feedback.
        """
        question = state["question"]
        queries = state.get("queries") or [question]
        feedback = state.get("feedback") or ""

        # ---------------- Fetch documents ----------------
        fetched_docs = []
        for q in queries:
            for tool in self.external_tools:
                result = tool.invoke({"query": q})
                fetched_docs.append(result)

        # ---------------- Call LLM ----------------
        context_msg = "\n".join(str(d) for d in fetched_docs)
        prompt_msg = f"Question: {question}\nContext:\n{context_msg}\nFeedback:\n{feedback}"
        msgs = self.responder_prompt.format_messages(
            first_instruction="Answer concisely then expand.",
            history=[HumanMessage(content=prompt_msg)]
        )
        resp = self.llm.invoke(msgs)

        # Parse structured RespondOutput
        draft = RespondOutput(
            answer=getattr(resp, "answer", getattr(resp, "content", "")),
            feedback=getattr(resp, "feedback", ""),
            search_queries=queries
        )

        conv = state.get("conversation", [])
        conv.append({"action": "respond", "queries": draft.search_queries})
        self.logger.debug(f"Respond output: {draft.answer[:200]} | feedback: {draft.feedback}")
        return {**state, "draft": draft, "conversation": conv}

    def node_revise(self, state: State) -> State:
        """
        Revise node: only sees draft answer + feedback, decides action and optional new queries.
        """
        draft: RespondOutput = state["draft"]
        feedback_msg = f"Draft answer:\n{draft.answer}\nSelf-feedback:\n{draft.feedback}"
        msgs = self.revision_prompt.format_messages(
            first_instruction="Critique, revise, and decide: reconsider or end.",
            history=[HumanMessage(content=feedback_msg)]
        )
        resp = self.llm.invoke(msgs)

        # Parse structured ReviseOutput
        revised = ReviseOutput(
            answer=getattr(resp, "answer", draft.answer),
            feedback=getattr(resp, "feedback", ""),
            action=getattr(resp, "action", "end"),
            new_queries=getattr(resp, "new_queries", [])
        )

        i = state.get("iteration", 0) + 1
        conv = state.get("conversation", [])
        conv.append({"action": "revise", "decision": revised.action, "new_queries": revised.new_queries})
        self.logger.debug(f"Revise output: {revised.answer[:200]} | feedback: {revised.feedback} | action: {revised.action}")
        return {**state, "revised": revised, "iteration": i, "conversation": conv, "feedback": revised.feedback, "queries": revised.new_queries}

    # ----- controller -----
    def _should_continue(self, state: State) -> Literal["respond", "end"]:
        if state.get("iteration", 0) >= self.max_iters:
            return "end"
        revised = state.get("revised")
        if revised and getattr(revised, "action", None) == "reconsider":
            return "respond"
        return "end"

    # ----- API -----
    def run(self, question: str) -> Dict:
        init: BaseReflexAgent.State = {
            "question": question,
            "iteration": 0,
            "conversation": [],
            "queries": [question],
            "feedback": ""
        }
        final = self.graph.invoke(init)
        final_answer = (final.get("revised") and final["revised"].answer) or (final.get("draft") and final["draft"].answer) or ""
        return {
            "question": question,
            "final_answer": final_answer,
            "conversation": final.get("conversation", []),
            "iterations": final.get("iteration", 0),
        }
